"""Coaching: read the prompt before it runs, and turn recurring waste into lessons.

Two jobs:

1. ``coach_prompt``: at UserPromptSubmit, classify the task, recommend a model tier,
   flag prompt patterns that burn tokens, and say STOP when the session is in the red.
   Rule-based, instant, free. The model-assisted version lives in analyze.py.

2. ``habits``: over a window, group waste into named habits with counts, week-over-week
   trend, avoidable tokens, and the exact sentence the person should say next time.
   This is the learning curve a team looks at on Monday.

3. ``cache_resets``: turns where the cached prefix was dropped and re-written instead of
   re-read. Unlike the rest of this module that is an observation, not a counterfactual:
   both token classes are present in the transcript on adjacent turns.

4. ``compaction_breakeven``: how many more turns it takes for compacting now to pay back the
   cache write it forces. A projection from published prices, not a measured saving.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

from .aggregate import Aggregates
from .config import PriceTable, TokenPrices, load_published_prices
from .findings import Thresholds
from .model import DIAGNOSTIC_COMMAND, IMAGE_SUFFIXES, Session, Turn

TIER_FAST, TIER_STANDARD, TIER_PREMIUM = "haiku", "sonnet", "opus"

# Candidate tiers to evaluate; classification does not establish equivalent quality.
TASK_TIER: dict[str, str] = {
    "lookup": TIER_FAST,
    "research": TIER_STANDARD,
    "summarize": TIER_STANDARD,
    "small-edit": TIER_STANDARD,
    "tests": TIER_STANDARD,
    "docs": TIER_STANDARD,
    "debug": TIER_PREMIUM,
    "design": TIER_PREMIUM,
    "general": TIER_STANDARD,
}
TASK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("lookup", re.compile(r"\b(where is|find|grep|search for|which file|list all|what does .* return|look up|locate)\b", re.I)),
    ("research", re.compile(r"\b(research|compare|what are others|survey|state of the art|options for|pros and cons|deep dive)\b", re.I)),
    ("summarize", re.compile(r"\b(summari[sz]e|explain (this|the)|walk me through|what is going on in|describe)\b", re.I)),
    ("tests", re.compile(r"\b(write|add|fix) (the |a |unit |integration )?tests?\b|\bpytest\b|\bfailing test", re.I)),
    ("docs", re.compile(r"\b(readme|docs?|documentation|docstring|changelog|comment)\b", re.I)),
    ("small-edit", re.compile(r"\b(rename|typo|add (a |the )?(flag|option|field|param|log)|bump|update the (version|import)|small change|one[- ]line)\b", re.I)),
    ("debug", re.compile(r"\b(bugs?|why (does|is|did)|broken|crash(es|ed)?|traceback|exceptions?|not working|fails?|failing|flaky|race|deadlock|wrong (output|result))\b", re.I)),
    ("design", re.compile(r"\b(architect|design|refactor|restructure|migrate|rewrite|plan (out|the)|trade-?offs?|scal(e|ing))\b", re.I)),
)
WASTE_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("whole-repo", re.compile(r"\b(whole|entire|all( the)?|every) (repo|codebase|project|files?|directory|folder)\b|\bread everything\b|\bcheck (any|all) bugs\b", re.I),
     "Asking for the whole repo makes the agent read everything into context. Name the two or three files, or the symptom."),
    ("full-output", re.compile(r"\b(full|complete|entire|all) (logs?|output|trace|stack ?trace|diff|history)\b|\bshow me everything\b|\bcat\b", re.I),
     "Full logs and diffs land in context and get re-read every turn. Ask for the last 50 lines or the failing test only."),
    ("vague", re.compile(r"^(fix it|make it work|improve|clean ?up|check|look at this|do it|continue|go on|try again)[.! ]*$", re.I),
     "A vague prompt makes the agent explore. Say what changed, which file, and what 'done' looks like."),
    ("fan-out", re.compile(r"\b(spawn|fan out|parallel|multiple|several|many) (sub)?agents?\b|\bdeep research\b|\bresearch (everything|all)\b", re.I),
     "Parallel agents can repeat context. Give each agent a distinct scope and check whether their results justify the extra work."),
)
FILE_HINT = re.compile(r"[\w./-]+\.(py|ts|tsx|js|go|rs|java|md|yaml|yml|json|toml|sql|html|css)\b|\bline[s]? \d+", re.I)
QUOTE_LIMITER = re.compile(
    r"\b(?:only the fail|just the fail|failing (?:test|assertion|line)s?|first error|tail|head|last \d+ lines|summar)"
    r"|(?<![\w-])-(?:q|-quiet)\b",
    re.I,
)


@dataclass
class Coaching:
    task: str
    recommended_tier: str
    current_model: str
    zone: str
    context_now: int
    stop: bool
    messages: list[str] = field(default_factory=list)   # shown to the human
    agent_notes: list[str] = field(default_factory=list)  # injected as additionalContext for the agent
    rewrite: str = ""
    compaction: "CompactionEconomics | None" = None

    def as_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            "recommended_tier": self.recommended_tier,
            "current_model": self.current_model,
            "zone": self.zone,
            "context_now": self.context_now,
            "stop": self.stop,
            "messages": list(self.messages),
            "agent_notes": list(self.agent_notes),
            "rewrite": self.rewrite,
            "compaction": self.compaction.as_dict() if self.compaction else None,
        }


def classify_task(prompt: str) -> str:
    text = prompt.strip()
    if not text:
        return "general"
    matches = {kind for kind, pattern in TASK_PATTERNS if pattern.search(text)}
    for complex_kind in ("debug", "design"):
        if complex_kind in matches:
            return complex_kind
    if len(matches) == 1:
        return next(iter(matches))
    return "general"


@dataclass
class CacheReset:
    """A turn that re-wrote the cached prefix instead of reading it."""

    turn_index: int
    at: datetime
    rewritten: int
    previously_read: int
    from_model: str
    to_model: str

    @property
    def cause(self) -> str:
        """Only the model is recorded per turn; effort and mode changes look identical here."""
        if self.from_model and self.to_model and self.from_model != self.to_model:
            return f"model changed {self.from_model} -> {self.to_model}"
        return "cause not recorded (effort or mode change, or the cache expired)"

    def as_dict(self) -> dict[str, object]:
        return {
            "turn_index": self.turn_index,
            "at": self.at.isoformat(),
            "rewritten": self.rewritten,
            "previously_read": self.previously_read,
            "from_model": self.from_model,
            "to_model": self.to_model,
            "cause": self.cause,
        }


def cache_resets(turns: list[Turn], th: Thresholds) -> list[CacheReset]:
    """Turns that paid cache-write on a prefix the turn before had read from cache.

    Switching model, effort or mode mid-session drops the cached prefix, so the next turn
    re-sends the whole conversation as a cache write. The token count does not change; the
    class does, and the classes are priced differently.
    """
    baseline = _median_positive([t.usage.cache_creation_input_tokens for t in turns])
    out: list[CacheReset] = []
    for index in range(1, len(turns)):
        previous, turn = turns[index - 1], turns[index]
        written = turn.usage.cache_creation_input_tokens
        if written < th.cache_reset_min_tokens or written < baseline * th.cache_reset_spike_ratio:
            continue
        # without this the first write of any prefix looks like a reset, including every session start
        if previous.usage.cache_read_input_tokens < th.cache_reset_min_tokens:
            continue
        out.append(
            CacheReset(
                turn_index=index,
                at=turn.timestamp,
                rewritten=written,
                previously_read=previous.usage.cache_read_input_tokens,
                from_model=previous.model,
                to_model=turn.model,
            )
        )
    return out


def _median_positive(values: list[int]) -> float:
    positive = sorted(v for v in values if v > 0)
    return float(positive[len(positive) // 2]) if positive else 0.0


def context_growth(turns: list[Turn]) -> float | None:
    """Mean per-turn context increase; negative steps are compactions, not growth."""
    deltas = [b.usage.context - a.usage.context for a, b in zip(turns, turns[1:], strict=False) if b.usage.context > a.usage.context]
    return sum(deltas) / len(deltas) if deltas else None


@lru_cache(maxsize=1)
def _published_prices() -> PriceTable:
    return load_published_prices()


@dataclass
class CompactionEconomics:
    """Requests until compacting now repays the cache write it forces."""

    context_now: int
    retained_tokens: int
    saved_per_request: int
    cache_write_read_ratio: float
    one_off_read_equivalents: int
    breakeven_requests: float
    window_headroom_requests: int | None
    worth_it: bool | None
    savings_status: str = "projected"
    basis: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "context_now": self.context_now,
            "retained_tokens": self.retained_tokens,
            "saved_per_request": self.saved_per_request,
            "cache_write_read_ratio": round(self.cache_write_read_ratio, 2),
            "one_off_read_equivalents": self.one_off_read_equivalents,
            "breakeven_requests": round(self.breakeven_requests, 1),
            "window_headroom_requests": self.window_headroom_requests,
            "worth_it": self.worth_it,
            "savings_status": self.savings_status,
            "basis": self.basis,
        }


def compaction_breakeven(
    context_now: int,
    prices: TokenPrices,
    th: Thresholds,
    growth_per_turn: float | None = None,
) -> CompactionEconomics | None:
    """How many more requests compacting has to survive to pay for itself. None when too small to matter.

    Compacting drops the cached prefix, so the next request re-writes a short one at cache-write
    price and the summarising call reads the old context once. Both are one-off; the saving is
    per request from then on.
    """
    retained = th.compaction_retained_tokens + th.compaction_summary_tokens
    saved = context_now - retained
    if context_now < th.compaction_min_context or saved <= 0 or prices.cache_read <= 0:
        return None
    ratio = prices.cache_write / prices.cache_read
    one_off = (ratio - 1) * retained + context_now + (prices.output / prices.cache_read) * th.compaction_summary_tokens
    breakeven = one_off / saved
    headroom: int | None = None
    if growth_per_turn and growth_per_turn > 0 and th.context_window_tokens > context_now:
        headroom = int((th.context_window_tokens - context_now) // growth_per_turn)
    return CompactionEconomics(
        context_now=context_now,
        retained_tokens=retained,
        saved_per_request=saved,
        cache_write_read_ratio=ratio,
        one_off_read_equivalents=int(one_off),
        breakeven_requests=breakeven,
        window_headroom_requests=headroom,
        worth_it=None if headroom is None else headroom >= breakeven,
        basis=(
            f"Projection from published prices ({ratio:.1f}x cache write over cache read), a retained tail of "
            f"{th.compaction_retained_tokens:,} and a {th.compaction_summary_tokens:,}-token summary. "
            "The tail and summary sizes are assumptions, not measurements of this agent, and context is "
            "held flat across the projected turns, so a growing session breaks even sooner than this."
        ),
    )


def coach_prompt(
    prompt: str,
    context_now: int,
    current_model: str,
    th: Thresholds,
    task_tiers: dict[str, str] | None = None,
    waste: list[str] | None = None,
    resets: list[CacheReset] | None = None,
    prices: PriceTable | None = None,
    growth_per_turn: float | None = None,
) -> Coaching:
    """Instant, rule-based coaching for one prompt about to be submitted.

    ``waste`` is concrete recent waste (from the transcript tail). Red context plus waste is a
    STOP. Red context alone is a cost note: mid-feature work legitimately carries context, and
    the fix is a handoff at the next natural break, never a restart per prompt.
    """
    task = classify_task(prompt)
    tier = {**TASK_TIER, **(task_tiers or {})}.get(task, TIER_STANDARD)
    if task in {"debug", "design", "general"}:
        tier = current_model or "keep current model"
    zone = _zone(context_now, th)
    coaching = Coaching(task=task, recommended_tier=tier, current_model=current_model, zone=zone, context_now=context_now, stop=False)
    table = prices or _published_prices()
    coaching.compaction = compaction_breakeven(context_now, table.for_model(current_model), th, growth_per_turn)

    if zone == "red" and waste:
        coaching.stop = True
        coaching.messages.append(
            f"STOP: {context_now:,} tokens of context, re-read every turn, and the last turns show waste ({'; '.join(waste[:2])}). "
            "Finish this step, run `burnlens handoff`, paste the brief into a new session. The brief carries the context; you are not starting over."
        )
    elif zone == "red":
        coaching.messages.append(
            f"Expensive, not wrong: context is {context_now:,} tokens, so every turn now costs about {context_now // 1000}k. "
            + _compaction_sentence(coaching.compaction)
        )

    if th.is_premium(current_model) and tier != current_model and task not in {"debug", "design", "general"}:
        coaching.messages.append(
            f"This reads as a {task} task. {tier.capitalize()} is a candidate to evaluate on representative tasks; quality and savings have not been measured. Keep the current model until validated."
        )
        coaching.agent_notes.append(f"Burnlens: candidate model='{tier}' for {task}; evaluate task success and rework before changing models. Preserve the current model for this run.")

    for name, pattern, advice in WASTE_PATTERNS:
        if pattern.search(prompt):
            coaching.messages.append(advice)
            if name == "whole-repo":
                coaching.agent_notes.append("Burnlens: do not read the whole repository; grep for symbols and Read line ranges.")
            if name == "full-output":
                coaching.agent_notes.append("Burnlens: keep shell output under 50 lines (tail/grep); never cat whole logs.")
            if name == "fan-out":
                coaching.agent_notes.append("Burnlens: keep delegated tasks distinct and avoid duplicate investigations; preserve the selected models.")

    if DIAGNOSTIC_COMMAND.search(prompt) and not QUOTE_LIMITER.search(prompt):
        coaching.messages.append(
            "Build and test runners print long logs, and the log is then re-read on every later turn. "
            "Ask for the failing lines quoted back, not the log."
        )
        coaching.agent_notes.append(
            "Burnlens: run the build or test command quietly and quote only the first fatal line and the failing assertions; do not paste the whole log."
        )

    if not FILE_HINT.search(prompt) and task in {"debug", "small-edit", "tests", "docs"} and len(prompt) < 400:
        coaching.messages.append("No file or line range named. Naming it saves the agent a search pass through context.")

    if resets:
        latest = resets[-1]
        coaching.messages.append(
            f"Cache reset at turn {latest.turn_index}: {latest.rewritten:,} tokens were re-written that the turn before read from cache "
            f"({latest.cause}). Same tokens, cache-write class instead of cache-read. "
            "Pick the model and effort at the start of a session; change them at a handoff, not mid-thread."
        )
        coaching.agent_notes.append("Burnlens: the cached prefix was dropped in this session; keep the model and effort as they are until a handoff.")

    substantive = [m for m in coaching.messages if not m.startswith("Expensive")]
    if substantive and task != "general":
        coaching.rewrite = _rewrite(prompt, task, tier)
    return coaching


def _compaction_sentence(econ: CompactionEconomics | None) -> str:
    """The handoff advice, with the payback turn count when prices allow one."""
    tail = "Keep going if you are mid-feature. At the next natural break, hand off to a new session; the brief carries the context."
    if econ is None:
        return tail
    turns = math.ceil(econ.breakeven_requests)
    lead = (
        f"Compacting pays for itself after about {turns} more turn{'' if turns == 1 else 's'} "
        f"(one cache write now, then about {econ.saved_per_request // 1000}k less to read every turn). "
    )
    if econ.window_headroom_requests is not None:
        lead += f"At the current growth rate you have room for about {econ.window_headroom_requests} more turns before the window fills. "
    return lead + tail


def _rewrite(prompt: str, task: str, tier: str) -> str:
    head = prompt.strip().rstrip(".")
    scope = {
        "lookup": "Answer from a grep; do not read whole files.",
        "research": "Give each research task a distinct scope and return evidence with links; preserve the selected models.",
        "summarize": "Read only the files I name, line ranges where possible; 200 words.",
        "small-edit": "Edit only the file I name; show the diff, not the file.",
        "tests": "Run pytest -q and show only failures; edit only the test file and the function under test.",
        "docs": "Edit only the doc I name; no code reads beyond the public signatures.",
        "debug": "Start from the failing line I paste; read line ranges around it, not whole files; tail logs to 50 lines.",
        "design": "Write the plan first (under 400 words) and wait for my go before editing anything.",
    }.get(task, "Keep reads to line ranges and shell output short.")
    return f"{head}. Files: <name them>. {scope}"


@dataclass
class Habit:
    key: str
    title: str
    this_week: int
    last_week: int
    avoidable_tokens: int | None
    say_this: str
    examples: list[str] = field(default_factory=list)
    savings_status: str = "unmeasured"
    savings_basis: str = "No matched before/after measurement; potential savings are unknown."

    @property
    def trend(self) -> str:
        if self.last_week == 0:
            return "new" if self.this_week else "seen"
        change = (self.this_week - self.last_week) / self.last_week
        return "worse" if change > 0.15 else "better" if change < -0.15 else "same"

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "title": self.title,
            "this_week": self.this_week,
            "last_week": self.last_week,
            "trend": self.trend,
            "avoidable_tokens": self.avoidable_tokens,
            "savings_status": self.savings_status,
            "savings_basis": self.savings_basis,
            "say_this": self.say_this,
            "examples": list(self.examples),
        }


def habits(sessions: list[Session], agg: Aggregates, th: Thresholds, now: datetime | None = None, user: str | None = None) -> list[Habit]:
    """Recurring waste as lessons, with week-over-week trend and avoidable tokens. ``user`` narrows to one person."""
    now = now or datetime.now(timezone.utc)
    if user:
        keep = {s.session_id for s in sessions if s.user == user}
        sessions = [s for s in sessions if s.session_id in keep or (s.parent_session_id in keep)]
    week_ago = now - timedelta(days=7)
    two_weeks = now - timedelta(days=14)
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [this_week, last_week]
    examples: dict[str, list[str]] = defaultdict(list)

    def bump(key: str, ts: datetime, example: str | None = None) -> None:
        if ts < two_weeks or ts > now:
            return
        if ts >= week_ago:
            counts[key][0] += 1
        elif ts >= two_weeks:
            counts[key][1] += 1
        if example and len(examples[key]) < 3 and example not in examples[key]:
            examples[key].append(example)

    for session in sessions:
        if session.is_subagent:
            continue
        seen: set[tuple[str, str, str, str]] = set()
        for turn in session.turns:
            if turn.usage.context > th.context_tokens:
                bump("bloat", turn.timestamp)
            for call in turn.tool_calls:
                if call.name == "Read" and call.file_path:
                    read_key = (call.file_path, str(call.input.get("offset", "")), str(call.input.get("limit", "")), str(call.input.get("pages", "")))
                    if read_key in seen:
                        key = "image-reread" if Path(call.file_path).suffix.lower() in IMAGE_SUFFIXES else "reread"
                        bump(key, turn.timestamp, Path(call.file_path).name)
                    seen.add(read_key)
                if call.result_bytes > th.large_payload_bytes:
                    bump("payload", turn.timestamp, (call.command_head or call.name)[:60])
                if call.name == "Agent":
                    kind = classify_task(str(call.input.get("prompt") or call.input.get("description") or ""))
                    model = str(call.input.get("model") or "")
                    inherits_premium = (not model and th.is_premium(turn.model)) or th.is_premium(model)
                    if inherits_premium and kind not in {"general", "debug", "design"}:
                        bump("premium-subagent", turn.timestamp, f"{kind}: {str(call.input.get('description') or '')[:50]}")
    catalogue = {
        "bloat": ("Kept going past the context threshold", "Say: 'that's enough for this session' and paste the handoff brief into a new one."),
        "reread": ("Repeated file reads to review", "Check edits and requested ranges; reuse prior results only when they still answer the task."),
        "image-reread": ("Repeated image reads to review", "Check whether the image or question changed before treating another read as redundant."),
        "payload": ("Let commands dump big output into context", "Say: 'tail -50' or 'show only the failures'."),
        "premium-subagent": ("Ran lookup or research subagents on the premium model", "Evaluate a lower-cost model on representative tasks before changing the selected model."),
    }
    out: list[Habit] = []
    for key, (title, say) in catalogue.items():
        this_week, last_week = counts.get(key, [0, 0])
        if this_week == 0 and last_week == 0:
            continue
        out.append(Habit(key=key, title=title, this_week=this_week, last_week=last_week, avoidable_tokens=None, say_this=say, examples=examples.get(key, [])))
    out.sort(key=lambda h: (h.this_week, h.last_week), reverse=True)
    return out


def _zone(context_now: int, th: Thresholds) -> str:
    if context_now >= th.context_tokens * th.live_context_high_multiplier:
        return "red"
    if context_now >= th.context_tokens:
        return "amber"
    return "green"
