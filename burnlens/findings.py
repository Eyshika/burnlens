"""Turn aggregates into findings: what wasted tokens, with evidence and a fix."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .aggregate import Aggregates
from .model import IMAGE_SUFFIXES

PREMIUM_MODEL_MARKERS: tuple[str, ...] = ("opus", "fable", "mythos")
RULE_NAMES: tuple[str, ...] = ("context-bloat", "long-session", "repeated-reads", "image-rereads", "large-payloads", "subagent-premium-model", "unattended-runaway", "edit-then-verify")


@dataclass(frozen=True)
class Thresholds:
    """Every knob the rules use. Override from the CLI, never inline."""

    context_tokens: int = 150_000
    bloat_share: float = 0.5
    long_session_turns: int = 500
    repeated_read_count: int = 10
    image_reread_count: int = 2
    large_payload_bytes: int = 50_000
    large_payload_high_total_bytes: int = 10_000_000
    bloat_min_turns: int = 20
    subagent_premium_share: float = 0.5
    subagent_min_tokens: int = 1_000_000
    loop_repeat_count: int = 3               # identical calls in a row before the next one is refused
    cache_reset_min_tokens: int = 20_000     # a re-written prefix smaller than this is not worth a word
    cache_reset_spike_ratio: float = 4.0     # vs the session's own median cache write
    context_window_tokens: int = 1_000_000   # only used for the compaction headroom line
    compaction_retained_tokens: int = 20_000   # tail the agent keeps; unverified for Claude Code
    compaction_summary_tokens: int = 1_000     # size of the summary compaction writes; unverified
    compaction_min_context: int = 50_000
    fuse_pair_min_count: int = 5
    max_examples: int = 5
    live_active_minutes: int = 10
    live_window_minutes: int = 5
    live_burn_warn_per_min: int = 1_000_000
    live_burn_high_per_min: int = 3_000_000
    live_context_high_multiplier: float = 2.0
    workflow_runaway_ratio: float = 2.0  # latest run vs the workflow's median
    workflow_min_runs: int = 3
    premium_markers: tuple[str, ...] = PREMIUM_MODEL_MARKERS

    def is_premium(self, model: str) -> bool:
        lowered = (model or "").lower()
        return any(marker in lowered for marker in self.premium_markers)


@dataclass
class Finding:
    rule: str
    severity: str  # "high" | "medium" | "low"
    title: str
    evidence: list[str] = field(default_factory=list)
    suggestion: str = ""
    avoidable_tokens: int | None = None
    savings_status: str = "unmeasured"
    savings_basis: str = "No matched before/after measurement; potential savings are unknown."

    def as_dict(self) -> dict[str, object]:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "title": self.title,
            "evidence": list(self.evidence),
            "suggestion": self.suggestion,
            "avoidable_tokens": self.avoidable_tokens,
            "savings_status": self.savings_status,
            "savings_basis": self.savings_basis,
        }


def detect(agg: Aggregates, th: Thresholds, disabled: frozenset[str] = frozenset()) -> list[Finding]:
    """Run every enabled rule; return findings ordered high to low severity."""
    findings: list[Finding] = []
    for rule in (_context_bloat, _long_sessions, _repeated_reads, _image_rereads, _large_payloads, _subagent_premium, _unattended_runaway, _edit_then_verify):
        finding = rule(agg, th)
        if finding is not None and finding.rule not in disabled:
            findings.append(finding)
    order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: order[f.severity])
    return findings


def _edit_then_verify(agg: Aggregates, th: Thresholds) -> Finding | None:
    """Edits whose check ran as its own model request, the pattern SoL-Pi fuses into one call."""
    pairs = agg.fuse_pairs
    if len(pairs) < th.fuse_pair_min_count:
        return None
    total = sum(pair.avoidable_tokens for pair in pairs)
    top = sorted(pairs, key=lambda pair: pair.avoidable_tokens, reverse=True)[: th.max_examples]
    evidence = [
        f"{pair.avoidable_tokens:>9,} tok  {pair.at.date()}  session={pair.session_id[:8]}  "
        f"{_short(pair.file_path)} -> {pair.command}"
        for pair in top
    ]
    return Finding(
        rule="edit-then-verify",
        severity="medium",
        title=f"{len(pairs)} edits ran their check as a separate request: {total:,} tokens in those turns",
        evidence=evidence,
        suggestion=(
            "Ask for the edit and its check in one step - 'edit X, then run Y on it' - so the agent does not "
            "spend a whole request issuing the command. Keep them apart when the command depends on reading the edit first."
        ),
        avoidable_tokens=total,
        savings_status="attributed",
        savings_basis=(
            "Each turn's context and output are read from the transcript; that the pair could have been fused "
            "is inferred from the command, not observed. Not a verified reduction and not a ledger row."
        ),
    )


def _context_bloat(agg: Aggregates, th: Thresholds) -> Finding | None:
    bloated = [s for s in agg.sessions if not s.is_subagent and s.turns >= th.bloat_min_turns and s.share_over_threshold >= th.bloat_share]
    if not bloated:
        return None
    wasted = sum(s.usage.total for s in bloated)
    share = wasted / agg.usage.total if agg.usage.total else 0.0
    evidence = [
        f"{s.session_id[:8]}  {s.turns:>5} turns  {s.share_over_threshold:>4.0%} over {th.context_tokens // 1000}k  "
        f"peak {s.peak_context:,}  {s.usage.total:,} tok  | {s.first_prompt[:50]}"
        for s in bloated[: th.max_examples]
    ]
    return Finding(
        rule="context-bloat",
        severity="high",
        title=f"{len(bloated)} sessions ran mostly above {th.context_tokens // 1000}k context: {share:.0%} of all tokens",
        evidence=evidence,
        suggestion=(
            f"Context exceeded {th.context_tokens // 1000}k; this can be necessary for complex work. "
            "At a natural task boundary, review a handoff brief and retain the evidence needed for the next step. "
            "Compare completion and rework before treating a shorter session as an improvement."
        ),
    )


def _long_sessions(agg: Aggregates, th: Thresholds) -> Finding | None:
    long = [s for s in agg.sessions if not s.is_subagent and s.turns >= th.long_session_turns]
    if not long:
        return None
    evidence = [
        f"{s.session_id[:8]}  {s.turns:>5} turns  {s.usage.total:,} tok  {s.start.date()} -> {s.end.date()}  | {s.first_prompt[:50]}"
        for s in long[: th.max_examples]
    ]
    return Finding(
        rule="long-session",
        severity="medium",
        title=f"{len(long)} sessions exceeded {th.long_session_turns} turns",
        evidence=evidence,
        suggestion="Don't: resume a 500-turn session for the next task. Do: one task per session; write a five-line handoff note yourself and start fresh with it.",
    )


def _repeated_reads(agg: Aggregates, th: Thresholds) -> Finding | None:
    return _read_opportunity(agg, th, images=False)


def _image_rereads(agg: Aggregates, th: Thresholds) -> Finding | None:
    return _read_opportunity(agg, th, images=True)


def _read_opportunity(agg: Aggregates, th: Thresholds, images: bool) -> Finding | None:
    threshold = th.image_reread_count if images else th.repeated_read_count
    repeated = [
        group for group in agg.read_groups
        if group.tally.count >= threshold
        and (Path(group.file_path).suffix.lower() in IMAGE_SUFFIXES) == images
    ]
    if not repeated:
        return None
    repeated.sort(key=lambda group: group.tally.bytes, reverse=True)
    evidence = [
        f"{group.tally.count}x  {_human(group.tally.bytes)}  session={group.session_id} "
        f"project={group.project}  {_short(group.file_path)}  "
        f"offset={group.offset or 'default'} limit={group.limit or 'default'} pages={group.pages or 'default'}"
        for group in repeated[:th.max_examples]
    ]
    evidence.append("Content revisions and continued context availability are unknown; repeated reads may be necessary.")
    return Finding(
        rule="image-rereads" if images else "repeated-reads",
        severity="medium",
        title=f"{len(repeated)} same-session {'image' if images else 'file'} read groups reached {threshold}+ calls: review opportunity",
        evidence=evidence,
        suggestion="Check whether the file changed or earlier content left context before removing a read. Reuse available content when sufficient; otherwise request only the range needed and verify task completion.",
    )


def _large_payloads(agg: Aggregates, th: Thresholds) -> Finding | None:
    if not agg.large_payloads:
        return None
    total_bytes = sum(t.bytes for t in agg.large_payloads.values())
    count = sum(t.count for t in agg.large_payloads.values())
    evidence = [f"{name:<12} {t.count:>5} results  {_human(t.bytes):>9}" for name, t in list(agg.large_payloads.items())[: th.max_examples]]
    top_commands = [c for c in agg.commands.items() if c[1].bytes > th.large_payload_bytes][: th.max_examples]
    evidence.extend(f"  bash: {_human(t.bytes):>8}  {head[:70]}" for head, t in top_commands)
    return Finding(
        rule="large-payloads",
        severity="high" if total_bytes > th.large_payload_high_total_bytes else "medium",
        title=f"{count} tool results over {_human(th.large_payload_bytes)} pushed {_human(total_bytes)} into context",
        evidence=evidence,
        suggestion="Don't: let a command dump a whole log or dataframe into the conversation. Do: ask for the last 50 lines, a grep for the error, or run it yourself and paste the failing line.",
    )


def _subagent_premium(agg: Aggregates, th: Thresholds) -> Finding | None:
    sub_total = agg.subagent_usage.total
    if sub_total < th.subagent_min_tokens:
        return None
    premium = sum(u.total for model, u in agg.subagent_by_model.items() if th.is_premium(model))
    share = premium / sub_total if sub_total else 0.0
    if share < th.subagent_premium_share:
        return None
    evidence = [f"{model:<24} {u.total:>16,} tok" for model, u in agg.subagent_by_model.items()][: th.max_examples]
    overall = sub_total / agg.usage.total if agg.usage.total else 0.0
    return Finding(
        rule="subagent-premium-model",
        severity="high",
        title=f"{share:.0%} of subagent tokens ran on a premium model (subagents are {overall:.0%} of all tokens)",
        evidence=evidence,
        suggestion="Evaluate a lower-cost model on representative subagent tasks. Compare completion quality, retries, latency and billed cost before changing routing; model choice alone does not eliminate tokens.",
    )


def _unattended_runaway(agg: Aggregates, th: Thresholds) -> Finding | None:
    """Scheduled or CI runs nobody watches: the latest run cost far more than the workflow's median, or runs produced nothing."""
    groups: dict[str, list] = {}
    for s in agg.sessions:
        if not s.is_subagent and s.workflow:
            groups.setdefault(s.workflow, []).append(s)
    evidence: list[str] = []
    for name, runs in groups.items():
        if len(runs) < th.workflow_min_runs:
            continue
        runs.sort(key=lambda r: r.start)
        per_run = [r.usage.total + r.subagent_usage.total for r in runs]
        median = sorted(per_run)[len(per_run) // 2]
        silent = [r for r in runs if r.usage.output_tokens == 0]
        for run, tokens in zip(runs, per_run, strict=True):
            if median and tokens >= th.workflow_runaway_ratio * median:
                evidence.append(f"{name:<28} run on {run.start.date()} cost {tokens:,} tok = {tokens / median:.1f}x its median ({median:,})")
        if silent:
            evidence.append(f"{name:<28} {len(silent)} run(s) produced no output; {sum(r.usage.total for r in silent):,} tok")
    if not evidence:
        return None
    return Finding(
        rule="unattended-runaway",
        severity="high",
        title=f"{len(evidence)} unattended workflow signal(s): runs that blew past their median or produced nothing",
        evidence=evidence[: th.max_examples],
        suggestion="Review workload changes and completion evidence for these runs. Configure a budget in the agent runtime or gateway where supported; Burnlens does not yet enforce per-workflow budgets. Missing output tokens alone do not prove a failed run.",
    )


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


def _short(path: str) -> str:
    home = str(Path.home())
    return path.replace(home, "~", 1) if path.startswith(home) else path
