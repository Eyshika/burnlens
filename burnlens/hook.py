"""Pre-execution guard: a Claude Code PreToolUse hook.

Claude Code runs this before a tool call, with the call as JSON on stdin. We
estimate what the call will push into context, read the session's current
context size from the tail of its transcript, and answer allow / ask / deny
with a reason the agent can act on. This is the brake: it fires before the
tokens are spent, not after.

Install once with ``burnlens install-hooks``; it writes the hook into
``~/.claude/settings.json`` pointing at this interpreter.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
import shutil
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .coach import TASK_TIER, TIER_PREMIUM, CacheReset, cache_resets, classify_task, coach_prompt, context_growth
from .config import PriceTable
from .findings import Thresholds
from .interventions import InterventionConfig, InterventionService
from .ledger import BYTES_PER_TOKEN, Ledger, PreventedLoad
from .model import IMAGE_SUFFIXES, Turn, Usage

logger = logging.getLogger(__name__)

HOOK_EVENT = "PreToolUse"
PROMPT_EVENT = "UserPromptSubmit"
HOOK_MATCHER = "Read|Bash|Agent"
HOOK_TIMEOUT_SECONDS = 10
TAIL_BYTES = 256 * 1024
EVENTS_PATH = Path.home() / ".burnlens" / "hook-events.jsonl"
STATE_DIR = Path.home() / ".burnlens" / "hook-state"
DEFAULT_SETTINGS = Path.home() / ".claude" / "settings.json"

# Shell commands that tend to dump a lot of output unless something limits them.
VERBOSE_COMMAND = re.compile(
    r"^(cat|find|tree|ls\s+-R|git\s+log|git\s+diff|grep\s+-r|rg\b|pytest(?!.*-q)|npm\s+install|pip\s+install|docker\s+build|tail\s+-f|env|printenv|pip\s+list|conda\s+list)\b"
)
LIMITERS = ("| head", "| tail", "| grep", "| wc", "| sed -n", "| awk", "-m ", "--max-count", "| cut", "| sort | uniq", "> /dev/null", "2>/dev/null | head", "| jq")


@dataclass(frozen=True)
class Decision:
    permission: str  # "allow" | "ask" | "deny"
    reason: str
    context_note: str = ""
    updated_input: dict[str, Any] | None = None  # reserved for transformations with verified semantic safety

    def as_hook_output(self) -> dict[str, Any]:
        specific: dict[str, Any] = {"hookEventName": HOOK_EVENT, "permissionDecision": self.permission}
        if self.reason:
            specific["permissionDecisionReason"] = self.reason
        if self.context_note:
            specific["additionalContext"] = self.context_note
        if self.updated_input is not None:
            specific["updatedInput"] = self.updated_input
        out: dict[str, Any] = {"hookSpecificOutput": specific}
        if self.permission != "allow" or self.updated_input is not None:
            out["systemMessage"] = self.announcement()
        return out

    def announcement(self) -> str:
        """What the human reads. A refusal has to be unmistakable about the call not running."""
        body = self.reason or self.context_note
        if self.permission == "deny":
            return f"BLOCKED by Burnlens - this call did not run. {body}"
        if self.permission == "ask":
            return f"Burnlens is asking you to confirm before this runs. {body}"
        return body


class HookGuard:
    """Decide whether a tool call should run, given the session's current zone."""

    def __init__(self, thresholds: Thresholds, strict: bool = False, autofix: bool = True, task_tiers: dict[str, str] | None = None) -> None:
        self._th = thresholds
        self._strict = strict  # strict: ask the human on every automatic subagent spawn, even in green
        # Kept for CLI compatibility; heuristic advice never rewrites tool inputs.
        self._task_tiers = {**TASK_TIER, **(task_tiers or {})}

    def decide(self, payload: dict[str, Any]) -> Decision:
        tool = str(payload.get("tool_name") or "")
        tool_input = payload.get("tool_input") or {}
        if not isinstance(tool_input, dict):
            tool_input = {}
        context_now = read_context_now(payload.get("transcript_path"))
        zone = self._zone(context_now)
        looping = self._loop_guard(tool, tool_input, payload, context_now)
        if looping is not None:
            return looping
        if tool == "Agent":
            return self._agent(tool_input, zone, context_now)
        if tool == "Read":
            return self._read(tool_input, zone, context_now)
        if tool == "Bash":
            return self._bash(tool_input, zone, context_now)
        return Decision("allow", "")

    def _loop_guard(self, tool: str, tool_input: dict[str, Any], payload: dict[str, Any], context_now: int) -> Decision | None:
        """Refuse a call that has already run identically several times over.

        The size is not estimated: the identical results already in the transcript say what this
        one would have returned. That measurement is what makes the row countable.
        """
        if not tool:
            return None
        repeat = repeated_call(payload.get("transcript_path"), tool, tool_input, self._th)
        if repeat is None:
            return None
        runs, size = repeat
        target = _call_target(tool, tool_input)
        reason = (
            f"{tool} has already run {runs} times in a row with identical input and returned "
            f"{_human_bytes(size)} each time. Repeating it re-reads the same result into a "
            f"{context_now:,}-token context. Use the result already above, or change the input."
        )
        Ledger().append(
            PreventedLoad(
                at=datetime.now(timezone.utc),
                rule="loop-guard",
                target=target,
                bytes_prevented=size,
                divisor=BYTES_PER_TOKEN,
                session_id=str(payload.get("session_id") or ""),
                repo=str(payload.get("cwd") or ""),
                decision="deny",
                basis=f"{runs} identical prior results in this session, median {size} bytes",
            )
        )
        return Decision("deny", f"Burnlens: {reason}")

    def _zone(self, context_now: int) -> str:
        if context_now >= self._th.context_tokens * self._th.live_context_high_multiplier:
            return "red"
        if context_now >= self._th.context_tokens:
            return "amber"
        return "green"

    def _agent(self, tool_input: dict[str, Any], zone: str, context_now: int) -> Decision:
        model = str(tool_input.get("model") or "").lower()
        subagent_type = str(tool_input.get("subagent_type") or "general-purpose")
        task = classify_task(str(tool_input.get("prompt") or tool_input.get("description") or ""))
        cheap_enough = task not in {"general", "debug", "design"} and self._task_tiers.get(task, TIER_PREMIUM) != TIER_PREMIUM
        premium_or_inherited = not model or self._th.is_premium(model)
        if premium_or_inherited and subagent_type != "fork" and cheap_enough:
            tier = self._task_tiers[task]
            reason = (
                f"Burnlens: this looks like a {task} subagent and it would run on {model or 'the parent model (premium)'}. "
                f"Consider evaluating {tier} on a representative task before changing models. Compare correctness, retries, and total cost; equivalent quality is not established. "
                "Hard debugging or design subagents can stay on the premium model."
            )
            if zone == "red":
                return Decision("ask", reason + f" Session context is already {context_now:,} tokens (red zone).")
            if zone == "amber" or self._strict:
                return Decision("ask", reason)
            return Decision("allow", "", context_note=reason)
        if zone == "red":
            return Decision(
                "ask",
                f"Burnlens: red zone. Session context is {context_now:,} tokens; every subagent result lands in it. Consider a new session first.",
            )
        return Decision("allow", "", context_note="Burnlens: keep subagents flat (no nested spawning) and cap their searches.")

    def _read(self, tool_input: dict[str, Any], zone: str, context_now: int) -> Decision:
        path_text = str(tool_input.get("file_path") or "")
        if not path_text:
            return Decision("allow", "")
        path = Path(path_text)
        try:
            size = path.stat().st_size
        except (OSError, ValueError):
            return Decision("allow", "")
        partial = tool_input.get("offset") is not None or tool_input.get("limit") is not None
        is_image = path.suffix.lower() in IMAGE_SUFFIXES
        est_tokens = size // BYTES_PER_TOKEN
        if is_image:
            note = f"Burnlens: {path.name} is a {size // 1024} KB image; it stays in context for the rest of the session."
            return Decision("ask" if zone != "green" else "allow", note if zone != "green" else "", context_note="" if zone != "green" else note)
        if not partial and size > self._th.large_payload_bytes:
            note = f"Burnlens: {path.name} is {size // 1024} KB (~{est_tokens:,} tokens). Don't read it whole. Do: grep for the symbol and Read a line range, or ask the user which function matters."
            if zone == "red":
                return Decision("deny", note + f" Session context is {context_now:,} tokens (red zone).")
            return Decision("ask" if zone == "amber" else "allow", note if zone == "amber" else "", context_note="" if zone == "amber" else note)
        return Decision("allow", "")

    def _bash(self, tool_input: dict[str, Any], zone: str, context_now: int) -> Decision:
        command = str(tool_input.get("command") or "").strip()
        head = command.splitlines()[0] if command else ""
        looks_verbose = bool(VERBOSE_COMMAND.search(head)) and not any(l in command for l in LIMITERS)
        if not looks_verbose:
            return Decision("allow", "")
        note = "Burnlens: this command may produce large output. Do: consider a native quiet or bounded-output option, or inspect captured output separately. Preserve exit status and required diagnostics; no command changes were applied."
        if zone == "red":
            return Decision("deny", note + f" Session context is {context_now:,} tokens (red zone).")
        if zone == "amber":
            return Decision("ask", note)
        return Decision("allow", "", context_note=note)


def read_context_now(transcript_path: Any) -> int:
    """Context size of the latest assistant turn, from the tail of the transcript."""
    if not transcript_path:
        return 0
    path = Path(str(transcript_path))
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - TAIL_BYTES))
            tail = handle.read().decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return 0
    latest = 0
    for line in tail.splitlines():
        if '"usage"' not in line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        message = record.get("message")
        usage = message.get("usage") if isinstance(message, dict) else None
        if isinstance(usage, dict):
            latest = Usage.from_dict(usage).context
    return latest


def run_hook(stdin_text: str, thresholds: Thresholds, events_path: Path = EVENTS_PATH, strict: bool = False, autofix: bool = True, state_dir: Path = STATE_DIR, task_tiers: dict[str, str] | None = None, prices: PriceTable | None = None) -> dict[str, Any]:
    """Entry for ``burnlens hook``: read one payload, decide, log, return output JSON."""
    try:
        payload = json.loads(stdin_text or "{}")
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    if payload.get("hook_event_name") == PROMPT_EVENT:
        try:
            out = prompt_hook_output(payload, thresholds, strict, state_dir, task_tiers, prices)
        except Exception:  # fail open
            logger.exception("prompt coaching failed; allowing")
            return {}
        permission = "block" if out.get("decision") == "block" else ("coach" if out.get("systemMessage") else "allow")
        _append_event(events_path, {**payload, "tool_name": PROMPT_EVENT}, Decision(permission, str(out.get("systemMessage") or "")[:200]))
        return out
    try:
        decision = HookGuard(thresholds, strict=strict, autofix=autofix, task_tiers=task_tiers).decide(payload)
    except Exception:  # fail open: a hook bug must never block the user's tool call
        logger.exception("hook decision failed; allowing")
        decision = Decision("allow", "")
    if payload.get("tool_name") == "Bash":
        try:
            advice = InterventionService(InterventionConfig()).advice_for(payload.get("transcript_path", ""))
            if advice:
                decision = replace(decision, context_note=" ".join(filter(None, [decision.context_note, advice])))
        except Exception:
            logger.exception("optional intervention advice unavailable; preserving guard decision")
    _append_event(events_path, payload, decision)
    return decision.as_hook_output()


def prompt_hook_output(
    payload: dict[str, Any],
    thresholds: Thresholds,
    strict: bool,
    state_dir: Path = STATE_DIR,
    task_tiers: dict[str, str] | None = None,
    prices: PriceTable | None = None,
) -> dict[str, Any]:
    """UserPromptSubmit: coach the human, brief the agent, and in strict red zone stop the prompt.

    Quiet by design: the cost note for a big context is said once per zone crossing per session,
    never on every prompt. Everything else the coach says is about THIS prompt (a cheaper tier for
    this task, a whole-repo ask, pasted logs) or a STOP backed by waste evidence.
    """
    prompt = str(payload.get("prompt") or "")
    context_now = read_context_now(payload.get("transcript_path"))
    model = read_model_now(payload.get("transcript_path"))
    transcript = payload.get("transcript_path")
    coaching = coach_prompt(
        prompt,
        context_now,
        model,
        thresholds,
        task_tiers=task_tiers,
        waste=read_recent_waste(transcript, thresholds),
        resets=read_recent_cache_resets(transcript, thresholds),
        prices=prices,
        growth_per_turn=read_context_growth(transcript),
    )
    session_id = str(payload.get("session_id") or "")
    already = _zone_announced(state_dir, session_id, coaching.zone)
    messages = [m for m in coaching.messages if not (m.startswith("Expensive") and already)]
    if coaching.stop and already and len(messages) > 1:
        messages = messages[1:]  # the STOP was said once; keep only the prompt-specific advice
    out: dict[str, Any] = {}
    if messages:
        text = "Burnlens: " + " ".join(messages)
        if coaching.rewrite:
            text += f"\nTry instead: {coaching.rewrite}"
        out["systemMessage"] = text
    if coaching.agent_notes:
        out["hookSpecificOutput"] = {"hookEventName": PROMPT_EVENT, "additionalContext": " ".join(coaching.agent_notes)}
    if coaching.stop and strict:
        out["decision"] = "block"
        out["reason"] = coaching.messages[0]
    return out


def _zone_announced(state_dir: Path, session_id: str, zone: str) -> bool:
    """True if this zone was already announced for this session; records it otherwise."""
    if not session_id or zone == "green":
        return False
    path = state_dir / f"{session_id}.json"
    try:
        state = json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError):
        state = {}
    if state.get("zone") == zone:
        return True
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"zone": zone}))
    except OSError:
        pass
    return False


def read_recent_waste(transcript_path: Any, thresholds: Thresholds) -> list[str]:
    """Concrete waste in the transcript tail: one file read 3+ times, oversized tool results."""
    if not transcript_path:
        return []
    path = Path(str(transcript_path))
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - TAIL_BYTES))
            tail = handle.read().decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return []
    reads: dict[str, int] = {}
    big = 0
    for line in tail.splitlines():
        if '"tool_use"' not in line and '"tool_result"' not in line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = record.get("message") if isinstance(record, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and block.get("name") == "Read":
                fp = str((block.get("input") or {}).get("file_path") or "")
                if fp:
                    reads[fp] = reads.get(fp, 0) + 1
            elif block.get("type") == "tool_result":
                body = block.get("content")
                length = len(body) if isinstance(body, str) else sum(len(str(b.get("text", ""))) for b in body if isinstance(b, dict)) if isinstance(body, list) else 0
                if length > thresholds.large_payload_bytes:
                    big += 1
    out: list[str] = []
    repeated = [(Path(p).name, n) for p, n in reads.items() if n >= 3]
    if repeated:
        name, n = max(repeated, key=lambda kv: kv[1])
        out.append(f"{name} read {n}x")
    if big >= 2:
        out.append(f"{big} tool results over {thresholds.large_payload_bytes // 1000} KB")
    return out


def repeated_call(transcript_path: Any, name: str, tool_input: dict[str, Any], th: Thresholds) -> tuple[int, int] | None:
    """Length and median result size of the run of identical calls ending the transcript.

    Returns None unless the run is long enough AND every call in it produced a result we measured;
    without the sizes there is nothing countable and the guard stays out of the way.
    """
    signature = _signature(name, tool_input)
    calls, sizes = _tail_calls(transcript_path)
    run: list[int] = []
    for call_id, call_signature in reversed(calls):
        if call_signature != signature:
            break
        if call_id not in sizes:
            break
        run.append(sizes[call_id])
    if len(run) < th.loop_repeat_count:
        return None
    ordered = sorted(run)
    return len(run), ordered[len(ordered) // 2]


def _signature(name: str, tool_input: dict[str, Any]) -> str:
    try:
        return name + "\x00" + json.dumps(tool_input, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return name + "\x00" + repr(sorted(tool_input.items()))


def _tail_calls(transcript_path: Any) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """Ordered (tool_use_id, signature) pairs from the tail, plus the size each result came back as."""
    if not transcript_path:
        return [], {}
    path = Path(str(transcript_path))
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - TAIL_BYTES))
            tail = handle.read().decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return [], {}
    calls: list[tuple[str, str]] = []
    sizes: dict[str, int] = {}
    for line in tail.splitlines():
        if '"tool_use"' not in line and '"tool_result"' not in line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = record.get("message") if isinstance(record, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and block.get("id"):
                calls.append((str(block["id"]), _signature(str(block.get("name") or ""), block.get("input") or {})))
            elif block.get("type") == "tool_result" and block.get("tool_use_id"):
                measured, _ = _result_bytes(block.get("content"))
                sizes[str(block["tool_use_id"])] = measured
    return calls, sizes


def _result_bytes(content: Any) -> tuple[int, bool]:
    if isinstance(content, str):
        return len(content), False
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict):
                total += len(str(block.get("text", "")))
        return total, False
    return 0, False


def _call_target(tool: str, tool_input: dict[str, Any]) -> str:
    for key in ("file_path", "command", "pattern", "path", "prompt"):
        value = tool_input.get(key)
        if value:
            return f"{tool}: {str(value).splitlines()[0][:120]}"
    return tool


def _human_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB"):
        if size < 1024 or unit == "MB":
            return f"{size:,.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} MB"


def read_recent_cache_resets(transcript_path: Any, thresholds: Thresholds) -> list[CacheReset]:
    """Cache resets visible in the transcript tail, using the same rule as the report."""
    turns = _tail_turns(transcript_path)
    return cache_resets(turns, thresholds) if len(turns) > 1 else []


def read_context_growth(transcript_path: Any) -> float | None:
    """Mean per-turn context increase in the tail; feeds the compaction window headroom."""
    return context_growth(_tail_turns(transcript_path))


def _tail_turns(transcript_path: Any) -> list[Turn]:
    """Assistant turns from the tail, with model and usage only; enough for the cache rule."""
    if not transcript_path:
        return []
    path = Path(str(transcript_path))
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - TAIL_BYTES))
            tail = handle.read().decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return []
    turns: list[Turn] = []
    for line in tail.splitlines():
        if '"usage"' not in line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = record.get("message") if isinstance(record, dict) else None
        if not isinstance(message, dict):
            continue
        usage = message.get("usage")
        model = message.get("model")
        # a synthetic turn carries no model, and treating it as a change invents a reset
        if not isinstance(usage, dict) or not model or model == "<synthetic>":
            continue
        turns.append(Turn(message_id=str(message.get("id") or ""), timestamp=_parse_ts(record.get("timestamp")), model=str(model), usage=Usage.from_dict(usage)))
    return turns


def _parse_ts(raw: Any) -> datetime:
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)


def read_model_now(transcript_path: Any) -> str:
    """Model of the latest assistant turn, from the tail of the transcript."""
    if not transcript_path:
        return ""
    path = Path(str(transcript_path))
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - TAIL_BYTES))
            tail = handle.read().decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return ""
    model = ""
    for line in tail.splitlines():
        if '"model"' not in line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = record.get("message") if isinstance(record, dict) else None
        if isinstance(message, dict) and message.get("model") and message.get("model") != "<synthetic>":
            model = str(message["model"])
    return model


def _append_event(events_path: Path, payload: dict[str, Any], decision: Decision) -> None:
    try:
        events_path.parent.mkdir(parents=True, exist_ok=True)
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "session_id": payload.get("session_id"),
                        "tool": payload.get("tool_name"),
                        "permission": decision.permission,
                        "reason": (decision.reason or decision.context_note)[:200],
                        "target": _event_target(payload),
                        "intervention_status": "applied" if decision.updated_input is not None else ("advised" if decision.permission in {"allow", "coach"} and (decision.reason or decision.context_note) else "gated" if decision.permission in {"ask", "deny", "block"} else "none"),
                        "input_changed": decision.updated_input is not None,
                        "savings_verified": False,
                    }
                )
                + "\n"
            )
    except OSError as exc:
        logger.debug("could not append hook event: %s", exc)


def _event_target(payload: dict[str, Any]) -> str:
    """Short label of what the decision was about: file name, command head, or subagent description."""
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return ""
    if tool_input.get("file_path"):
        return Path(str(tool_input["file_path"])).name
    if tool_input.get("command"):
        return " ".join(str(tool_input["command"]).strip().splitlines()[0].split())[:60]
    if tool_input.get("description"):
        return str(tool_input["description"])[:60]
    return ""


def hook_command(strict: bool = False, autofix: bool = True, config_path: Path | None = None) -> str:
    """Build a shell-safe command preserving the explicitly selected configuration."""
    args = [sys.executable, "-m", "burnlens", "hook"]
    if config_path is not None:
        args.extend(["--config", str(config_path.expanduser().resolve())])
    if strict:
        args.append("--strict")
    if not autofix:
        args.append("--no-autofix")
    return shlex.join(args)


def install_hooks(settings_path: Path = DEFAULT_SETTINGS, remove: bool = False, strict: bool = False, autofix: bool = True, config_path: Path | None = None) -> str:
    """Add (or remove) the PreToolUse hook in Claude Code's settings. Idempotent."""
    settings: dict[str, Any] = {}
    if settings_path.exists():
        settings = json.loads(settings_path.read_text() or "{}")
        shutil.copy2(settings_path, settings_path.with_suffix(".json.bak"))
    hooks = settings.setdefault("hooks", {})
    for event, matcher in ((HOOK_EVENT, HOOK_MATCHER), (PROMPT_EVENT, None)):
        entries: list[dict[str, Any]] = [e for e in hooks.get(event, []) if not _is_ours(e)]
        if not remove:
            entry: dict[str, Any] = {"hooks": [{"type": "command", "command": hook_command(strict, autofix, config_path), "timeout": HOOK_TIMEOUT_SECONDS}]}
            if matcher:
                entry["matcher"] = matcher
            entries.append(entry)
        if entries:
            hooks[event] = entries
        else:
            hooks.pop(event, None)
    if not hooks:
        settings.pop("hooks", None)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    verb = "removed from" if remove else ("installed (strict: you approve every automatic subagent) into" if strict else "installed into")
    return f"Burnlens hook {verb} {settings_path} (backup at {settings_path.with_suffix('.json.bak')}). Restart Claude Code sessions to pick it up."


def _is_ours(entry: dict[str, Any]) -> bool:
    return any(marker in str(h.get("command", "")) for h in entry.get("hooks", []) for marker in ("-m burnlens hook", "-m tokprof hook"))
