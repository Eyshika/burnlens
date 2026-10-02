"""Input adapters: turn any agent's records into Sessions.

Claude Code transcripts are one source. Teams often run their own tracing
(a gateway, LiteLLM, Langfuse, an OTel collector, a home-grown JSONL). The
``generic`` adapter reads a small normalised JSONL so any of those can be exported
into Burnlens without waiting for a bespoke adapter. One record per model call:

    {"session_id": "abc", "project": "checkout-service", "user": "sam", "agent": "codex",
     "ts": "2026-09-07T10:00:00Z", "model": "gpt-5",
     "usage": {"input_tokens": 12, "cache_read_input_tokens": 90000, "cache_creation_input_tokens": 0, "output_tokens": 400},
     "prompt": "fix the flaky loader test",            # optional: the user's message that started this turn
     "text": "I'll start with the failing test",       # optional: what the agent said
     "tools": [{"name": "read_file", "input": {"file_path": "src/loader.py"}, "result_bytes": 18000}],
     "parent_session_id": null}                        # set for subagent runs

Tool names are mapped onto the canonical set (Read, Edit, Write, Bash, Agent) so the
same rules, health score and graph apply. Unknown names pass through unchanged.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .adapter_native import load_codex, load_gemini
from .adapter_langfuse import load_langfuse
from .adapter_litellm import load_litellm
from .model import Session, ToolCall, Turn, Usage
from .transcripts import DEFAULT_ROOT, TranscriptError, load_sessions as load_claude_code

logger = logging.getLogger(__name__)

SOURCES = ("claude-code", "codex", "gemini-cli", "generic", "litellm", "langfuse")
GENERIC_SUFFIX = ".jsonl"
PROMPT_CHARS = 100
TEXT_CHARS = 160
TOOL_ALIASES: dict[str, str] = {
    "read": "Read", "read_file": "Read", "view": "Read", "cat": "Read", "open_file": "Read",
    "edit": "Edit", "edit_file": "Edit", "apply_patch": "Edit", "str_replace_editor": "Edit", "patch": "Edit",
    "write": "Write", "write_file": "Write", "create_file": "Write",
    "bash": "Bash", "shell": "Bash", "exec": "Bash", "run_command": "Bash", "terminal": "Bash", "execute": "Bash",
    "agent": "Agent", "task": "Agent", "spawn": "Agent", "subagent": "Agent", "delegate": "Agent",
    "web_search": "WebSearch", "search": "WebSearch", "web_fetch": "WebFetch", "fetch": "WebFetch",
}


def load(source: str, root: Path, since: datetime | None = None) -> list[Session]:
    """Dispatch on source name."""
    if source == "claude-code":
        return load_claude_code(root, since=since)
    if source == "codex":
        return load_codex(root, since=since)
    if source == "gemini-cli":
        return load_gemini(root, since=since)
    if source == "generic":
        return load_generic(root, since=since)
    if source == "litellm":
        return load_litellm(root, since=since)
    if source == "langfuse":
        return load_langfuse(root, since=since)
    raise TranscriptError(f"unknown source {source!r}; choose from {SOURCES}")


def load_all(source: str, root: Path, extras: list[tuple[str, Path]], since: datetime | None = None) -> list[Session]:
    """One view: the primary source plus any number of extra (source, path) pairs."""
    sessions = load(source, root, since=since)
    seen = {s.session_id for s in sessions}
    for extra_source, extra_root in extras:
        for session in load(extra_source, extra_root, since=since):
            if session.session_id in seen:
                session.session_id = f"{session.agent}:{session.session_id}"
            seen.add(session.session_id)
            sessions.append(session)
    return sessions


def load_generic(root: Path, since: datetime | None = None) -> list[Session]:
    """Read every ``*.jsonl`` under ``root`` in the normalised shape; group by session_id."""
    if not root.exists():
        logger.error("generic source does not exist: %s", root)
        raise TranscriptError(f"generic source does not exist: {root}")
    files = [root] if root.is_file() else sorted(root.rglob(f"*{GENERIC_SUFFIX}"))
    sessions: dict[str, Session] = {}
    for path in files:
        for record in _records(path):
            _absorb(record, sessions, root)
    out: list[Session] = []
    for session in sessions.values():
        session.turns.sort(key=lambda t: t.timestamp)
        if since is not None:
            session.turns = [t for t in session.turns if t.timestamp >= since]
        if session.turns:
            out.append(session)
    return out


def _records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("skipping malformed line %s:%d", path, line_no)
                continue
            if isinstance(record, dict):
                yield record


def _absorb(record: dict[str, Any], sessions: dict[str, Session], root: Path) -> None:
    session_id = str(record.get("session_id") or "")
    usage_raw = record.get("usage")
    ts = _parse_ts(record.get("ts") or record.get("timestamp"))
    if not session_id or not isinstance(usage_raw, dict) or ts is None:
        return
    session = sessions.get(session_id)
    if session is None:
        project = str(record.get("project") or record.get("agent") or "generic")
        parent = record.get("parent_session_id")
        session = Session(
            session_id=session_id, project=project, path=root, parent_session_id=str(parent) if parent else None, first_prompt="",
            agent=str(record.get("agent") or "generic"), user=str(record.get("user") or ""),
            workflow=str(record.get("workflow") or ""), run_kind=str(record.get("run_kind") or ("scheduled" if record.get("workflow") else "interactive")),
        )
        sessions[session_id] = session
    prompt = record.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        clean = " ".join(prompt.split())[:PROMPT_CHARS]
        session.last_prompt = clean
        if not session.first_prompt:
            session.first_prompt = clean
        if len(session.prompts) < 300:
            session.prompts.append(prompt.strip()[:400])
            session.prompt_chars.append(len(prompt))
    tools: list[ToolCall] = []
    for index, raw in enumerate(record.get("tools") or []):
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "?")
        tools.append(
            ToolCall(
                tool_use_id=str(raw.get("id") or f"{session_id}:{ts.isoformat()}:{index}"),
                name=TOOL_ALIASES.get(name.lower(), name),
                input=dict(raw.get("input") or {}),
                result_bytes=int(raw.get("result_bytes") or 0),
            )
        )
    text = record.get("text")
    session.turns.append(
        Turn(
            message_id=str(record.get("id") or f"{session_id}:{ts.isoformat()}:{len(session.turns)}"),
            timestamp=ts,
            model=str(record.get("model") or "?"),
            usage=Usage.from_dict(usage_raw),
            tool_calls=tools,
            text_preview=" ".join(str(text).split())[:TEXT_CHARS] if isinstance(text, str) else "",
        )
    )


def _parse_ts(raw: Any) -> datetime | None:
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(float(raw), tz=timezone.utc)
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


__all__ = ["DEFAULT_ROOT", "SOURCES", "load", "load_all", "load_generic"]


def default_root(source: str) -> Path:
    """Native roots; export-based sources require an explicit path."""
    roots = {
        "claude-code": Path.home() / ".claude" / "projects",
        "codex": Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "sessions",
        "gemini-cli": Path.home() / ".gemini" / "tmp",
    }
    if source not in roots:
        raise TranscriptError(f"{source} requires an explicit --root export path")
    return roots[source]


def discover_sources() -> list[tuple[str, Path]]:
    """Only report existing native transcript directories; never install hooks."""
    return [(source, root) for source in ("claude-code", "codex", "gemini-cli")
            if (root := default_root(source)).is_dir()]
