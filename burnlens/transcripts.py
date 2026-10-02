"""Read Claude Code transcripts from disk into Session records.

Layout on disk (Claude Code 2.x):

    ~/.claude/projects/<project-slug>/<session-id>.jsonl
    ~/.claude/projects/<project-slug>/<session-id>/subagents/<agent-id>.jsonl

Each line is a JSON record. Assistant records carry ``message.usage`` and
``message.content`` (text / tool_use blocks). User records carry the prompt
or ``tool_result`` blocks that answer an earlier ``tool_use``.

A streamed assistant message is written several times with the same
``message.id`` as it grows; the last line has the complete usage, so records
are merged by id and the latest usage wins.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .model import Session, ToolCall, Turn, Usage
from .repo import resolve as resolve_repo

logger = logging.getLogger(__name__)

DEFAULT_ROOT = Path.home() / ".claude" / "projects"
SUBAGENT_DIR = "subagents"
FIRST_PROMPT_CHARS = 100
TEXT_PREVIEW_CHARS = 160
PROMPT_KEEP_CHARS = 400
PROMPTS_PER_SESSION = 300


class TranscriptError(RuntimeError):
    """Raised when the transcript root cannot be read."""


def iter_transcript_paths(root: Path, modified_after: datetime | None = None) -> Iterator[Path]:
    """Yield every transcript file, skipping files untouched since ``modified_after``."""
    if not root.is_dir():
        logger.error("transcript root does not exist: %s", root)
        raise TranscriptError(f"transcript root does not exist: {root}")
    cutoff = modified_after.timestamp() if modified_after else None
    for path in sorted(root.rglob("*.jsonl")):
        if cutoff is not None and path.stat().st_mtime < cutoff:
            continue
        yield path


def load_sessions(root: Path = DEFAULT_ROOT, since: datetime | None = None) -> list[Session]:
    """Parse every transcript under ``root``, keeping turns at or after ``since``."""
    sessions: list[Session] = []
    for path in iter_transcript_paths(root, modified_after=since):
        session = parse_session(path, root, since=since)
        if session.turns:
            sessions.append(session)
    return sessions


def parse_session(path: Path, root: Path, since: datetime | None = None) -> Session:
    """Parse one transcript file. Turns before ``since`` are dropped."""
    session = _session_shell(path, root)
    turns_by_id: dict[str, Turn] = {}
    calls_by_id: dict[str, ToolCall] = {}

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            record = _parse_line(line, path, line_no)
            if record is None:
                continue
            if not session.cwd and record.get("cwd"):
                session.cwd = str(record["cwd"])
                session.git_branch = str(record.get("gitBranch") or "")
            kind = record.get("type")
            message = record.get("message") or {}
            if kind == "assistant":
                _absorb_assistant(record, message, turns_by_id, calls_by_id)
            elif kind == "user":
                _absorb_user(message, calls_by_id, session)

    session.repo = resolve_repo(session.cwd)
    turns = sorted(turns_by_id.values(), key=lambda t: t.timestamp)
    if since is not None:
        turns = [t for t in turns if t.timestamp >= since]
    session.turns = turns
    return session


def _session_shell(path: Path, root: Path) -> Session:
    relative = path.relative_to(root)
    parts = relative.parts
    project = parts[0] if parts else "?"
    parent: str | None = None
    if len(parts) >= 3 and parts[-2] == SUBAGENT_DIR:
        parent = parts[-3]
    return Session(
        session_id=path.stem,
        project=project,
        path=path,
        parent_session_id=parent,
        first_prompt="",
    )


def _parse_line(line: str, path: Path, line_no: int) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        logger.debug("skipping malformed line %s:%d", path, line_no)
        return None
    return record if isinstance(record, dict) else None


def _absorb_assistant(
    record: dict[str, Any],
    message: dict[str, Any],
    turns_by_id: dict[str, Turn],
    calls_by_id: dict[str, ToolCall],
) -> None:
    raw_usage = message.get("usage")
    if not isinstance(raw_usage, dict):
        return
    message_id = str(message.get("id") or record.get("uuid") or "")
    if not message_id:
        return
    timestamp = _parse_timestamp(record.get("timestamp"))
    if timestamp is None:
        return
    turn = turns_by_id.get(message_id)
    if turn is None:
        turn = Turn(message_id=message_id, timestamp=timestamp, model=str(message.get("model") or "?"), usage=Usage())
        turns_by_id[message_id] = turn
    turn.usage = Usage.from_dict(raw_usage)
    for block in _blocks(message.get("content")):
        if block.get("type") == "text" and block.get("text"):
            turn.text_preview = " ".join(str(block["text"]).split())[:TEXT_PREVIEW_CHARS]
        if block.get("type") != "tool_use":
            continue
        call_id = str(block.get("id") or "")
        if not call_id or call_id in calls_by_id:
            continue
        call = ToolCall(tool_use_id=call_id, name=str(block.get("name") or "?"), input=dict(block.get("input") or {}))
        calls_by_id[call_id] = call
        turn.tool_calls.append(call)


def _absorb_user(message: dict[str, Any], calls_by_id: dict[str, ToolCall], session: Session) -> None:
    content = message.get("content")
    if isinstance(content, str):
        if not content.startswith("<"):
            prompt = " ".join(content.split())[:FIRST_PROMPT_CHARS]
            session.last_prompt = prompt
            if not session.first_prompt:
                session.first_prompt = prompt
            if len(session.prompts) < PROMPTS_PER_SESSION:
                session.prompts.append(content.strip()[:PROMPT_KEEP_CHARS])
                session.prompt_chars.append(len(content))
        return
    for block in _blocks(content):
        if block.get("type") != "tool_result":
            continue
        call = calls_by_id.get(str(block.get("tool_use_id") or ""))
        if call is None:
            continue
        size, has_image = _result_size(block.get("content"))
        call.result_bytes += size
        call.result_has_image = call.result_has_image or has_image


def _result_size(content: Any) -> tuple[int, bool]:
    """Approximate bytes a tool_result pushed into context, and whether it held an image."""
    if content is None:
        return 0, False
    if isinstance(content, str):
        return len(content), False
    size = 0
    has_image = False
    for block in _blocks(content):
        block_type = block.get("type")
        if block_type == "text":
            size += len(str(block.get("text") or ""))
        elif block_type == "image":
            has_image = True
            source = block.get("source") or {}
            size += len(str(source.get("data") or ""))
        else:
            size += len(json.dumps(block))
    return size, has_image


def _blocks(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict)]


def _parse_timestamp(raw: Any) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
