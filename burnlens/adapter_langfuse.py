"""Langfuse adapter: traces + observations exports into sessions.

Accepts what Langfuse gives you without transformation, mixed under one directory:

* ``GET /api/public/observations`` pages (``{"data": [...], "meta": {...}}``), or the UI's
  observations export (JSON list / JSONL).
* ``GET /api/public/traces`` pages or the traces export, used for ``userId``,
  ``sessionId``, ``name`` and ``tags``.

Mapping: a GENERATION observation is one turn (usage from ``usageDetails`` with the provider's
cache keys, else ``usage``); TOOL / RETRIEVER observations become tool calls attached to the
generation that preceded them in the same trace; session = trace ``sessionId`` else ``traceId``;
person = trace ``userId``; application = trace ``name`` (else ``metadata.agent`` / ``app``, else
``"langfuse"``). Field names follow the Langfuse public API objects; confirm against one real
export if your version differs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .adapter_litellm import TOOL_ALIASES, _arguments, _parse_ts, _text_of
from .model import Session, ToolCall, Turn, Usage
from .transcripts import TranscriptError

logger = logging.getLogger(__name__)

GENERATION_TYPES = frozenset({"GENERATION", "EMBEDDING"})
TOOL_TYPES = frozenset({"TOOL", "RETRIEVER"})
DEFAULT_AGENT = "langfuse"
PROMPT_CHARS = 100
TEXT_CHARS = 160
CACHE_READ_KEYS = ("cache_read_input_tokens", "input_cached_tokens", "cached_tokens", "input_cache_read")
CACHE_WRITE_KEYS = ("cache_creation_input_tokens", "input_cache_creation", "cache_creation")


@dataclass
class _TraceInfo:
    user: str = ""
    session: str = ""
    name: str = ""
    prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def load_langfuse(root: Path, since: datetime | None = None) -> list[Session]:
    if not root.exists():
        logger.error("langfuse source does not exist: %s", root)
        raise TranscriptError(f"langfuse source does not exist: {root}")
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.suffix in (".json", ".jsonl"))
    traces: dict[str, _TraceInfo] = {}
    generations: list[dict[str, Any]] = []
    tools: list[dict[str, Any]] = []
    for path in files:
        for record in _records(path):
            kind = _classify(record)
            if kind == "trace":
                traces[str(record["id"])] = _TraceInfo(
                    user=str(record.get("userId") or ""), session=str(record.get("sessionId") or ""), name=str(record.get("name") or ""),
                    prompt=_prompt_from(record.get("input")), metadata=record.get("metadata") if isinstance(record.get("metadata"), dict) else {},
                )
            elif kind == "generation":
                generations.append(record)
            elif kind == "tool":
                tools.append(record)
    sessions: dict[str, Session] = {}
    turns_by_trace: dict[str, list[Turn]] = {}
    for gen in sorted(generations, key=lambda g: str(g.get("startTime") or "")):
        ts = _parse_ts(gen.get("startTime"))
        if ts is None:
            continue
        trace_id = str(gen.get("traceId") or gen.get("id"))
        info = traces.get(trace_id, _TraceInfo())
        session_id = info.session or trace_id
        session = sessions.get(session_id)
        if session is None:
            agent = info.name or str(info.metadata.get("agent") or info.metadata.get("app") or DEFAULT_AGENT)
            workflow = str(info.metadata.get("workflow") or "")
            session = Session(session_id=session_id, project=str(info.metadata.get("project") or agent), path=root, parent_session_id=None, first_prompt="", agent=agent, user=info.user,
                              workflow=workflow, run_kind="scheduled" if workflow else "interactive")
            sessions[session_id] = session
        prompt = _prompt_from(gen.get("input")) or info.prompt
        if prompt:
            session.last_prompt = prompt
            if not session.first_prompt:
                session.first_prompt = prompt
        turn = Turn(
            message_id=str(gen.get("id")), timestamp=ts, model=str(gen.get("model") or "?"), usage=_usage(gen),
            text_preview=" ".join(_text_of(_assistant_text(gen.get("output"))).split())[:TEXT_CHARS],
        )
        session.turns.append(turn)
        turns_by_trace.setdefault(trace_id, []).append(turn)
    for tool in tools:
        ts = _parse_ts(tool.get("startTime"))
        trace_turns = turns_by_trace.get(str(tool.get("traceId") or ""))
        if ts is None or not trace_turns:
            continue
        host = max((t for t in trace_turns if t.timestamp <= ts), key=lambda t: t.timestamp, default=trace_turns[0])
        name = str(tool.get("name") or tool.get("type") or "?")
        host.tool_calls.append(ToolCall(
            tool_use_id=str(tool.get("id")), name=TOOL_ALIASES.get(name.lower(), name),
            input=_arguments(tool.get("input")) if isinstance(tool.get("input"), (dict, str)) else {},
            result_bytes=len(_text_of(tool.get("output"))),
        ))
    out: list[Session] = []
    for session in sessions.values():
        session.turns.sort(key=lambda t: t.timestamp)
        if since is not None:
            session.turns = [t for t in session.turns if t.timestamp >= since]
        if session.turns:
            out.append(session)
    return out


def _classify(record: dict[str, Any]) -> str:
    kind = str(record.get("type") or "").upper()
    if record.get("traceId") or kind:
        if kind in GENERATION_TYPES:
            return "generation"
        if kind in TOOL_TYPES:
            return "tool"
        return "other"
    if "id" in record and ("userId" in record or "sessionId" in record or "tags" in record or "timestamp" in record):
        return "trace"
    return "other"


def _usage(gen: dict[str, Any]) -> Usage:
    details = gen.get("usageDetails") if isinstance(gen.get("usageDetails"), dict) else {}
    usage = gen.get("usage") if isinstance(gen.get("usage"), dict) else {}
    prompt = int(details.get("input") or usage.get("input") or usage.get("promptTokens") or 0)
    output = int(details.get("output") or usage.get("output") or usage.get("completionTokens") or 0)
    cache_read = sum(int(details.get(k) or 0) for k in CACHE_READ_KEYS)
    cache_write = sum(int(details.get(k) or 0) for k in CACHE_WRITE_KEYS)
    fresh = max(0, prompt - cache_read - cache_write) if (cache_read or cache_write) and prompt >= cache_read + cache_write else prompt
    return Usage(input_tokens=fresh, cache_creation_input_tokens=cache_write, cache_read_input_tokens=cache_read, output_tokens=output)


def _prompt_from(value: Any) -> str:
    if isinstance(value, str):
        text = value
    elif isinstance(value, list):
        users = [m for m in value if isinstance(m, dict) and m.get("role") == "user"]
        text = _text_of(users[-1].get("content")) if users else ""
    elif isinstance(value, dict):
        messages = value.get("messages")
        if isinstance(messages, list):
            return _prompt_from(messages)
        text = _text_of(value.get("input") or value.get("prompt") or value.get("query") or "")
    else:
        text = ""
    return " ".join(text.split())[:PROMPT_CHARS]


def _assistant_text(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("content") or value.get("text") or value.get("output") or ""
    return value


def _records(path: Path) -> Iterator[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".jsonl":
        for line in text.splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record
        return
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        data = data["data"]
    if isinstance(data, dict):
        yield data
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item


__all__ = ["load_langfuse"]
_ = timezone  # re-exported for callers building fixtures
