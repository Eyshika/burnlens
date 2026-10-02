"""LiteLLM adapter: read what the proxy already logs, no export step.

Two shapes are accepted, mixed freely under one directory:

1. **StandardLoggingPayload** objects, as written by LiteLLM's S3 / GCS / custom-callback
   logging. One JSON object per file, a JSON list per file, or JSONL. Epoch-second
   ``startTime``, raw provider usage under ``metadata.usage_object``, chat ``messages`` and
   the ``response`` (tool calls live in ``response.choices[].message.tool_calls`` and tool
   results in ``messages`` with ``role: "tool"``).

2. **Spend log rows**, as returned by ``GET /spend/logs?summarize=false`` or exported from the
   ``LiteLLM_SpendLogs`` table. ISO ``startTime``, ``metadata`` sometimes a JSON string,
   ``messages`` / ``response`` present only when ``store_prompts_in_spend_logs`` is on.

Grouping into sessions: ``session_id`` if the caller set one, else ``trace_id``, else
``spend_logs_metadata.session_id``, else one session per user per day (documented fallback).
Identity: person = ``metadata.user_api_key_user_id`` / ``user`` / ``end_user``;
application = ``metadata.user_api_key_alias`` (the virtual key, usually one per app), else the
team alias, else ``"litellm"``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .model import Session, ToolCall, Turn, Usage
from .transcripts import TranscriptError

logger = logging.getLogger(__name__)

PROMPT_CHARS = 100
TEXT_CHARS = 160
DEFAULT_AGENT = "litellm"
TOOL_ALIASES: dict[str, str] = {
    "read": "Read", "read_file": "Read", "view": "Read", "cat": "Read", "open_file": "Read", "get_file_contents": "Read",
    "edit": "Edit", "edit_file": "Edit", "apply_patch": "Edit", "str_replace_editor": "Edit", "patch": "Edit", "replace_in_file": "Edit",
    "write": "Write", "write_file": "Write", "create_file": "Write",
    "bash": "Bash", "shell": "Bash", "exec": "Bash", "run_command": "Bash", "terminal": "Bash", "execute": "Bash", "execute_command": "Bash",
    "agent": "Agent", "task": "Agent", "spawn": "Agent", "subagent": "Agent", "delegate": "Agent",
    "web_search": "WebSearch", "search": "WebSearch", "web_fetch": "WebFetch", "fetch": "WebFetch", "retrieve": "Retrieve", "retrieval": "Retrieve",
}
PATH_KEYS = ("file_path", "path", "filepath", "filename", "file")
COMMAND_KEYS = ("command", "cmd", "script")


def load_litellm(root: Path, since: datetime | None = None) -> list[Session]:
    """Read every ``*.json`` / ``*.jsonl`` under ``root`` (or the single file) and group into sessions."""
    if not root.exists():
        logger.error("litellm source does not exist: %s", root)
        raise TranscriptError(f"litellm source does not exist: {root}")
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.suffix in (".json", ".jsonl"))
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
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
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
        logger.debug("skipping non-JSON file %s", path)
        return
    if isinstance(data, dict) and isinstance(data.get("data"), list):  # /spend/logs style envelope
        data = data["data"]
    if isinstance(data, dict):
        yield data
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item


def _absorb(record: dict[str, Any], sessions: dict[str, Session], root: Path) -> None:
    ts = _parse_ts(record.get("startTime") or record.get("start_time") or record.get("endTime"))
    if ts is None:
        return
    meta = _metadata(record)
    user = _first(meta.get("user_api_key_user_id"), record.get("user"), meta.get("user_api_key_end_user_id"), record.get("end_user"))
    agent = _first(meta.get("user_api_key_alias"), meta.get("user_api_key_team_alias"), record.get("model_group")) or DEFAULT_AGENT
    project = _first(meta.get("user_api_key_team_alias"), record.get("team_id")) or agent
    session_id = _session_key(record, meta, user, ts)
    session = sessions.get(session_id)
    if session is None:
        spend_meta = meta.get("spend_logs_metadata") if isinstance(meta.get("spend_logs_metadata"), dict) else {}
        workflow = _first(spend_meta.get("workflow"), next((t.split(":", 1)[1] for t in _as_list(record.get("request_tags")) if isinstance(t, str) and t.startswith("workflow:")), None))
        session = Session(session_id=session_id, project=str(project), path=root, parent_session_id=None, first_prompt="", agent=str(agent), user=str(user or ""),
                          workflow=str(workflow or ""), run_kind="scheduled" if workflow else "interactive")
        sessions[session_id] = session

    messages = _as_list(record.get("messages"))
    prompt = _last_user_text(messages)
    if prompt:
        session.last_prompt = prompt
        if not session.first_prompt:
            session.first_prompt = prompt

    tool_calls = _tool_calls(record, messages)
    session.turns.append(
        Turn(
            message_id=str(record.get("id") or record.get("request_id") or f"{session_id}:{ts.isoformat()}:{len(session.turns)}"),
            timestamp=ts,
            model=str(_first(record.get("model"), (record.get("hidden_params") or {}).get("litellm_model_name"), record.get("model_group")) or "?"),
            usage=_usage(record, meta),
            tool_calls=tool_calls,
            text_preview=_response_text(record.get("response")),
        )
    )


def _usage(record: dict[str, Any], meta: dict[str, Any]) -> Usage:
    prompt_tokens = int(record.get("prompt_tokens") or 0)
    completion_tokens = int(record.get("completion_tokens") or 0)
    raw = meta.get("usage_object") if isinstance(meta.get("usage_object"), dict) else {}
    cache_read = int(raw.get("cache_read_input_tokens") or 0)
    cache_write = int(raw.get("cache_creation_input_tokens") or 0)
    details = raw.get("prompt_tokens_details") if isinstance(raw.get("prompt_tokens_details"), dict) else {}
    if not cache_read and details:
        cache_read = int(details.get("cached_tokens") or 0)
    if not cache_read and record.get("cache_hit") in (True, "True", "true"):
        cache_read = prompt_tokens  # a proxy cache hit means nothing new was sent
    fresh = max(0, prompt_tokens - cache_read - cache_write)
    return Usage(input_tokens=fresh, cache_creation_input_tokens=cache_write, cache_read_input_tokens=cache_read, output_tokens=completion_tokens)


def _tool_calls(record: dict[str, Any], messages: list[dict[str, Any]]) -> list[ToolCall]:
    """Tool calls the model made in this response, sized by the tool results that came back later
    in the conversation (found in ``messages`` with role ``tool`` and a matching ``tool_call_id``)."""
    results: dict[str, int] = {}
    for message in messages:
        if message.get("role") == "tool":
            results[str(message.get("tool_call_id") or "")] = len(_text_of(message.get("content")))
    calls: list[ToolCall] = []
    response = record.get("response")
    choices = response.get("choices") if isinstance(response, dict) else None
    for choice in choices or []:
        message = choice.get("message") if isinstance(choice, dict) else None
        for raw in (message or {}).get("tool_calls") or []:
            if not isinstance(raw, dict):
                continue
            function = raw.get("function") or {}
            name = str(function.get("name") or raw.get("name") or "?")
            calls.append(ToolCall(tool_use_id=str(raw.get("id") or f"{record.get('id')}:{len(calls)}"), name=TOOL_ALIASES.get(name.lower(), name), input=_arguments(function.get("arguments")), result_bytes=results.get(str(raw.get("id") or ""), 0)))
    mcp = (record.get("metadata") or {}).get("mcp_tool_call_metadata") if isinstance(record.get("metadata"), dict) else None
    if isinstance(mcp, dict) and mcp.get("name"):
        name = str(mcp["name"])
        calls.append(ToolCall(tool_use_id=f"{record.get('id')}:mcp", name=TOOL_ALIASES.get(name.lower(), name), input=dict(mcp.get("arguments") or {}), result_bytes=len(json.dumps(mcp.get("result") or ""))))
    return calls


def _arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        args = raw
    elif isinstance(raw, str):
        try:
            args = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if not isinstance(args, dict):
            return {}
    else:
        return {}
    out = dict(args)
    for key in PATH_KEYS:
        if key in out and "file_path" not in out:
            out["file_path"] = str(out[key])
            break
    for key in COMMAND_KEYS:
        if key in out and "command" not in out:
            out["command"] = str(out[key])
            break
    return out


def _session_key(record: dict[str, Any], meta: dict[str, Any], user: str | None, ts: datetime) -> str:
    spend_meta = meta.get("spend_logs_metadata") if isinstance(meta.get("spend_logs_metadata"), dict) else {}
    explicit = _first(record.get("session_id"), record.get("litellm_session_id"), spend_meta.get("session_id"), record.get("trace_id"), record.get("litellm_trace_id"))
    if explicit:
        return str(explicit)
    for tag in _as_list(record.get("request_tags")):
        if isinstance(tag, str) and tag.startswith("session:"):
            return tag.split(":", 1)[1]
    return f"{user or 'anonymous'}-{ts.date().isoformat()}"


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    meta = record.get("metadata")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            meta = {}
    return meta if isinstance(meta, dict) else {}


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            text = _text_of(message.get("content"))
            if text and not text.startswith("<"):
                return " ".join(text.split())[:PROMPT_CHARS]
    return ""


def _response_text(response: Any) -> str:
    if isinstance(response, str):
        return " ".join(response.split())[:TEXT_CHARS]
    if isinstance(response, dict):
        for choice in response.get("choices") or []:
            message = choice.get("message") if isinstance(choice, dict) else None
            text = _text_of((message or {}).get("content"))
            if text:
                return " ".join(text.split())[:TEXT_CHARS]
    return ""


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
    if content is None:
        return ""
    return json.dumps(content)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return [v for v in value if v is not None]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _first(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def _parse_ts(raw: Any) -> datetime | None:
    if isinstance(raw, (int, float)):
        seconds = float(raw) / (1000.0 if raw > 1e12 else 1.0)  # tolerate milliseconds
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    if isinstance(raw, str) and raw:
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            try:
                return _parse_ts(float(raw))
            except ValueError:
                return None
        return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
    return None
