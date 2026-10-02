"""Read local Codex rollouts and Gemini saved chats without changing agent state.

Formats are internal upstream formats, not stable APIs. Tool result bytes describe
stored text (which the agent may already have truncated), not original output.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .model import Session, ToolCall, Turn, Usage
from .transcripts import (FIRST_PROMPT_CHARS, PROMPT_KEEP_CHARS, PROMPTS_PER_SESSION,
                          TEXT_PREVIEW_CHARS, TranscriptError, _parse_timestamp)

logger = logging.getLogger(__name__)
ALIASES = {"exec_command": "Bash", "shell_command": "Bash", "run_shell_command": "Bash",
           "shell": "Bash", "apply_patch": "Edit", "replace": "Edit", "write_file": "Write",
           "read_file": "Read", "read_many_files": "Read", "spawn_agent": "Agent"}


def _records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(str(x.get("text", "")) if isinstance(x, dict) and "text" in x
                         else json.dumps(x, ensure_ascii=False) for x in value)
    return json.dumps(value, ensure_ascii=False) if value is not None else ""


def _prompt(session: Session, value: Any) -> None:
    text = _text(value).strip()
    if not text:
        return
    session.last_prompt = " ".join(text.split())[:FIRST_PROMPT_CHARS]
    if not session.first_prompt:
        session.first_prompt = session.last_prompt
    if len(session.prompts) < PROMPTS_PER_SESSION:
        session.prompts.append(text[:PROMPT_KEEP_CHARS])
        session.prompt_chars.append(len(text))


def _number(raw: dict[str, Any], key: str) -> int:
    value = raw.get(key)
    return max(0, value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _usage(raw: dict[str, Any], gemini: bool = False) -> Usage:
    incoming = _number(raw, "input" if gemini else "input_tokens")
    cached = min(incoming, _number(raw, "cached" if gemini else "cached_input_tokens"))
    outgoing = _number(raw, "output" if gemini else "output_tokens")
    if gemini:
        outgoing += _number(raw, "thoughts")
    return Usage(input_tokens=incoming - cached, cache_read_input_tokens=cached, output_tokens=outgoing)


def _files(root: Path, pattern: str) -> list[Path]:
    if not root.exists():
        logger.error("native transcript source does not exist: %s", root)
        raise TranscriptError(f"native transcript source does not exist: {root}")
    return [root] if root.is_file() else sorted(root.rglob(pattern))


def load_codex(root: Path, since: datetime | None = None) -> list[Session]:
    sessions = []
    seen_responses: set[str] = set()
    for path in _files(root, "*.jsonl"):
        session = _codex_session(path, seen_responses)
        if since:
            session.turns = [turn for turn in session.turns if turn.timestamp >= since]
        if session.turns:
            sessions.append(session)
    return sessions


def _codex_session(path: Path, seen_responses: set[str]) -> Session:
    records = _records(path)
    session = Session(path.stem, "codex", path, None, "", agent="codex")
    detailed = any(r.get("type") == "token_usage_record" for r in records)
    calls: dict[str, ToolCall] = {}
    pending: list[ToolCall] = []
    previous: dict[str, Any] = {}
    model = "?"
    preview = ""
    for index, record in enumerate(records):
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        kind = record.get("type")
        timestamp = _parse_timestamp(record.get("timestamp"))
        if kind == "session_meta":
            session.session_id = str(payload.get("id") or payload.get("session_id") or path.stem)
            session.project = str(payload.get("cwd") or "codex")
        elif kind == "turn_context":
            model = str(payload.get("model") or model)
        elif kind == "event_msg" and payload.get("type") == "thread_settings_applied":
            model = str((payload.get("thread_settings") or {}).get("model") or model)
        elif kind == "response_item":
            subtype = payload.get("type")
            if subtype == "message":
                if payload.get("role") == "user":
                    _prompt(session, payload.get("content"))
                elif payload.get("role") == "assistant":
                    preview = _text(payload.get("content"))[:TEXT_PREVIEW_CHARS]
            elif subtype in {"function_call", "custom_tool_call"}:
                call_id = str(payload.get("call_id") or payload.get("id") or index)
                if call_id not in calls:
                    raw = payload.get("arguments", payload.get("input", {}))
                    if isinstance(raw, str):
                        try:
                            raw = json.loads(raw)
                        except json.JSONDecodeError:
                            raw = {}
                    args = dict(raw) if isinstance(raw, dict) else {}
                    if "cmd" in args:
                        args["command"] = args["cmd"]
                    name = str(payload.get("name") or "?")
                    call = ToolCall(call_id, ALIASES.get(name, name), args)
                    calls[call_id] = call
                    pending.append(call)
            elif subtype in {"function_call_output", "custom_tool_call_output"}:
                call = calls.get(str(payload.get("call_id")))
                if call:
                    call.result_bytes = len(_text(payload.get("output")).encode("utf-8"))
        elif kind == "event_msg" and payload.get("type") == "item_completed":
            item = payload.get("item") or {}
            if isinstance(item, dict) and item.get("type") == "CommandExecution":
                call_id = str(item.get("id") or index)
                call = calls.get(call_id)
                if call is None:
                    call = ToolCall(call_id, "Bash", {"command": item.get("command", "")})
                    calls[call_id] = call
                    pending.append(call)
                output = item.get("formatted_output")
                if not isinstance(output, str):
                    output = item.get("aggregated_output")
                if not isinstance(output, str):
                    output = str(item.get("stdout") or "") + str(item.get("stderr") or "")
                call.result_bytes = len(output.encode("utf-8"))
        raw_usage = None
        if detailed and kind == "token_usage_record":
            response_id = str(payload.get("response_id") or "")
            if response_id and response_id not in seen_responses:
                seen_responses.add(response_id)
                raw_usage = payload.get("usage")
            else:
                # Copied fork history must not attach its tools to a later new response.
                pending = []
                preview = ""
        elif not detailed and kind == "event_msg" and payload.get("type") == "token_count":
            info = payload.get("info") or {}
            total = info.get("total_token_usage")
            if isinstance(total, dict) and total != previous:
                # Last-response usage is preferable to cumulative deltas after compaction.
                raw_usage = info.get("last_token_usage")
                previous = total
        if isinstance(raw_usage, dict) and timestamp:
            usage = _usage(raw_usage)
            # Some desktop histories expose only a total budget count; do not invent its split.
            if not usage.total and _number(raw_usage, "total_tokens"):
                logger.warning("Codex usage components unavailable in %s; excluding total-only record", path)
                continue
            session.turns.append(Turn(str(payload.get("response_id") or index), timestamp,
                                      model, usage, pending, preview))
            pending = []
            preview = ""
    if pending and session.turns:
        session.turns[-1].tool_calls.extend(pending)
    return session


def load_gemini(root: Path, since: datetime | None = None) -> list[Session]:
    paths = _files(root, "session-*.json*")
    sessions = []
    for path in paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            records = _records(path)
        else:
            records = [value] if isinstance(value, dict) else []
        metadata: dict[str, Any] = {}
        messages: dict[str, dict[str, Any]] = {}
        for record in records:
            if "$rewindTo" in record:
                keys = list(messages)
                target = record["$rewindTo"]
                for key in keys[keys.index(target):] if target in keys else keys:
                    del messages[key]
            elif "id" in record:
                messages[str(record["id"])] = record
            else:
                update = record.get("$set", record)
                if not isinstance(update, dict):
                    continue
                metadata.update(update)
                if isinstance(update.get("messages"), list):
                    messages = {str(m["id"]): m for m in update["messages"] if isinstance(m, dict) and "id" in m}
        if not metadata.get("sessionId"):
            continue
        session = Session(str(metadata["sessionId"]), str(metadata.get("projectHash") or "gemini"),
                          path, None, "", agent="gemini-cli")
        for message in messages.values():
            if message.get("type") == "user":
                _prompt(session, message.get("content"))
            timestamp = _parse_timestamp(message.get("timestamp"))
            if message.get("type") != "gemini" or not timestamp or (since and timestamp < since):
                continue
            raw = message.get("tokens")
            if not isinstance(raw, dict):
                continue
            tools = []
            for item in message.get("toolCalls") or []:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "?")
                args = dict(item.get("args") or {})
                if "path" in args and "file_path" not in args:
                    args["file_path"] = args["path"]
                tools.append(ToolCall(str(item.get("id") or ""), ALIASES.get(name, name), args,
                                      len(_text(item.get("result")).encode("utf-8"))))
            session.turns.append(Turn(str(message["id"]), timestamp, str(message.get("model") or "?"),
                                      _usage(raw, gemini=True), tools, _text(message.get("content"))[:TEXT_PREVIEW_CHARS]))
        session.turns.sort(key=lambda turn: turn.timestamp)
        if session.turns:
            sessions.append(session)
    return sessions
