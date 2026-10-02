"""OTLP receiver: every developer's Claude Code streams telemetry into one Burnlens.

Claude Code exports OpenTelemetry logs (``http/json``) with ``claude_code.api_request``,
``claude_code.tool_result``, ``claude_code.user_prompt`` and ``claude_code.assistant_response``
events, each carrying ``session.id`` and, for a signed-in account, ``user.email``. That is the
per-person identity a laptop transcript never has, so this receiver is the start of the
team server: point a fleet at ``http://<burnlens>:8765`` and the same rules, health score,
habits and coach run per person.

Received events are folded into generic-JSONL records in a spool directory, one file per
day, which the server reads as an extra generic source. Nothing else in the pipeline needs
to know about OTLP.

Enable on each developer machine::

    export CLAUDE_CODE_ENABLE_TELEMETRY=1
    export OTEL_LOGS_EXPORTER=otlp
    export OTEL_EXPORTER_OTLP_PROTOCOL=http/json
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://<burnlens-host>:8765
    export OTEL_LOG_TOOL_DETAILS=1      # optional: tool parameters (file paths, commands)
    export OTEL_LOG_USER_PROMPTS=1      # optional: prompt text for the coach
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_SPOOL = Path(os.environ.get("BURNLENS_OTLP_DIR", str(Path.home() / ".burnlens" / "otlp")))
AGENT_NAME = "claude-code"
EVENT_API_REQUEST = "claude_code.api_request"
EVENT_TOOL_RESULT = "claude_code.tool_result"
EVENT_USER_PROMPT = "claude_code.user_prompt"
EVENT_ASSISTANT = "claude_code.assistant_response"
MAX_PENDING_TOOLS = 200


@dataclass
class _Pending:
    tools: list[dict[str, Any]] = field(default_factory=list)
    prompt: str = ""
    text: str = ""


class OtlpIngest:
    """Turn OTLP/JSON log payloads into generic records, keeping per-session tool state."""

    def __init__(self, spool: Path = DEFAULT_SPOOL) -> None:
        self._spool = spool
        self._pending: dict[str, _Pending] = {}
        self._lock = threading.Lock()

    @property
    def spool(self) -> Path:
        return self._spool

    def ingest_logs(self, payload: dict[str, Any]) -> int:
        """Handle one ``POST /v1/logs`` body. Returns the number of records written."""
        records: list[dict[str, Any]] = []
        with self._lock:
            for resource_logs in payload.get("resourceLogs") or []:
                resource_attrs = _attrs((resource_logs.get("resource") or {}).get("attributes"))
                for scope_logs in resource_logs.get("scopeLogs") or []:
                    for log in scope_logs.get("logRecords") or []:
                        record = self._fold(log, resource_attrs)
                        if record is not None:
                            records.append(record)
        if records:
            self._append(records)
        return len(records)

    def _fold(self, log: dict[str, Any], resource_attrs: dict[str, Any]) -> dict[str, Any] | None:
        attrs = {**resource_attrs, **_attrs(log.get("attributes"))}
        name = str(attrs.get("event.name") or _body_text(log.get("body")) or "")
        session_id = str(attrs.get("session.id") or "")
        if not session_id:
            return None
        pending = self._pending.setdefault(session_id, _Pending())
        if name == EVENT_TOOL_RESULT:
            if len(pending.tools) < MAX_PENDING_TOOLS:
                pending.tools.append({
                    "id": attrs.get("tool_use_id"), "name": attrs.get("tool_name"),
                    "input": _tool_input(attrs), "result_bytes": int(attrs.get("tool_result_size_bytes") or 0),
                })
            return None
        if name == EVENT_USER_PROMPT:
            pending.prompt = str(attrs.get("prompt") or "")
            return None
        if name == EVENT_ASSISTANT:
            pending.text = str(attrs.get("response") or "")
            return None
        if name != EVENT_API_REQUEST:
            return None
        ts = _timestamp(log, attrs)
        record = {
            "session_id": session_id,
            "agent": AGENT_NAME,
            "user": str(attrs.get("user.email") or attrs.get("user.id") or attrs.get("user.account_uuid") or ""),
            "project": str(attrs.get("organization.id") or AGENT_NAME),
            "ts": ts.isoformat(),
            "model": str(attrs.get("model") or "?"),
            "usage": {
                "input_tokens": int(attrs.get("input_tokens") or 0),
                "output_tokens": int(attrs.get("output_tokens") or 0),
                "cache_read_input_tokens": int(attrs.get("cache_read_tokens") or 0),
                "cache_creation_input_tokens": int(attrs.get("cache_creation_tokens") or 0),
            },
            "tools": pending.tools,
            "id": attrs.get("request_id") or attrs.get("message.uuid"),
        }
        workflow = attrs.get("workflow.name")
        if workflow:
            record["workflow"] = str(workflow)
            record["run_kind"] = "ci"
            if attrs.get("workflow.run_id"):
                record["session_id"] = f"{workflow}:{attrs['workflow.run_id']}"
                session_id = record["session_id"]
        if attrs.get("agent.name"):
            record["parent_session_id"] = session_id
            record["session_id"] = f"{session_id}:{attrs['agent.name']}"
        if pending.prompt:
            record["prompt"] = pending.prompt
        if pending.text:
            record["text"] = pending.text
        self._pending[session_id] = _Pending()
        return record

    def _append(self, records: list[dict[str, Any]]) -> None:
        try:
            self._spool.mkdir(parents=True, exist_ok=True)
            path = self._spool / f"{datetime.now(timezone.utc).date().isoformat()}.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")
        except OSError as exc:
            logger.error("cannot write OTLP spool %s: %s", self._spool, exc)


def _attrs(items: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items or []:
        if not isinstance(item, dict) or "key" not in item:
            continue
        out[str(item["key"])] = _value(item.get("value"))
    return out


def _value(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    for key in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if key in value:
            raw = value[key]
            return int(raw) if key == "intValue" else raw
    if "arrayValue" in value:
        return [_value(v) for v in (value["arrayValue"] or {}).get("values", [])]
    if "kvlistValue" in value:
        return _attrs((value["kvlistValue"] or {}).get("values"))
    return None


def _body_text(body: Any) -> str:
    value = _value(body)
    return value if isinstance(value, str) else ""


def _tool_input(attrs: dict[str, Any]) -> dict[str, Any]:
    raw = attrs.get("tool_input") or attrs.get("tool_parameters")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if not isinstance(raw, dict):
        return {}
    out = dict(raw)
    for key in ("file_path", "path", "notebook_path"):
        if key in out and "file_path" not in out:
            out["file_path"] = str(out[key])
    for key in ("bash_command", "full_command", "command"):
        if key in out and "command" not in out:
            out["command"] = str(out[key])
            break
    return out


def _timestamp(log: dict[str, Any], attrs: dict[str, Any]) -> datetime:
    for key in ("timeUnixNano", "observedTimeUnixNano"):
        raw = log.get(key)
        if raw:
            try:
                return datetime.fromtimestamp(int(raw) / 1e9, tz=timezone.utc)
            except (ValueError, OverflowError, OSError):
                pass
    stamp = attrs.get("event.timestamp")
    if isinstance(stamp, str):
        try:
            return datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)
