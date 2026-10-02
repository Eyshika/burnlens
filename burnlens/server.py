"""Local dashboard server: JSON API over the profiler plus the static app.

Binds to localhost only. Every request reads the live transcript tree; parsed
sessions are cached per window and invalidated when any transcript's mtime
changes, so the dashboard is never stale and never re-parses needlessly.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import secrets
import threading
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import __version__
from .adapters import load_all
from .aggregate import aggregate
from .analyze import AnalysisError, LLMConfig, SessionNarrator
from .coach import coach_prompt, habits
from .config import Settings
from .findings import Thresholds, detect
from .graph import build_graph
from .handoff import build_handoff
from .live import AlertNotifier, LiveMonitor
from .otlp import OtlpIngest
from .model import Session
from .interventions import InterventionConfig, InterventionError, InterventionService, default_state_dir
from .report import report_payload, session_payload
from .teacher import lessons
from .transcripts import TranscriptError

logger = logging.getLogger(__name__)

APP_NAME = "Burnlens"
STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
TIMELINE_MAX_POINTS = 400
ALL_DAYS = 0
LIVE_INTERVAL_SECONDS = 5.0
ACTION_BODY_LIMIT = 16_384


@dataclass
class _Entry:
    sessions: list[Session]
    newest_mtime: float
    loaded_at: datetime


@dataclass
class SessionCache:
    """Parsed sessions per window, refreshed when the transcript tree changes."""

    root: Path
    source: str = "claude-code"
    extras: list[tuple[str, Path]] = field(default_factory=list)
    _entries: dict[int, _Entry] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def get(self, days: int) -> list[Session]:
        newest = self._newest_mtime()
        with self._lock:
            entry = self._entries.get(days)
            if entry is not None and entry.newest_mtime == newest:
                return entry.sessions
            since = None if days == ALL_DAYS else datetime.now(timezone.utc) - timedelta(days=days)
            logger.info("loading transcripts root=%s days=%s", self.root, days or "all")
            sessions = load_all(self.source, self.root, self.extras, since=since)
            self._entries[days] = _Entry(sessions=sessions, newest_mtime=newest, loaded_at=datetime.now(timezone.utc))
            return sessions

    def loaded_at(self, days: int) -> datetime | None:
        entry = self._entries.get(days)
        return entry.loaded_at if entry else None

    def _newest_mtime(self) -> float:
        roots = [self.root, *(p for _, p in self.extras)]
        return max((p.stat().st_mtime for r in roots for p in (r.rglob("*") if r.is_dir() else [r]) if p.is_file()), default=0.0)


class DashboardServer(ThreadingHTTPServer):
    """ThreadingHTTPServer carrying the cache and thresholds for the handler."""

    daemon_threads = True

    def __init__(self, host: str, port: int, root: Path, settings: Settings | Thresholds, notify: bool = True, source: str = "claude-code", extras: list[tuple[str, Path]] | None = None) -> None:
        super().__init__((host, port), DashboardHandler)
        self.action_token = secrets.token_urlsafe(32)
        self.interventions = InterventionService(InterventionConfig(default_state_dir()))
        self.settings = settings if isinstance(settings, Settings) else Settings(thresholds=settings)
        self.otlp = OtlpIngest()
        merged = list(extras or [])
        if self.otlp.spool.exists() and ("generic", self.otlp.spool) not in merged:
            merged.append(("generic", self.otlp.spool))
        self.cache = SessionCache(root=root, source=source, extras=merged)
        self.thresholds = self.settings.thresholds
        self.root = root
        self.source = source
        self.live = AlertNotifier(LiveMonitor(root, self.thresholds, prices=self.settings.prices), LIVE_INTERVAL_SECONDS, notify=notify)

    def start_live(self) -> None:
        self.live.start()

    def server_close(self) -> None:
        if hasattr(self, "live"):
            self.live.stop()
        super().server_close()


class DashboardHandler(BaseHTTPRequestHandler):
    server: DashboardServer

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        url = urlparse(self.path)
        query = parse_qs(url.query)
        try:
            if url.path.startswith("/api/"):
                self._api(url.path, query)
            else:
                self._static(url.path)
        except TranscriptError as exc:
            self._json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        except Exception as exc:  # one bad request must not kill the server
            logger.exception("request failed path=%s", self.path)
            self._json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        url = urlparse(self.path)
        if url.path.startswith("/api/actions/"):
            self._action_post(url.path)
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        try:
            if url.path == "/v1/logs":
                payload = json.loads(body.decode("utf-8") or "{}")
                written = self.server.otlp.ingest_logs(payload if isinstance(payload, dict) else {})
                if written and ("generic", self.server.otlp.spool) not in self.server.cache.extras:
                    self.server.cache.extras.append(("generic", self.server.otlp.spool))
                self._json({"partialSuccess": {}})
                return
            if url.path in ("/v1/metrics", "/v1/traces"):
                self._json({"partialSuccess": {}})  # accepted and ignored; logs carry what we need
                return
            self._json({"error": "unknown endpoint"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            logger.exception("POST failed path=%s", self.path)
            self._json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.BAD_REQUEST)

    def _local_actions(self) -> bool:
        port = self.server.server_address[1]
        allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}
        if self.server.server_address[0] not in {"127.0.0.1", "::1"} or self.headers.get("Host") not in allowed_hosts:
            self._json({"error": "Interventions are available only on the local dashboard."}, HTTPStatus.FORBIDDEN)
            return False
        return True

    def _action_post(self, path: str) -> None:
        if not self._local_actions():
            return
        token = self.headers.get("X-Burnlens-Token", "")
        if not secrets.compare_digest(token, self.server.action_token):
            self._json({"error": "Refresh the dashboard before changing an intervention."}, HTTPStatus.FORBIDDEN)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= ACTION_BODY_LIMIT or self.headers.get_content_type() != "application/json":
                raise ValueError("Expected a small JSON request.")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Expected an object.")
            action_id = payload.get("action_id")
            if not isinstance(action_id, str):
                raise ValueError("An action ID is required.")
            service = self.server.interventions
            if path == "/api/actions/apply":
                action = service.apply(action_id, payload.get("revision"))
            elif path == "/api/actions/revert":
                action = service.revert(action_id)
            elif path == "/api/actions/outcome":
                action = service.record_outcome(action_id, payload.get("run_id"), payload.get("outcome"), payload.get("notes", ""))
            else:
                self._json({"error": "Unknown action."}, HTTPStatus.NOT_FOUND)
                return
            self._json({"action": action.as_dict()})
        except (ValueError, InterventionError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception:
            logger.exception("intervention request failed")
            self._json({"error": "Could not save the intervention."}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, fmt: str, *args: object) -> None:
        logger.debug("%s " + fmt, self.address_string(), *args)

    def _api(self, path: str, query: dict[str, list[str]]) -> None:
        days = _int_param(query, "days", ALL_DAYS)
        if path == "/api/meta":
            self._json(
                {
                    "app": APP_NAME,
                    "version": __version__,
                    "action_token": self.server.action_token,
                    "root": str(self.server.root),
                    "thresholds": _jsonable(self.server.thresholds),
                    "settings": self.server.settings.as_dict(),
                    "source": self.server.source,
                    "loaded_at": _iso(self.server.cache.loaded_at(days)),
                    "llm": (LLMConfig.from_env() or _NoLLM()).public(),
                }
            )
            return
        if path == "/api/live":
            with self.server.live.lock:
                snap = self.server.live.latest
            if snap is None:
                snap = self.server.live.monitor.snapshot()
            self._json(snap.as_dict())
            return
        if path == "/api/actions":
            if not self._local_actions():
                return
            actions = self.server.interventions.refresh(self.server.cache.get(days), self.server.thresholds)
            self._json({"actions": [action.as_dict() for action in actions]})
            return
        sessions = self.server.cache.get(days)
        if path == "/api/report":
            th = self.server.thresholds
            agg = aggregate(sessions, context_threshold=th.context_tokens, large_payload_bytes=th.large_payload_bytes, premium_markers=th.premium_markers)
            payload = report_payload(agg, detect(agg, th, self.server.settings.disabled_rules))
            user = (query.get("user") or [None])[0]
            payload["habits"] = [h.as_dict() for h in habits(sessions, agg, th, user=user)]
            payload["lessons"] = [l.as_dict() for l in lessons([s for s in sessions if not user or s.user == user or s.is_subagent])]
            payload["days"] = days
            payload["loaded_at"] = _iso(self.server.cache.loaded_at(days))
            self._json(payload)
            return
        if path.startswith("/api/explain/"):
            config = LLMConfig.from_env()
            if config is None:
                self._json({"error": "no model configured; set BURNLENS_LLM_API_KEY or OPENROUTER_API_KEY and restart"}, HTTPStatus.BAD_REQUEST)
                return
            session_id = path[len("/api/explain/") :]
            matches = [s for s in sessions if not s.is_subagent and s.session_id.startswith(session_id)]
            if len(matches) != 1:
                self._json({"error": f"{len(matches)} sessions match {session_id!r}"}, HTTPStatus.NOT_FOUND)
                return
            children = [s for s in sessions if s.parent_session_id == matches[0].session_id]
            try:
                explanation = SessionNarrator(config).explain(matches[0], children, refresh=_int_param(query, "refresh", 0) == 1)
            except AnalysisError as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_GATEWAY)
                return
            self._json(explanation.as_dict())
            return
        if path == "/api/graph":
            th = self.server.thresholds
            agg = aggregate(sessions, context_threshold=th.context_tokens, large_payload_bytes=th.large_payload_bytes, premium_markers=th.premium_markers)
            self._json(build_graph(sessions, agg).as_dict())
            return
        if path == "/api/coach":
            prompt = (query.get("prompt") or [""])[0]
            model = (query.get("model") or [""])[0]
            context_now = _int_param(query, "context", 0)
            self._json(coach_prompt(prompt, context_now, model, self.server.thresholds, self.server.settings.task_tiers, prices=self.server.settings.prices).as_dict())
            return
        if path.startswith("/api/handoff/"):
            session_id = path[len("/api/handoff/") :]
            matches = [s for s in sessions if not s.is_subagent and s.session_id.startswith(session_id)]
            if len(matches) != 1:
                self._json({"error": f"{len(matches)} sessions match {session_id!r}"}, HTTPStatus.NOT_FOUND)
                return
            children = [s for s in sessions if s.parent_session_id == matches[0].session_id]
            self._json({"session_id": matches[0].session_id, "markdown": build_handoff(matches[0], children)})
            return
        if path.startswith("/api/session/"):
            session_id = path[len("/api/session/") :]
            matches = [s for s in sessions if not s.is_subagent and s.session_id.startswith(session_id)]
            if len(matches) != 1:
                self._json({"error": f"{len(matches)} sessions match {session_id!r}"}, HTTPStatus.NOT_FOUND)
                return
            children = [s for s in sessions if s.parent_session_id == matches[0].session_id]
            self._json(session_payload(matches[0], children, TIMELINE_MAX_POINTS))
            return
        self._json({"error": "unknown endpoint"}, HTTPStatus.NOT_FOUND)

    def _static(self, path: str) -> None:
        name = "index.html" if path in ("", "/") else path.lstrip("/")
        target = (STATIC_DIR / name).resolve()
        if STATIC_DIR.resolve() not in target.parents or not target.is_file():
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        body = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)


def serve(
    root: Path,
    settings: Settings | Thresholds,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
    notify: bool = True,
    source: str = "claude-code",
    extras: list[tuple[str, Path]] | None = None,
) -> None:
    """Run the dashboard and the live alert loop until interrupted."""
    if not root.exists():
        logger.error("source root does not exist: %s", root)
        raise TranscriptError(f"source root does not exist: {root}")
    for extra_source, extra_root in extras or []:
        if not extra_root.exists():
            raise TranscriptError(f"{extra_source} source does not exist: {extra_root}")
    server = DashboardServer(host, port, root, settings, notify=notify, source=source, extras=extras)
    server.start_live()
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"{APP_NAME} running at {url}  (Ctrl+C to stop)")
    print(f"OTLP receiver: POST {url}v1/logs  (spool {server.otlp.spool})")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


def _jsonable(thresholds: Thresholds) -> dict[str, object]:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(thresholds).items()}


class _NoLLM:
    @staticmethod
    def public() -> dict[str, object]:
        return {"configured": False}


def _int_param(query: dict[str, list[str]], key: str, default: int) -> int:
    values = query.get(key)
    if not values:
        return default
    try:
        return max(0, int(values[0]))
    except ValueError:
        return default


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None
