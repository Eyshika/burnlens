"""Live view: which sessions are active right now, what they are doing, how fast
they burn, and alerts when a session crosses a threshold.

Reads only transcripts modified in the last few minutes, so a snapshot is cheap
enough to take every few seconds. Alerts are evaluated server-side so they fire
even when no dashboard tab is open.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .findings import Thresholds
from .coach import TASK_TIER, TIER_PREMIUM, CompactionEconomics, classify_task, compaction_breakeven, context_growth
from .config import PriceTable
from .hook import EVENTS_PATH
from .model import Session
from .transcripts import TranscriptError, parse_session

logger = logging.getLogger(__name__)

RECENT_TOOLS = 6
ZONE_ORDER = {"green": 0, "amber": 1, "red": 2}
RECENT_HOOK_EVENTS = 20
HOOK_EVENTS_TAIL_BYTES = 256 * 1024
HOOK_PERMISSIONS = ("deny", "ask", "block", "coach", "allow")


@dataclass
class ActiveSession:
    session_id: str
    project: str
    model: str
    last_prompt: str
    last_text: str
    last_activity: datetime
    seconds_idle: float
    context_now: int
    turns_total: int
    turns_in_window: int
    tokens_in_window: int
    tokens_per_min: float
    recent_tools: list[str]
    subagents_active: int
    subagent_models: list[str]
    zone: str = "green"
    brake: dict[str, int] = field(default_factory=dict)  # hook decisions for this session in the window
    waste: list[str] = field(default_factory=list)  # concrete waste seen in the window; red needs at least one
    cheap_task_premium_subagents: int = 0  # subagents doing lookup/research-class work on a premium model
    compaction: CompactionEconomics | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "project": self.project,
            "model": self.model,
            "last_prompt": self.last_prompt,
            "last_text": self.last_text,
            "last_activity": self.last_activity.isoformat(),
            "seconds_idle": round(self.seconds_idle),
            "context_now": self.context_now,
            "turns_total": self.turns_total,
            "turns_in_window": self.turns_in_window,
            "tokens_in_window": self.tokens_in_window,
            "tokens_per_min": round(self.tokens_per_min),
            "recent_tools": list(self.recent_tools),
            "subagents_active": self.subagents_active,
            "subagent_models": list(self.subagent_models),
            "zone": self.zone,
            "brake": dict(self.brake),
            "waste": list(self.waste),
            "cheap_task_premium_subagents": self.cheap_task_premium_subagents,
            "compaction": self.compaction.as_dict() if self.compaction else None,
        }


@dataclass(frozen=True)
class Alert:
    key: str  # stable per (session, rule); used to notify once
    level: str  # "amber" | "red"
    session_id: str
    rule: str
    message: str

    def as_dict(self) -> dict[str, object]:
        return {"key": self.key, "level": self.level, "session_id": self.session_id, "rule": self.rule, "message": self.message}


@dataclass
class LiveSnapshot:
    taken_at: datetime
    zone: str
    sessions: list[ActiveSession]
    alerts: list[Alert]
    hook_events: list[dict[str, object]] = field(default_factory=list)  # newest first
    hook_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "taken_at": self.taken_at.isoformat(),
            "zone": self.zone,
            "sessions": [s.as_dict() for s in self.sessions],
            "alerts": [a.as_dict() for a in self.alerts],
            "hook_events": list(self.hook_events),
            "hook_counts": dict(self.hook_counts),
        }


class LiveMonitor:
    """Take snapshots of active sessions and derive alerts."""

    def __init__(self, root: Path, thresholds: Thresholds, events_path: Path = EVENTS_PATH, prices: PriceTable | None = None) -> None:
        self._root = root
        self._th = thresholds
        self._events_path = events_path
        self._prices = prices or PriceTable()

    def snapshot(self, now: datetime | None = None) -> LiveSnapshot:
        now = now or datetime.now(timezone.utc)
        active_cutoff = now - timedelta(minutes=self._th.live_active_minutes)
        window_start = now - timedelta(minutes=self._th.live_window_minutes)
        recent = self._recent_sessions(active_cutoff)
        mains = [s for s in recent if not s.is_subagent]
        subs_by_parent: dict[str, list[Session]] = {}
        for s in recent:
            if s.is_subagent and s.parent_session_id:
                subs_by_parent.setdefault(s.parent_session_id, []).append(s)
        # A parent whose own file is quiet but whose subagents are busy is still active.
        known = {s.session_id for s in mains}
        for parent_id in subs_by_parent:
            if parent_id not in known:
                parent = self._find_parent(parent_id)
                if parent is not None:
                    mains.append(parent)
        sessions = [self._describe(s, subs_by_parent.get(s.session_id, []), now, window_start) for s in mains]
        # mtime only pre-filters files; real activity is the last turn's timestamp.
        active_seconds = self._th.live_active_minutes * 60
        alert_seconds = self._th.live_window_minutes * 60
        sessions = [s for s in sessions if s.turns_total and s.seconds_idle <= active_seconds]
        alerts: list[Alert] = []
        for s in sessions:
            session_alerts = self._alerts_for(s) if s.seconds_idle <= alert_seconds else []
            alerts.extend(session_alerts)
            s.zone = max((a.level for a in session_alerts), key=lambda z: ZONE_ORDER[z], default="green")
        sessions.sort(key=lambda s: s.last_activity, reverse=True)
        zone = max((a.level for a in alerts), key=lambda z: ZONE_ORDER[z], default="green")
        events = read_hook_events(self._events_path, since=active_cutoff)
        counts = {p: 0 for p in HOOK_PERMISSIONS}
        per_session: dict[str, dict[str, int]] = {}
        for event in events:
            permission = str(event.get("permission") or "allow")
            counts[permission] = counts.get(permission, 0) + 1
            sid = str(event.get("session_id") or "")
            if sid and permission != "allow":
                per_session.setdefault(sid, {})[permission] = per_session.setdefault(sid, {}).get(permission, 0) + 1
        for s in sessions:
            s.brake = per_session.get(s.session_id, {})
        return LiveSnapshot(
            taken_at=now, zone=zone, sessions=sessions, alerts=alerts,
            hook_events=[e for e in events if e.get("permission") != "allow"][:RECENT_HOOK_EVENTS],
            hook_counts=counts,
        )

    def _recent_sessions(self, cutoff: datetime) -> list[Session]:
        if not self._root.is_dir():
            raise TranscriptError(f"transcript root does not exist: {self._root}")
        stamp = cutoff.timestamp()
        out: list[Session] = []
        for path in self._root.rglob("*.jsonl"):
            if path.stat().st_mtime >= stamp:
                out.append(parse_session(path, self._root))
        return out

    def _find_parent(self, session_id: str) -> Session | None:
        for path in self._root.rglob(f"{session_id}.jsonl"):
            return parse_session(path, self._root)
        return None

    def _describe(self, session: Session, subs: list[Session], now: datetime, window_start: datetime) -> ActiveSession:
        turns = session.turns
        last = turns[-1] if turns else None
        in_window = [t for t in turns if t.timestamp >= window_start]
        sub_in_window = [t for s in subs for t in s.turns if t.timestamp >= window_start]
        tokens_window = sum(t.usage.total for t in in_window) + sum(t.usage.total for t in sub_in_window)
        minutes = max(self._th.live_window_minutes, 1)
        recent_tools: list[str] = []
        for turn in reversed(turns):
            for call in reversed(turn.tool_calls):
                recent_tools.append(_describe_call(call.name, call.file_path, call.command_head))
                if len(recent_tools) >= RECENT_TOOLS:
                    break
            if len(recent_tools) >= RECENT_TOOLS:
                break
        last_text = next((t.text_preview for t in reversed(turns) if t.text_preview), "")
        last_activity = max([t.timestamp for t in turns] + [t.timestamp for s in subs for t in s.turns], default=now)
        waste = _window_waste(in_window, self._th)
        cheap_premium = sum(
            1 for s in subs
            if s.turns and self._th.is_premium(s.turns[-1].model) and TASK_TIER.get(classify_task(s.first_prompt), TIER_PREMIUM) != TIER_PREMIUM
        )
        return ActiveSession(
            session_id=session.session_id,
            project=session.project,
            model=last.model if last else "?",
            last_prompt=session.last_prompt or session.first_prompt,
            last_text=last_text,
            last_activity=last_activity,
            seconds_idle=(now - last_activity).total_seconds(),
            context_now=last.usage.context if last else 0,
            turns_total=len(turns),
            turns_in_window=len(in_window),
            tokens_in_window=tokens_window,
            tokens_per_min=tokens_window / minutes,
            recent_tools=recent_tools,
            subagents_active=len(subs),
            subagent_models=sorted({s.turns[-1].model for s in subs if s.turns}),
            waste=waste,
            cheap_task_premium_subagents=cheap_premium,
            compaction=compaction_breakeven(
                last.usage.context if last else 0,
                self._prices.for_model(last.model if last else ""),
                self._th,
                context_growth(turns),
            ),
        )

    def _alerts_for(self, s: ActiveSession) -> list[Alert]:
        """Red means wasteful, not long. A big context with clean recent turns is a cost note, not an alarm."""
        th = self._th
        alerts: list[Alert] = []
        sid = s.session_id[:8]
        high = th.context_tokens * th.live_context_high_multiplier
        if s.context_now >= high and s.waste:
            alerts.append(Alert(
                f"{s.session_id}:context-waste", "red", s.session_id, "context-waste",
                f"{sid}: {s.context_now:,} tokens of context and the last few minutes show waste ({'; '.join(s.waste[:2])}). "
                "Finish this step, then paste the handoff brief into a new session. Don't restart per prompt; do cut at the next natural break.",
            ))
        elif s.context_now >= high:
            alerts.append(Alert(
                f"{s.session_id}:context-expensive", "amber", s.session_id, "context-expensive",
                f"{sid}: context is {s.context_now:,} tokens; every turn re-reads it, so each turn now costs about {s.context_now // 1000}k. "
                "Fine while you are mid-feature. At the next natural break, hand off to a new session (the brief carries the context).",
            ))
        if s.tokens_per_min >= th.live_burn_high_per_min:
            alerts.append(Alert(f"{s.session_id}:burn-high", "red", s.session_id, "burn-high", f"{sid}: burning {s.tokens_per_min:,.0f} tokens/min. Don't let it run unattended. Do: check what it is doing and stop it if it is looping."))
        elif s.tokens_per_min >= th.live_burn_warn_per_min:
            alerts.append(Alert(f"{s.session_id}:burn", "amber", s.session_id, "burn", f"{sid}: {s.tokens_per_min:,.0f} tokens/min, above the {th.live_burn_warn_per_min:,} warning line. Do: glance at its last tools."))
        if s.cheap_task_premium_subagents:
            premium = ", ".join(m for m in s.subagent_models if th.is_premium(m))
            alerts.append(Alert(
                f"{s.session_id}:subagent-premium", "amber", s.session_id, "subagent-premium",
                f"{sid}: {s.cheap_task_premium_subagents} lookup/research subagent(s) on {premium}. A cheaper tier is a candidate for evaluation; equivalent quality has not been measured. "
                "Do: compare successful outcomes and rework before changing models. Preserve the current model on complex tasks.",
            ))
        return alerts


def _window_waste(turns: list, th: Thresholds) -> list[str]:  # noqa: ANN001 - list[Turn]
    """Concrete, recent waste: repeated reads of one file, oversized payloads, image re-reads."""
    reads: dict[str, int] = {}
    big = 0
    images = 0
    for turn in turns:
        for call in turn.tool_calls:
            if call.name == "Read" and call.file_path:
                reads[call.file_path] = reads.get(call.file_path, 0) + 1
                if Path(call.file_path).suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp"} and reads[call.file_path] > 1:
                    images += 1
            if call.result_bytes > th.large_payload_bytes:
                big += 1
    out: list[str] = []
    repeated = [(Path(p).name, n) for p, n in reads.items() if n >= 3]
    if repeated:
        name, n = max(repeated, key=lambda kv: kv[1])
        out.append(f"{name} read {n}x")
    if big >= 2:
        out.append(f"{big} tool results over {th.large_payload_bytes // 1000} KB")
    if images:
        out.append(f"{images} image re-read(s)")
    return out


def read_hook_events(path: Path, since: datetime) -> list[dict[str, object]]:
    """Hook decisions logged since ``since``, newest first. Reads only the file's tail."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - HOOK_EVENTS_TAIL_BYTES))
            tail = handle.read().decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return []
    out: list[dict[str, object]] = []
    for line in tail.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        ts = event.get("ts")
        try:
            when = datetime.fromisoformat(str(ts)) if ts else None
        except ValueError:
            when = None
        if when is None or when < since:
            continue
        event["_when"] = when
        out.append(event)
    out.sort(key=lambda e: e["_when"], reverse=True)
    for event in out:
        event.pop("_when", None)
    return out


class AlertNotifier(threading.Thread):
    """Background loop: snapshot every interval, notify once per new alert key."""

    def __init__(self, monitor: LiveMonitor, interval_seconds: float, notify: bool) -> None:
        super().__init__(name="burnlens-alerts", daemon=True)
        self.monitor = monitor
        self._interval = interval_seconds
        self._notify = notify
        self._seen: set[str] = set()
        self._stop = threading.Event()
        self.latest: LiveSnapshot | None = None
        self.lock = threading.Lock()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                snap = self.monitor.snapshot()
                with self.lock:
                    self.latest = snap
                self._announce(snap)
            except Exception:  # keep the loop alive; the next tick may succeed
                logger.exception("live snapshot failed")
            self._stop.wait(self._interval)

    def stop(self) -> None:
        self._stop.set()

    def _announce(self, snap: LiveSnapshot) -> None:
        current = {a.key for a in snap.alerts}
        self._seen &= current  # forget alerts that cleared, so they can fire again later
        for alert in snap.alerts:
            if alert.key in self._seen:
                continue
            self._seen.add(alert.key)
            logger.warning("ALERT %s %s", alert.level.upper(), alert.message)
            if self._notify:
                notify_desktop(f"{alert.level.upper()} zone", alert.message)


def notify_desktop(title: str, message: str) -> None:
    """Best-effort OS notification. macOS only for now; silently no-op elsewhere."""
    if platform.system() != "Darwin":
        return
    script = f'display notification "{_osa_escape(message)}" with title "{_osa_escape(title)}" sound name "Basso"'
    try:
        subprocess.run(["osascript", "-e", script], check=False, timeout=5, capture_output=True)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("desktop notification failed: %s", exc)


def _osa_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _describe_call(name: str, file_path: str | None, command_head: str | None) -> str:
    if file_path:
        return f"{name} {Path(file_path).name}"
    if command_head:
        return f"{name} {command_head[:48]}"
    return name
