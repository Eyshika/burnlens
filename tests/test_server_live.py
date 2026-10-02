"""Server endpoints and live monitor on a synthetic tree with fresh timestamps."""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from burnlens.findings import Thresholds
from burnlens.live import AlertNotifier, LiveMonitor
from burnlens.server import DashboardServer

from test_profiler import _assistant, _prompt, _tool_result, _tool_use, _write

HUGE_CONTEXT = 400_000


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.fixture()
def live_tree(tmp_path: Path) -> Path:
    root = tmp_path / "projects"
    now = datetime.now(timezone.utc)
    recs = [_prompt("refactor the loader", _iso(now - timedelta(minutes=4)))]
    for i in range(10):
        ts = _iso(now - timedelta(minutes=4) + timedelta(seconds=20 * i))
        tools = [_tool_use(f"e{i}", "Edit", {"file_path": "/repo/loader.py"}), _tool_use(f"r{i}", "Read", {"file_path": "/repo/loader.py"})]
        recs.append(_assistant(f"a{i}", ts, "claude-opus-5", HUGE_CONTEXT, 200, tools))
        recs.append(_tool_result(f"e{i}", ts, "ok"))
        recs.append(_tool_result(f"r{i}", ts, "x" * 500))
    recs.append(_prompt("now add tests", _iso(now - timedelta(seconds=30))))
    _write(root / "-Users-me-repo" / "sess-live.jsonl", recs)
    sub = [_prompt("find where the retry logic is defined", _iso(now - timedelta(seconds=61)))]
    sub += [_assistant(f"s{i}", _iso(now - timedelta(seconds=60 - i)), "claude-opus-5", 50_000, 40) for i in range(5)]
    _write(root / "-Users-me-repo" / "sess-live" / "subagents" / "agent-x.jsonl", sub)
    # A clean big-context session mid-feature: edits only, no re-reads, no dumps. Expensive, not red.
    clean = [_prompt("add the export endpoint", _iso(now - timedelta(minutes=3)))]
    for i in range(6):
        ts = _iso(now - timedelta(minutes=3) + timedelta(seconds=25 * i))
        clean.append(_assistant(f"c{i}", ts, "claude-opus-5", HUGE_CONTEXT, 300, [_tool_use(f"ce{i}", "Edit", {"file_path": f"/repo/api/export_{i}.py"})]))
        clean.append(_tool_result(f"ce{i}", ts, "ok"))
    _write(root / "-Users-me-repo" / "sess-clean.jsonl", clean)
    # An old session with a FRESH mtime (as after a restore or sync): old turn timestamps
    # must keep it out of the live view.
    old = root / "-Users-me-repo" / "sess-old.jsonl"
    _write(old, [_assistant("o0", "2026-01-01T00:00:00Z", "claude-sonnet-5", 1_000, 5)])
    # A session that was busy 8 minutes ago: shown as active, but no alerts (idle beyond the window).
    stale = [_prompt("earlier task", _iso(now - timedelta(minutes=9)))]
    stale.append(_assistant("q0", _iso(now - timedelta(minutes=8)), "claude-opus-5", HUGE_CONTEXT, 50))
    _write(root / "-Users-me-repo" / "sess-quiet.jsonl", stale)
    return root


def test_live_snapshot_includes_hook_decisions(live_tree: Path, tmp_path: Path) -> None:
    events = tmp_path / "hook-events.jsonl"
    now = datetime.now(timezone.utc)
    rows = [
        {"ts": (now - timedelta(minutes=1)).isoformat(), "session_id": "sess-live", "tool": "Agent", "permission": "deny", "reason": "premium subagent", "target": "research x"},
        {"ts": (now - timedelta(minutes=2)).isoformat(), "session_id": "sess-live", "tool": "Read", "permission": "ask", "reason": "big file", "target": "server.py"},
        {"ts": (now - timedelta(minutes=3)).isoformat(), "session_id": "sess-live", "tool": "Bash", "permission": "allow", "reason": "", "target": "pytest -q"},
        {"ts": (now - timedelta(minutes=4)).isoformat(), "session_id": "other", "tool": "UserPromptSubmit", "permission": "coach", "reason": "Burnlens: lookup -> haiku", "target": ""},
        {"ts": (now - timedelta(hours=3)).isoformat(), "session_id": "sess-live", "tool": "Agent", "permission": "deny", "reason": "old", "target": ""},
    ]
    events.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    snap = LiveMonitor(live_tree, Thresholds(), events_path=events).snapshot()
    assert snap.hook_counts["deny"] == 1 and snap.hook_counts["ask"] == 1 and snap.hook_counts["allow"] == 1 and snap.hook_counts["coach"] == 1
    assert [e["permission"] for e in snap.hook_events] == ["deny", "ask", "coach"], "newest first, allows hidden, old dropped"
    live = next(s for s in snap.sessions if s.session_id == "sess-live")
    assert live.brake == {"deny": 1, "ask": 1}
    assert snap.as_dict()["hook_events"][0]["target"] == "research x"


def test_live_snapshot_describes_active_session_and_alerts(live_tree: Path) -> None:
    th = Thresholds(live_burn_warn_per_min=100_000, live_burn_high_per_min=10_000_000)
    snap = LiveMonitor(live_tree, th).snapshot()
    assert {s.session_id for s in snap.sessions} == {"sess-live", "sess-clean", "sess-quiet"}, "old timestamps beat a fresh mtime"
    assert not any(a.session_id == "sess-quiet" for a in snap.alerts), "idle sessions are shown but do not alert"
    clean = {a.rule: a.level for a in snap.alerts if a.session_id == "sess-clean"}
    assert clean.get("context-expensive") == "amber" and "context-waste" not in clean, "big but clean context is a cost note, never red"
    assert next(x for x in snap.sessions if x.session_id == "sess-clean").zone != "red"
    s = next(x for x in snap.sessions if x.session_id == "sess-live")
    assert s.context_now == HUGE_CONTEXT + 1_000 + 5
    assert s.last_prompt == "now add tests"
    assert s.recent_tools[0] == "Read loader.py" and "Edit loader.py" in s.recent_tools
    assert s.subagents_active == 1 and s.subagent_models == ["claude-opus-5"]
    assert s.turns_in_window == 10 and s.tokens_per_min > 0
    assert s.waste and s.waste[0].startswith("loader.py read")
    rules = {a.rule: a.level for a in snap.alerts if a.session_id == "sess-live"}
    assert rules["context-waste"] == "red", "400k context plus re-reads is red"
    assert rules["subagent-premium"] == "amber" and s.cheap_task_premium_subagents == 1
    assert rules["burn"] == "amber"
    assert snap.zone == "red" and s.zone == "red"


def test_notifier_announces_each_alert_once(live_tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[str] = []
    monkeypatch.setattr("burnlens.live.notify_desktop", lambda title, msg: sent.append(msg))
    notifier = AlertNotifier(LiveMonitor(live_tree, Thresholds()), interval_seconds=60, notify=True)
    snap = notifier.monitor.snapshot()
    notifier._announce(snap)
    notifier._announce(snap)
    assert len(sent) == len(snap.alerts) > 0, "second identical snapshot must not re-notify"


def test_http_endpoints(live_tree: Path) -> None:
    server = DashboardServer("127.0.0.1", 0, live_tree, Thresholds(), notify=False)
    port = server.server_address[1]
    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        def get(path: str):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as res:
                return res.status, res.headers.get("Content-Type", ""), res.read()

        status, ctype, body = get("/")
        assert status == 200 and "text/html" in ctype and b"Live now" in body
        status, ctype, body = get("/app.js")
        assert status == 200 and "javascript" in ctype
        status, _, body = get("/api/meta?days=7")
        assert status == 200 and json.loads(body)["app"]
        status, _, body = get("/api/report?days=0")
        report = json.loads(body)
        assert status == 200 and report["sessions"] == 4 and report["subagents"] == 1
        status, _, body = get("/api/session/sess-li?days=0")
        detail = json.loads(body)
        assert status == 200 and detail["turns"] == 10 and len(detail["subagents"]) == 1
        status, _, body = get("/api/graph?days=0")
        assert status == 200 and any(n["type"] == "session" for n in json.loads(body)["nodes"])
        status, _, body = get("/api/live")
        assert status == 200 and {s["session_id"] for s in json.loads(body)["sessions"]} >= {"sess-live", "sess-clean"}
        with pytest.raises(urllib.error.HTTPError) as exc:
            get("/../pyproject.toml")
        assert exc.value.code == 404
    finally:
        server.shutdown()
        server.server_close()


def test_live_session_carries_compaction_payback(live_tree: Path) -> None:
    """The live card prices compaction per session; without prices it would claim nothing."""
    from burnlens.config import load_published_prices

    snap = LiveMonitor(live_tree, Thresholds(), prices=load_published_prices()).snapshot()
    session = next(s for s in snap.sessions if not s.session_id.startswith("agent"))
    payload = session.as_dict()["compaction"]
    assert isinstance(payload, dict)
    assert payload["cache_write_read_ratio"] == 12.5  # opus-5, from prices.toml
    context_now = HUGE_CONTEXT + 1_005  # the fixture's usage: 5 fresh + 1,000 written + HUGE_CONTEXT cached
    assert payload["context_now"] == context_now and payload["saved_per_request"] == context_now - 21_000
    assert round(payload["breakeven_requests"], 1) == 1.8
    assert payload["savings_status"] == "projected"
