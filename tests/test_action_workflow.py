"""Exercise the real HTTP/CLI/hook workflow with isolated state and safe commands."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import urllib.error
import urllib.request

import pytest

from burnlens.findings import Thresholds
from burnlens.hook import run_hook
from burnlens.interventions import InterventionConfig, InterventionService
from burnlens.server import DashboardServer
from test_profiler import _assistant, _tool_result, _tool_use, _write


def test_http_cli_review_apply_measure_revert(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = tmp_path / "state"
    monkeypatch.setenv("BURNLENS_STATE_DIR", str(state))
    root = tmp_path / "projects"
    transcript = root / "-personal" / "s.jsonl"
    _write(transcript, [
        _assistant("m", "2026-09-10T10:00:00Z", "model", 1000, 5,
                   [_tool_use("b", "Bash", {"command": "tool diagnostics"})]),
        _tool_result("b", "2026-09-10T10:00:01Z", "x" * 60000),
    ])
    server = DashboardServer("127.0.0.1", 0, root, Thresholds(), notify=False)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def request(path: str, payload: dict[str, object] | None = None, token: str = "") -> dict[str, object]:
        headers = {"Content-Type": "application/json", "X-Burnlens-Token": token}
        data = json.dumps(payload).encode() if payload is not None else None
        with urllib.request.urlopen(urllib.request.Request(base + path, data=data, headers=headers), timeout=5) as response:
            return json.load(response)

    try:
        action = request("/api/actions")["actions"][0]
        token = request("/api/meta")["action_token"]
        body = {"action_id": action["id"], "revision": action["revision"]}
        with pytest.raises(urllib.error.HTTPError) as rejected:
            request("/api/actions/apply", body)
        assert rejected.value.code == 403
        prefix = [sys.executable, "-m", "burnlens", "capture", "--action", action["id"], "--task", "same-test"]
        command = [sys.executable, "-c", "import sys; sys.stdout.write('line\\n'*5000); sys.stderr.write('failure detail\\n'); sys.exit(7)"]
        before = subprocess.run(prefix + ["--baseline", "--"] + command, capture_output=True, env=os.environ.copy())
        assert before.returncode == 7 and b"failure detail" in before.stdout
        assert request("/api/actions/apply", body, token)["action"]["state"] == "applied"
        hook_payload = {"tool_name":"Bash", "tool_input":{"command":"echo hello"}, "transcript_path":str(transcript)}
        guidance = run_hook(json.dumps(hook_payload), Thresholds(), events_path=tmp_path / "events")
        assert "--action" in guidance["hookSpecificOutput"]["additionalContext"]
        assert "updatedInput" not in guidance["hookSpecificOutput"]
        after = subprocess.run(prefix + ["--"] + command, capture_output=True, env=os.environ.copy())
        assert after.returncode == 7 and len(after.stdout) < len(before.stdout)
        service = InterventionService(InterventionConfig(state))
        measured = service.get_action(action["id"])
        assert len(measured.runs) == 2
        trial = measured.runs[-1]
        assert Path(trial.output_path).read_bytes() == before.stdout
        assert trial.displayed_bytes == len(after.stdout)
        assert measured.as_dict()["measurement"]["savings"] == "unmeasured"
        response = request("/api/actions/outcome", {"action_id":action["id"],"run_id":trial.run_id,"outcome":"failed","notes":"test remains failing"}, token)
        assert response["action"]["runs"][-1]["outcome"] == "failed"
        assert request("/api/actions/revert", {"action_id":action["id"]}, token)["action"]["state"] == "reverted"
        assert service.advice_for(str(transcript)) == ""
        refused = subprocess.run(prefix + ["--"] + command, capture_output=True)
        assert refused.returncode == 2
        assert len(service.get_action(action["id"]).runs) == 2
        with pytest.raises(urllib.error.HTTPError):
            request("/api/actions/apply", {"action_id":"../../bad", "revision":"x"}, token)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def test_optional_advice_failure_preserves_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sqlite3

    transcript = tmp_path / "red.jsonl"
    _write(transcript, [_assistant("m", "2026-09-10T10:00:00Z", "model", 400000, 5)])
    def unavailable(self: InterventionService, transcript_path: str) -> str:
        raise sqlite3.OperationalError("locked")
    monkeypatch.setattr(InterventionService, "advice_for", unavailable)
    result = run_hook(json.dumps({"tool_name":"Bash", "tool_input":{"command":"cat large.log"}, "transcript_path":str(transcript)}), Thresholds(), events_path=tmp_path / "events")
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "no command changes" in result["hookSpecificOutput"]["permissionDecisionReason"]


@pytest.mark.parametrize("exit_code", [0, 7])
def test_measurement_failure_preserves_completed_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, exit_code: int) -> None:
    import argparse
    import sqlite3
    from types import SimpleNamespace
    from burnlens.capture import CaptureResult
    from burnlens.cli import _capture_command
    from burnlens.interventions import InterventionService

    monkeypatch.setattr(InterventionService, "get_action", lambda self, action_id: SimpleNamespace(state="applied", revision="v1"))
    def fail_write(self: InterventionService, action_id: str, measurement: object) -> None:
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(InterventionService, "record_run", fail_write)
    output = str(tmp_path / "retained.log")
    monkeypatch.setattr("burnlens.cli.CaptureRunner.run", lambda self, command, baseline: CaptureResult(
        "run", exit_code, 0.1, 10, 3, output, "abc", False))
    args = argparse.Namespace(argv=["--", "unused"], action="action", task="task", baseline=False)
    assert _capture_command(args) == exit_code
    assert "measurement was not saved" in caplog.text
    assert output in caplog.text
