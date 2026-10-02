"""Pre-execution guard decisions and settings installation."""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

import pytest

from burnlens.findings import Thresholds
from burnlens.hook import HookGuard, install_hooks, read_context_now, run_hook

from test_profiler import _assistant, _write

TH = Thresholds(context_tokens=150_000, large_payload_bytes=50_000)


def _transcript(tmp_path: Path, context: int) -> Path:
    path = tmp_path / "sess.jsonl"
    _write(path, [_assistant("m0", "2026-09-01T10:00:00Z", "claude-opus-5", 10, 5), _assistant("m1", "2026-09-01T10:01:00Z", "claude-opus-5", context, 5)])
    return path


def _payload(tool: str, tool_input: dict, transcript: Path | None) -> dict:
    return {"session_id": "sess", "hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": tool_input, "transcript_path": str(transcript) if transcript else None}


def test_read_context_now_takes_latest_turn(tmp_path: Path) -> None:
    assert read_context_now(_transcript(tmp_path, 400_000)) == 400_000 + 1_000 + 5
    assert read_context_now(tmp_path / "missing.jsonl") == 0


def test_legacy_autofix_preserves_tool_inputs(tmp_path: Path) -> None:
    guard = HookGuard(TH)
    big = tmp_path / "big.py"
    big.write_text("x" * 120_000)
    cases = [
        ("Agent", {"prompt": "research x", "model": "opus"}),
        ("Read", {"file_path": str(big)}),
        ("Bash", {"command": "cat logs/pipeline.log"}),
        ("Bash", {"command": "cat missing-file\necho done"}),
    ]
    for tool, tool_input in cases:
        original = dict(tool_input)
        decision = guard.decide(_payload(tool, tool_input, _transcript(tmp_path, 160_000)))
        assert decision.updated_input is None
        assert "updatedInput" not in decision.as_hook_output()["hookSpecificOutput"]
        assert tool_input == original


@pytest.mark.parametrize("command", ["cat missing-file", "cat missing-file && echo success", "cat missing-file; exit 7"])
def test_hook_preserves_failing_shell_status(tmp_path: Path, command: str) -> None:
    payload = _payload("Bash", {"command": command}, _transcript(tmp_path, 160_000))
    output = run_hook(json.dumps(payload), TH, events_path=tmp_path / "events.jsonl")
    effective = output["hookSpecificOutput"].get("updatedInput", payload["tool_input"])["command"]
    assert effective == command
    before = subprocess.run(["/bin/sh", "-c", command], cwd=tmp_path, capture_output=True)
    after = subprocess.run(["/bin/sh", "-c", effective], cwd=tmp_path, capture_output=True)
    assert before.returncode != 0
    assert (after.returncode, after.stdout, after.stderr) == (before.returncode, before.stdout, before.stderr)


def test_agent_spawn_without_model_escalates_with_zone(tmp_path: Path) -> None:
    guard = HookGuard(TH, autofix=False)
    hard = guard.decide(_payload("Agent", {"prompt": "debug why the loader crashes under load"}, _transcript(tmp_path, 160_000)))
    assert hard.permission == "allow", "a debugging subagent may stay on the premium model"
    green = guard.decide(_payload("Agent", {"prompt": "research x"}, _transcript(tmp_path, 20_000)))
    assert green.permission == "allow" and "sonnet" in green.context_note, "green zone nudges, never interrupts"
    amber = guard.decide(_payload("Agent", {"prompt": "research x"}, _transcript(tmp_path, 160_000)))
    assert amber.permission == "ask" and "sonnet" in amber.reason
    red = guard.decide(_payload("Agent", {"prompt": "research x"}, _transcript(tmp_path, 400_000)))
    assert red.permission == "ask" and "red zone" in red.reason
    ok = guard.decide(_payload("Agent", {"prompt": "research x", "model": "sonnet"}, _transcript(tmp_path, 20_000)))
    assert ok.permission == "allow" and "flat" in ok.context_note
    strict = HookGuard(TH, strict=True).decide(_payload("Agent", {"prompt": "research x"}, _transcript(tmp_path, 20_000)))
    assert strict.permission == "ask", "strict mode puts the human in the loop even in green"


def test_large_read_escalates_with_zone(tmp_path: Path) -> None:
    big = tmp_path / "big.py"
    big.write_text("x" * 120_000)
    guard = HookGuard(TH, autofix=False)
    assert guard.decide(_payload("Read", {"file_path": str(big)}, _transcript(tmp_path, 20_000))).permission == "allow"
    assert guard.decide(_payload("Read", {"file_path": str(big)}, _transcript(tmp_path, 160_000))).permission == "ask"
    assert guard.decide(_payload("Read", {"file_path": str(big)}, _transcript(tmp_path, 400_000))).permission == "deny"
    ranged = guard.decide(_payload("Read", {"file_path": str(big), "offset": 10, "limit": 40}, _transcript(tmp_path, 400_000)))
    assert ranged.permission == "allow", "a line-range read is the fix, never blocked"


def test_verbose_bash_is_flagged_unless_limited(tmp_path: Path) -> None:
    guard = HookGuard(TH, autofix=False)
    amber = _transcript(tmp_path, 160_000)
    assert guard.decide(_payload("Bash", {"command": "cat logs/pipeline.log"}, amber)).permission == "ask"
    assert guard.decide(_payload("Bash", {"command": "cat logs/pipeline.log | tail -50"}, amber)).permission == "allow"
    assert guard.decide(_payload("Bash", {"command": "pytest -q tests"}, amber)).permission == "allow"
    assert guard.decide(_payload("Bash", {"command": "pytest tests"}, amber)).permission == "ask"
    green = guard.decide(_payload("Bash", {"command": "cat logs/pipeline.log"}, _transcript(tmp_path, 1_000)))
    assert green.permission == "allow" and "Do:" in green.context_note and "exit status" in green.context_note


def test_run_hook_outputs_claude_code_shape_and_logs(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    out = run_hook(json.dumps(_payload("Agent", {"prompt": "research x"}, _transcript(tmp_path, 400_000))), TH, events_path=events, autofix=False)
    spec = out["hookSpecificOutput"]
    assert spec["hookEventName"] == "PreToolUse" and spec["permissionDecision"] == "ask" and spec["permissionDecisionReason"]
    assert out["systemMessage"]
    first = json.loads(events.read_text().splitlines()[0])
    assert first["permission"] == "ask" and first["tool"] == "Agent"
    assert run_hook("not json", TH, events_path=events)["hookSpecificOutput"]["permissionDecision"] == "allow"


def test_hook_fails_open_on_hostile_input(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    bad_path = {"tool_name": "Read", "tool_input": {"file_path": "abc\x00def"}, "transcript_path": "x\x00y"}
    assert run_hook(json.dumps(bad_path), TH, events_path=events)["hookSpecificOutput"]["permissionDecision"] == "allow"
    weird = tmp_path / "weird.jsonl"
    weird.write_text('["usage", 1]\n{"type":"assistant","message":{"usage":{"cache_read_input_tokens": 7}}}\n')
    assert read_context_now(weird) == 7
    assert run_hook(json.dumps({"tool_name": "Bash", "tool_input": "not a dict"}), TH, events_path=events)["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert run_hook("[1,2,3]", TH, events_path=events)["hookSpecificOutput"]["permissionDecision"] == "allow"


def test_install_hooks_is_idempotent_and_reversible(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"model": "opus", "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "echo done"}]}]}}))
    install_hooks(settings)
    install_hooks(settings)
    data = json.loads(settings.read_text())
    assert data["model"] == "opus" and "Stop" in data["hooks"], "existing settings survive"
    assert len(data["hooks"]["PreToolUse"]) == 1 and "-m burnlens hook" in data["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    assert settings.with_suffix(".json.bak").exists()
    install_hooks(settings, strict=True)
    assert json.loads(settings.read_text())["hooks"]["PreToolUse"][0]["hooks"][0]["command"].endswith("--strict")
    install_hooks(settings, autofix=False)
    assert json.loads(settings.read_text())["hooks"]["PreToolUse"][0]["hooks"][0]["command"].endswith("--no-autofix")
    install_hooks(settings, remove=True)
    assert "PreToolUse" not in json.loads(settings.read_text())["hooks"]


def test_install_replaces_a_hook_registered_under_the_old_name(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    old = {"hooks": [{"type": "command", "command": "/usr/bin/python3 -m tokprof hook"}]}
    settings.write_text(json.dumps({"hooks": {"PreToolUse": [old]}}))
    install_hooks(settings)
    commands = [h["command"] for e in json.loads(settings.read_text())["hooks"]["PreToolUse"] for h in e["hooks"]]
    assert len(commands) == 1 and "-m burnlens hook" in commands[0]


def test_hook_uses_configured_task_tier_without_switching(tmp_path: Path) -> None:
    output = run_hook(json.dumps(_payload("Agent", {"prompt": "research x"}, None)), TH,
                      events_path=tmp_path / "events.jsonl", task_tiers={"research": "haiku"})
    spec = output["hookSpecificOutput"]
    assert "haiku" in spec["additionalContext"]
    assert "equivalent quality is not established" in spec["additionalContext"]
    assert "updatedInput" not in spec


def test_unknown_and_complex_agents_preserve_model(tmp_path: Path) -> None:
    for prompt in ("x", "Find the root cause of a deadlock and design a safe fix"):
        decision = HookGuard(TH).decide(_payload("Agent", {"prompt": prompt, "model": "opus"}, None))
        assert decision.updated_input is None
        assert "Consider evaluating" not in decision.context_note


def test_install_preserves_quoted_absolute_config(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    config = tmp_path / "team's $(echo secret) config.toml"
    install_hooks(settings, config_path=config)
    for event in ("PreToolUse", "UserPromptSubmit"):
        command = json.loads(settings.read_text())["hooks"][event][0]["hooks"][0]["command"]
        args = shlex.split(command)
        assert args[args.index("--config") + 1] == str(config.resolve())


def test_hook_event_preserves_diagnostics_without_claiming_savings(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    run_hook(json.dumps(_payload("Bash", {"command": "cat build.log"}, None)), TH, events_path=events)
    event = json.loads(events.read_text())
    assert event["intervention_status"] == "advised"
    assert event["input_changed"] is False
    assert event["savings_verified"] is False
    assert event["target"] == "cat build.log"
    assert "large output" in event["reason"]


def test_prompt_hook_uses_configured_task_tier(tmp_path: Path) -> None:
    payload = {
        "session_id": "configured-tier",
        "hook_event_name": "UserPromptSubmit",
        "prompt": "research x",
        "transcript_path": str(_transcript(tmp_path, 20_000)),
    }
    output = run_hook(json.dumps(payload), TH, events_path=tmp_path / "events.jsonl",
                      state_dir=tmp_path / "state", task_tiers={"research": "haiku"})
    assert "haiku" in json.dumps(output).lower()


def _repeat_transcript(directory: Path, repeats: int, *, measured: bool = True, cmd: str = "pytest") -> str:
    """A transcript whose tail is `repeats` identical Bash calls, each returning the same bytes."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "t.jsonl"
    lines = []
    for i in range(repeats):
        call_id = f"c{i}"
        lines.append(json.dumps({
            "type": "assistant",
            "timestamp": "2026-09-13T10:00:00Z",
            "message": {"id": f"m{i}", "model": "claude-opus-5",
                        "usage": {"input_tokens": 1, "cache_read_input_tokens": 50_000, "output_tokens": 10},
                        "content": [{"type": "tool_use", "id": call_id, "name": "Bash", "input": {"command": cmd}}]},
        }))
        if measured:
            lines.append(json.dumps({
                "type": "user",
                "message": {"content": [{"type": "tool_result", "tool_use_id": call_id, "content": "x" * 40_000}]},
            }))
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_loop_guard_denies_the_next_identical_call_and_records_what_it_prevented(tmp_path: Path) -> None:
    import burnlens.hook as hook_module
    from burnlens.ledger import Ledger

    ledger_path = tmp_path / "prevented.jsonl"
    payload = {
        "tool_name": "Bash", "tool_input": {"command": "pytest"},
        "transcript_path": _repeat_transcript(tmp_path / "run", 3),
        "session_id": "s1", "cwd": "/repo",
    }
    original = hook_module.Ledger
    hook_module.Ledger = lambda: Ledger(ledger_path)
    try:
        decision = HookGuard(Thresholds()).decide(payload)
    finally:
        hook_module.Ledger = original

    assert decision.permission == "deny"
    assert "already run 3 times" in decision.reason
    assert decision.announcement().startswith("BLOCKED by Burnlens - this call did not run.")

    rows = Ledger(ledger_path).rows()
    assert len(rows) == 1 and rows[0].rule == "loop-guard"
    assert rows[0].bytes_prevented == 40_000 and rows[0].tokens_prevented == 10_000
    assert "3 identical prior results" in rows[0].basis


def test_loop_guard_stays_out_of_the_way_without_a_measurement(tmp_path: Path) -> None:
    """Three boundaries: too short a run, results never measured, and a different command."""
    guard = HookGuard(Thresholds())

    def decide(directory: str, *, repeats: int, measured: bool = True, cmd: str = "pytest") -> str:
        return guard.decide({
            "tool_name": "Bash", "tool_input": {"command": "pytest"},
            "transcript_path": _repeat_transcript(tmp_path / directory, repeats, measured=measured, cmd=cmd),
            "session_id": "s",
        }).permission

    assert decide("short", repeats=2) != "deny"
    assert decide("unmeasured", repeats=4, measured=False) != "deny"
    assert decide("other", repeats=4, cmd="ruff check") != "deny"


def test_ledger_refuses_a_row_with_no_measured_size(tmp_path: Path) -> None:
    import dataclasses
    from datetime import datetime, timezone

    from burnlens.ledger import Ledger, PreventedLoad

    ledger = Ledger(tmp_path / "p.jsonl")
    row = PreventedLoad(at=datetime(2026, 9, 13, tzinfo=timezone.utc), rule="loop-guard", target="Bash: pytest",
                        bytes_prevented=40_000, divisor=4, session_id="s", repo="r", decision="deny", basis="b")
    assert ledger.append(row) is True
    assert ledger.append(dataclasses.replace(row, bytes_prevented=0)) is False

    summary = ledger.summary()
    assert summary["events"] == 1
    assert summary["tokens_prevented"] == 10_000
    assert summary["evidence_class"] == "measured"
