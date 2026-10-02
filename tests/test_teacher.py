"""The teacher: repeated instructions -> CLAUDE.md, repeated tasks -> skill, repeated research -> saved doc."""

from __future__ import annotations

import json
from pathlib import Path

from burnlens.adapters import load_generic
from burnlens.teacher import lessons
from burnlens.transcripts import load_sessions

from test_profiler import _assistant, _prompt, _tool_result, _tool_use, _write


def _session(root: Path, sid: str, prompts: list[str], day: str, reads: list[str] | None = None) -> None:
    recs = []
    for i, text in enumerate(prompts):
        ts = f"{day}T10:{i:02d}:00Z"
        recs.append(_prompt(text, ts))
        tools = [_tool_use(f"{sid}r{i}{j}", "Read", {"file_path": p}) for j, p in enumerate(reads or [])] if i == 0 else []
        recs.append(_assistant(f"{sid}m{i}", ts, "claude-opus-5", 40_000, 100, tools))
        recs.extend(_tool_result(f"{sid}r{i}{j}", ts, "y" * 30_000) for j, _ in enumerate(reads or []) if i == 0)
    _write(root / "-p" / f"{sid}.jsonl", recs)


def test_lessons_from_repeated_prompts(tmp_path: Path) -> None:
    root = tmp_path / "cc"
    common_reads = ["/repo/product/ui/server.py", "/repo/product/process_mining/step_model.py"]
    for i in range(4):
        _session(root, f"s{i}", [
            "Always use the hss SDK for queues, never import pika directly. Fix the flaky loader test in tests/test_loader.py.",
            "write docs for the new profiler cli and update the readme",
            "research how other teams handle rate limits and compare options",
        ], f"2026-09-0{i + 1}", reads=common_reads)
    _session(root, "s9", ["hello there"], "2026-09-06")
    rows = {l.feature: l for l in lessons(load_sessions(root))}
    assert {"CLAUDE.md", "skill", "saved research", "project map"} <= set(rows)
    claude_md = rows["CLAUDE.md"]
    assert claude_md.occurrences == 4 and "Always use the hss SDK" in claude_md.draft and claude_md.avoidable_tokens is None and claude_md.savings_status == "unmeasured"
    skill = rows["skill"]
    assert skill.occurrences >= 3 and "SKILL.md" in skill.draft and "## Steps" in skill.draft
    research = rows["saved research"]
    assert research.occurrences == 4 and "docs/research/" in research.draft
    pmap = rows["project map"]
    assert "server.py" in pmap.draft and "step_model.py" in pmap.draft
    assert rows == {l.feature: l for l in lessons(load_sessions(root))}, "deterministic"


def test_pasted_content_lesson_and_min_sessions(tmp_path: Path) -> None:
    root = tmp_path / "cc"
    for i in range(2):
        _session(root, f"p{i}", ["here is the log:\n" + "ERROR line\n" * 400], f"2026-09-0{i + 1}")
    rows = {l.feature: l for l in lessons(load_sessions(root))}
    assert "file reference" in rows and rows["file reference"].occurrences == 2
    assert "CLAUDE.md" not in rows, "one-off prompts are not lessons"


def test_generic_records_feed_the_teacher(tmp_path: Path) -> None:
    rows = []
    for i in range(3):
        rows.append({"session_id": f"g{i}", "agent": "codex", "user": "sam", "ts": f"2026-09-0{i + 1}T10:00:00Z", "model": "gpt-5",
                     "usage": {"input_tokens": 10, "output_tokens": 10}, "prompt": "Remember: run pytest -q before every commit. Add tests for the ledger module."})
    (tmp_path / "t.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = {l.feature: l for l in lessons(load_generic(tmp_path))}
    assert out["CLAUDE.md"].people == ["sam"] and "Remember: run pytest -q before every commit" in out["CLAUDE.md"].draft


def test_heavy_claude_md_lesson(tmp_path: Path, monkeypatch) -> None:
    from burnlens import teacher

    project = tmp_path / "Users" / "demo" / "repo"
    project.mkdir(parents=True)
    (project / "CLAUDE.md").write_text("## Rules\n" + "- Never do X. Always do Y.\n" * 400 + "## Review patterns\n" + "text " * 2000)
    monkeypatch.setattr(teacher.Path, "home", staticmethod(lambda: tmp_path / "nohome"))
    monkeypatch.setattr(teacher, "_project_root", lambda slug: project)
    root = tmp_path / "cc"
    _session(root, "h1", ["hello"], "2026-09-01")
    rows = {l.feature: l for l in lessons(load_sessions(root))}
    lesson = rows["lean CLAUDE.md"]
    assert "never/always/must" in lesson.draft and "Review patterns" in lesson.draft and lesson.avoidable_tokens is None and lesson.savings_status == "unmeasured"
    assert any(e.startswith("Rules") for e in lesson.examples)



def _long_session(sid: str, turns: int):
    from datetime import datetime, timezone

    from burnlens.model import Session, Turn, Usage

    return Session(
        session_id=sid,
        project="p",
        path=Path(f"/tmp/{sid}.jsonl"),
        parent_session_id=None,
        first_prompt="go",
        turns=[
            Turn(
                message_id=f"{sid}m{i}",
                timestamp=datetime(2026, 9, 12, 10, i % 60, tzinfo=timezone.utc),
                model="claude-opus-5",
                usage=Usage(cache_creation_input_tokens=1_000, cache_read_input_tokens=200_000 + i, output_tokens=50),
            )
            for i in range(turns)
        ],
    )


def test_long_threads_teach_compact_clear_and_handoff() -> None:
    from burnlens.teacher import LONG_THREAD_TURNS

    sessions = [_long_session(f"s{i}", LONG_THREAD_TURNS + 10) for i in range(3)]
    out = [l for l in lessons(sessions, min_sessions=3) if l.feature == "session reset"]
    assert len(out) == 1
    lesson = out[0]
    assert lesson.occurrences == 3
    assert "/compact" in lesson.draft and "/clear" in lesson.draft and "burnlens handoff" in lesson.draft
    assert lesson.avoidable_tokens is None and lesson.savings_status == "unmeasured"

    # the boundary: one turn under the threshold and the lesson does not fire
    short = [_long_session(f"t{i}", LONG_THREAD_TURNS - 1) for i in range(3)]
    assert [l for l in lessons(short, min_sessions=3) if l.feature == "session reset"] == []
