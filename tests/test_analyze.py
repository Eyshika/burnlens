"""Digest building and model-response parsing with a fake transport (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from burnlens.analyze import AnalysisError, LLMConfig, SessionNarrator, build_digest, render_explanation
from burnlens.transcripts import load_sessions

from test_profiler import tree  # noqa: F401 - fixture

FAKE_ANSWER = {
    "narrative": "The session re-read server.py on every turn.",
    "drift": [{"turns": "3-29", "kind": "re-reading", "evidence": "Read(server.py) appears on 27 consecutive turns."}],
    "cut_point": 12,
    "dont": ["Don't let the agent re-read server.py whole."],
    "do": ["Do open server.py yourself and paste the loader function."],
    "better_prompt": "Fix the bug in server.py lines 120-180; run pytest -q and show only failures.",
}


def _fake_transport(calls: list[dict]):
    def post(url: str, headers: dict, body: dict, timeout: int) -> dict:
        calls.append({"url": url, "headers": headers, "body": body})
        if url.endswith("/messages"):
            return {"content": [{"type": "text", "text": json.dumps(FAKE_ANSWER)}]}
        return {"choices": [{"message": {"content": "```json\n" + json.dumps(FAKE_ANSWER) + "\n```"}}]}

    return post


def test_digest_has_prompts_tools_and_no_tool_output(tree: Path) -> None:
    sessions = {s.session_id: s for s in load_sessions(tree)}
    main = sessions["sess-main"]
    digest = build_digest(main, [sessions["agent-abc"]])
    assert "first_prompt: fix the bug in server.py" in digest
    assert "Read(server.py)<1KB" in digest and "Bash(cat big.log)<58KB" in digest
    assert "subagent agent-abc model claude-opus-5" in digest
    assert "xxxx" not in digest and "yyyy" not in digest, "tool output content must never enter the digest"
    assert digest.count("\n") + 1 <= 320 + 10


def test_digest_samples_long_sessions(tree: Path) -> None:
    main = next(s for s in load_sessions(tree) if s.session_id == "sess-main")
    digest = build_digest(main, [], max_lines=10)
    import re

    assert "digest sampled" in digest and len(re.findall(r"\nt\d+ ", digest)) <= 11


def test_narrator_openai_format_parses_and_caches(tree: Path, tmp_path: Path) -> None:
    sessions = {s.session_id: s for s in load_sessions(tree)}
    calls: list[dict] = []
    cfg = LLMConfig(api_key="k", model="openai/gpt-oss-20b")
    narrator = SessionNarrator(cfg, cache_dir=tmp_path / "cache", transport=_fake_transport(calls))
    first = narrator.explain(sessions["sess-main"], [sessions["agent-abc"]])
    assert first.cut_point == 12 and first.do[0].startswith("Do open")
    assert calls[0]["url"].endswith("/chat/completions") and calls[0]["headers"]["Authorization"] == "Bearer k"
    assert calls[0]["body"]["messages"][0]["role"] == "system"
    second = narrator.explain(sessions["sess-main"], [sessions["agent-abc"]])
    assert len(calls) == 1, "second call must come from the cache"
    assert second.narrative == first.narrative
    narrator.explain(sessions["sess-main"], [], refresh=True)
    assert len(calls) == 2
    text = render_explanation(first)
    assert "== don't ==" in text and "better prompt" in text


def test_narrator_anthropic_format(tree: Path, tmp_path: Path) -> None:
    main = next(s for s in load_sessions(tree) if s.session_id == "sess-main")
    calls: list[dict] = []
    cfg = LLMConfig(api_key="k", base_url="https://foundry.example/anthropic", model="claude-haiku-4-5", wire_format="anthropic")
    out = SessionNarrator(cfg, cache_dir=tmp_path, transport=_fake_transport(calls)).explain(main, [])
    assert calls[0]["url"] == "https://foundry.example/anthropic/messages"
    assert calls[0]["headers"]["x-api-key"] == "k" and calls[0]["body"]["system"]
    assert out.narrative.startswith("The session")


def test_bad_model_output_raises(tree: Path, tmp_path: Path) -> None:
    main = next(s for s in load_sessions(tree) if s.session_id == "sess-main")
    bad = lambda url, headers, body, timeout: {"choices": [{"message": {"content": "sorry, no"}}]}  # noqa: E731
    with pytest.raises(AnalysisError):
        SessionNarrator(LLMConfig(api_key="k"), cache_dir=tmp_path, transport=bad).explain(main, [])


def test_config_from_env_prefers_burnlens_then_openrouter() -> None:
    assert LLMConfig.from_env({}) is None
    cfg = LLMConfig.from_env({"OPENROUTER_API_KEY": "or"})
    assert cfg and cfg.api_key == "or" and cfg.model == "openai/gpt-oss-20b" and cfg.wire_format == "openai"
    cfg = LLMConfig.from_env({"BURNLENS_LLM_API_KEY": "f", "BURNLENS_LLM_BASE_URL": "https://x/v1/", "BURNLENS_LLM_MODEL": "m", "BURNLENS_LLM_FORMAT": "Anthropic"})
    assert cfg and cfg.base_url == "https://x/v1" and cfg.wire_format == "anthropic"
