"""Native source selection and discovery do not double-load the primary root."""
from pathlib import Path

import pytest

from burnlens.cli import _extras, build_parser, main


def test_discovery_deduplicates_explicit_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('burnlens.cli.discover_sources', lambda: [('codex', tmp_path)])
    args = build_parser().parse_args(['--source', 'codex', '--root', str(tmp_path),
                                     '--discover-agents', '--codex-root', str(tmp_path)])
    assert _extras(args) == []


def test_native_source_uses_its_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = []
    monkeypatch.setattr('burnlens.cli.default_root', lambda source: tmp_path / source)
    monkeypatch.setattr('burnlens.cli.load_all', lambda source, root, extras, since: captured.append((source, root)) or [])
    assert main(['--source', 'codex']) == 1
    assert captured == [('codex', tmp_path / 'codex')]


def test_discovery_works_without_claude_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = []
    monkeypatch.setattr('burnlens.cli.discover_sources', lambda: [('gemini-cli', tmp_path)])
    monkeypatch.setattr('burnlens.cli.load_all', lambda source, root, extras, since: captured.append((source, root, extras)) or [])
    assert main(['--discover-agents']) == 1
    assert captured == [('gemini-cli', tmp_path, [])]


def test_ui_demo_serves_synthetic_sessions_and_cleans_up(monkeypatch: pytest.MonkeyPatch) -> None:
    from burnlens.adapters import load_all

    seen: dict[str, object] = {}

    def fake_serve(root: Path, settings: object, **kw: object) -> None:
        extras = kw["extras"]
        sessions = load_all("claude-code", root, extras)
        seen.update(root=root, apps={s.agent for s in sessions}, sessions=len(sessions))

    monkeypatch.setattr('burnlens.cli.serve', fake_serve)
    assert main(['ui', '--demo', '--no-browser']) == 0
    assert seen['sessions'] > 20 and {'claude-code', 'codex', 'support-bot'} <= seen['apps']
    assert not Path(seen['root']).exists()
