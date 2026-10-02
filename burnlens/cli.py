"""burnlens command line: report | sessions | session <id>."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import __version__, demo
from .adapters import SOURCES, default_root, discover_sources, load_all
from .aggregate import aggregate
from .analyze import AnalysisError, LLMConfig, SessionNarrator, render_explanation
from .coach import coach_prompt, habits
from .config import ConfigError, PriceTable, Settings, load_settings
from .ledger import Ledger, LedgerError
from .capture import CaptureConfig, CaptureError, CaptureRunner
from .interventions import InterventionConfig, InterventionError, InterventionService, RunMeasurement, default_state_dir
from .findings import detect
from .report import render_json, render_report, render_session_detail, render_sessions
from .teacher import lessons
from .handoff import build_handoff
from .hook import DEFAULT_SETTINGS, install_hooks, run_hook
from .server import DEFAULT_HOST, DEFAULT_PORT, serve
from .transcripts import TranscriptError

logger = logging.getLogger(__name__)

DEFAULT_TOP = 10


def _global_options(parser: argparse.ArgumentParser, suppress: bool) -> None:
    """Options accepted both before and after the subcommand.

    The copy attached to each subcommand uses SUPPRESS defaults so that an option
    left unset after the subcommand does not overwrite one given before it.
    """
    d = (lambda v: argparse.SUPPRESS) if suppress else (lambda v: v)
    parser.add_argument("--root", type=Path, default=d(None), help="transcript root (defaults to the selected native agent directory)")
    parser.add_argument("--source", choices=SOURCES, default=d("claude-code"), help="what --root contains (default claude-code)")
    parser.add_argument("--codex-root", type=Path, action="append", default=d(None), help="extra Codex session directories (repeatable)")
    parser.add_argument("--gemini-root", type=Path, action="append", default=d(None), help="extra Gemini CLI chat directories (repeatable)")
    parser.add_argument("--discover-agents", action="store_true", default=d(False), help="include installed native agents with local session directories")
    parser.add_argument("--generic-root", type=Path, action="append", default=d(None), help="extra generic-JSONL trace dirs to merge into the same view (repeatable)")
    parser.add_argument("--litellm-root", type=Path, action="append", default=d(None), help="LiteLLM logs to merge: StandardLoggingPayload files (S3/GCS/callback) or /spend/logs JSON (repeatable)")
    parser.add_argument("--langfuse-root", type=Path, action="append", default=d(None), help="Langfuse traces + observations exports or API pages to merge (repeatable)")
    parser.add_argument("--config", type=Path, default=d(None), help="burnlens.toml with thresholds, disabled rules, premium models, task tiers")
    parser.add_argument("--days", type=int, default=d(None), help="only turns from the last N days")
    parser.add_argument("--since", type=str, default=d(None), help="only turns at or after this date (YYYY-MM-DD, UTC)")
    parser.add_argument("--top", type=int, default=d(DEFAULT_TOP), help="rows per table")
    parser.add_argument("--json", action="store_true", default=d(False), help="emit JSON instead of text")
    parser.add_argument("--context-threshold", type=int, default=d(None), help="tokens of context that count as bloated (overrides config)")
    parser.add_argument("--large-payload", type=int, default=d(None), help="tool result bytes that count as large (overrides config)")
    parser.add_argument("-v", "--verbose", action="store_true", default=d(False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="burnlens", description="Review AI coding usage and evaluate improvements across agents.")
    parser.add_argument("--version", action="version", version=f"burnlens {__version__}")
    _global_options(parser, suppress=False)

    sub = parser.add_subparsers(dest="command")
    subs = {
        "report": sub.add_parser("report", help="full report with findings (default)"),
        "sessions": sub.add_parser("sessions", help="rank sessions by tokens"),
        "session": sub.add_parser("session", help="deep dive into one session"),
        "ui": sub.add_parser("ui", help="open the dashboard app in your browser"),
        "hook": sub.add_parser("hook", help="Claude Code PreToolUse hook (reads the call as JSON on stdin)"),
        "install-hooks": sub.add_parser("install-hooks", help="install the pre-execution guard into Claude Code settings"),
        "explain": sub.add_parser("explain", help="ask a model why one session burned tokens (needs BURNLENS_LLM_API_KEY or OPENROUTER_API_KEY)"),
    }
    subs["prevented"] = sub.add_parser("prevented", help="the prevented-spend ledger: what was stopped, and what it would have cost")
    subs["prevented"].add_argument("--rows", action="store_true", help="print every row instead of the totals")
    subs["coach"] = sub.add_parser("coach", help="coach a prompt before you send it: task type, candidate model to evaluate, waste patterns, rewrite")
    subs["coach"].add_argument("prompt", help="the prompt you are about to send (quote it)")
    subs["coach"].add_argument("--model", default="", help="model the session runs on, if known")
    subs["coach"].add_argument("--context", type=int, default=0, help="current context tokens of the session")
    subs["habits"] = sub.add_parser("habits", help="recurring waste as lessons: this week vs last, avoidable tokens, what to say")
    subs["handoff"] = sub.add_parser("handoff", help="write the brief for a fresh session: goal, files in play, opening prompt")
    subs["handoff"].add_argument("session_id", help="session id or unique prefix")
    subs["explain"].add_argument("session_id", help="session id or unique prefix")
    subs["explain"].add_argument("--refresh", action="store_true", help="ignore the cached explanation")
    subs["session"].add_argument("session_id", help="session id or unique prefix")
    subs["ui"].add_argument("--port", type=int, default=DEFAULT_PORT)
    subs["ui"].add_argument("--host", default=DEFAULT_HOST)
    subs["ui"].add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    subs["ui"].add_argument("--no-notify", action="store_true", help="disable desktop notifications for alerts")
    subs["ui"].add_argument("--demo", action="store_true", help="open the dashboard on synthetic sessions instead of your own")
    subs["install-hooks"].add_argument("--settings", type=Path, default=DEFAULT_SETTINGS, help="Claude Code settings.json to edit")
    subs["install-hooks"].add_argument("--remove", action="store_true", help="remove the hook instead")
    subs["install-hooks"].add_argument("--strict", action="store_true", help="ask you before every automatic subagent spawn, even in the green zone")
    subs["hook"].add_argument("--strict", action="store_true", help="ask on every automatic subagent spawn")
    subs["hook"].add_argument("--no-autofix", action="store_true", help="compatibility flag; calls are never automatically rewritten")
    subs["install-hooks"].add_argument("--no-autofix", action="store_true", help="compatibility flag; calls are never automatically rewritten")
    subs["lessons"] = sub.add_parser("lessons", help="the teacher: which platform features (CLAUDE.md, skills, saved research) your prompts say you are missing")
    subs["capture"] = sub.add_parser("capture", help="explicitly run a command with full diagnostics and bounded output")
    subs["capture"].add_argument("--action", required=True, help="reviewed intervention ID from the dashboard")
    subs["capture"].add_argument("--task", required=True, help="same task label for comparable baseline/trial runs")
    subs["capture"].add_argument("--baseline", action="store_true", help="display full output to record a baseline before applying")
    subs["capture"].add_argument("argv", nargs=argparse.REMAINDER, help="command after --; no implicit shell")
    for sp in subs.values():
        _global_options(sp, suppress=True)
    return parser


def _capture_command(args: argparse.Namespace) -> int:
    command = args.argv[1:] if args.argv and args.argv[0] == "--" else args.argv
    try:
        service = InterventionService(InterventionConfig(default_state_dir()))
        action = service.get_action(args.action)
        if not args.baseline and action.state != "applied":
            raise InterventionError("Review and apply this intervention before a trial run.")
        if not command or not args.task.strip() or len(args.task) > 4096:
            raise InterventionError("Provide a task label and a command after --.")
        result = CaptureRunner(CaptureConfig(default_state_dir() / "captures")).run(command, baseline=args.baseline)
    except (CaptureError, InterventionError, ValueError, sqlite3.Error, OSError) as exc:
        logger.error("Capture could not start: %s", exc)
        return 2
    displayed_bytes = result.displayed_bytes
    try:
        if args.baseline:
            with Path(result.output_path).open("rb") as output:
                shutil.copyfileobj(output, sys.stdout.buffer)
            displayed_bytes = result.output_bytes
        else:
            sys.stdout.write(result.preview)
        sys.stdout.flush()
    except (BrokenPipeError, OSError):
        logger.error("Output delivery failed; no reduction will be recorded. Full log: %s", result.output_path)
        return result.exit_code if result.exit_code > 0 else 2
    try:
        service.record_run(args.action, RunMeasurement(
            run_id=result.run_id, task_label=args.task, baseline=args.baseline,
            exit_code=result.exit_code, duration_seconds=result.duration_seconds,
            output_bytes=result.output_bytes, displayed_bytes=displayed_bytes, output_path=result.output_path,
            policy_revision=action.revision,
        ))
    except (InterventionError, sqlite3.Error, OSError) as exc:
        logger.error("Command finished but measurement was not saved: %s. Full log: %s", exc, result.output_path)
        return result.exit_code if result.exit_code >= 0 else 128 - result.exit_code
    print(f"\nBurnlens run {result.run_id}: {displayed_bytes:,}/{result.output_bytes:,} output bytes displayed. "
          f"Full log: {result.output_path}. Record the task outcome in the dashboard. "
          "Token and dollar savings are unmeasured.", file=sys.stderr)
    return result.exit_code if result.exit_code >= 0 else 128 - result.exit_code


def _serve_ui(args: argparse.Namespace, settings: Settings) -> int:
    try:
        serve(args.root, settings, host=args.host, port=args.port, open_browser=not args.no_browser, notify=not args.no_notify, source=args.source, extras=_extras(args))
    except TranscriptError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    since = _resolve_since(args.days, args.since)
    try:
        settings = load_settings(args.config, overrides={"context_tokens": args.context_threshold, "large_payload_bytes": args.large_payload})
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    thresholds = settings.thresholds

    if args.command == "capture":
        return _capture_command(args)

    if args.command == "hook":
        print(json.dumps(run_hook(sys.stdin.read(), thresholds, strict=args.strict, autofix=not args.no_autofix, task_tiers=settings.task_tiers, prices=settings.prices)))
        return 0
    if args.command == "install-hooks":
        print(install_hooks(args.settings, remove=args.remove, strict=args.strict, autofix=not args.no_autofix, config_path=settings.source))
        return 0
    if args.command == "prevented":
        return _prevented(args.rows, args.json, since)
    if args.command == "coach":
        c = coach_prompt(args.prompt, args.context, args.model, thresholds, settings.task_tiers, prices=settings.prices)
        if args.json:
            print(json.dumps(c.as_dict(), indent=2))
        else:
            print(f"task: {c.task}   candidate tier (unvalidated): {c.recommended_tier}   zone: {c.zone}")
            for m in c.messages:
                print(f"- {m}")
            if c.rewrite:
                print(f"\ntry instead:\n  {c.rewrite}")
            if not c.messages:
                print("- looks lean; go ahead")
        return 0
    if args.command == "ui" and args.demo:
        with tempfile.TemporaryDirectory(prefix="burnlens-demo-") as tmp:
            args.source = "claude-code"
            args.root, args.generic_root, args.litellm_root = demo.generate(Path(tmp))
            return _serve_ui(args, settings)
    try:
        if args.discover_agents and args.root is None:
            discovered = discover_sources()
            if discovered:
                args.source, args.root = discovered[0]
        args.root = args.root or default_root(args.source)
    except TranscriptError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.command == "ui":
        return _serve_ui(args, settings)

    try:
        sessions = load_all(args.source, args.root, _extras(args), since=since)
    except TranscriptError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not sessions:
        print("no transcripts with turns in the selected window", file=sys.stderr)
        return 1

    command = args.command or "report"
    if command == "session":
        return _session_detail(sessions, args.session_id, args.top, settings.prices)
    if command == "explain":
        return _explain(sessions, args.session_id, args.refresh, args.json)
    if command == "handoff":
        matches = [s for s in sessions if not s.is_subagent and s.session_id.startswith(args.session_id)]
        if len(matches) != 1:
            print(f"error: {len(matches)} sessions match '{args.session_id}'; give a longer prefix", file=sys.stderr)
            return 2
        print(build_handoff(matches[0], [s for s in sessions if s.parent_session_id == matches[0].session_id]))
        return 0

    agg = aggregate(sessions, context_threshold=thresholds.context_tokens, large_payload_bytes=thresholds.large_payload_bytes, premium_markers=thresholds.premium_markers)
    if command == "lessons":
        rows = lessons(sessions)
        if args.json:
            print(json.dumps([l.as_dict() for l in rows], indent=2))
            return 0
        if not rows:
            print("no repeated patterns yet; the teacher needs a few sessions")
        for l in rows:
            print(f"[{l.feature:<14}] {l.title}  (savings unmeasured, {l.occurrences} sessions)")
            print(f"                 {l.why}")
            for e in l.examples[:3]:
                print(f"                 e.g. {e}")
            print()
            print("\n".join("    " + line for line in l.draft.splitlines()))
            print()
        return 0
    if command == "habits":
        rows = habits(sessions, agg, thresholds)
        if args.json:
            print(json.dumps([h.as_dict() for h in rows], indent=2))
            return 0
        if not rows:
            print("no recurring waste in this window")
        for h in rows:
            print(f"[{h.trend:<6}] {h.title}: {h.this_week} this week, {h.last_week} last week, savings unmeasured")
            print(f"         {h.say_this}")
            if h.examples:
                print(f"         e.g. {'; '.join(h.examples)}")
        return 0
    if command == "sessions":
        print(render_sessions(agg, args.top))
        return 0
    findings = detect(agg, thresholds, settings.disabled_rules)
    print(render_json(agg, findings, settings.prices) if args.json else render_report(agg, findings, args.top, settings.prices))
    return 0


def _prevented(rows: bool, as_json: bool, since) -> int:  # noqa: ANN001 - datetime | None
    """The only counter in this product that is a measurement rather than an estimate."""
    ledger = Ledger()
    try:
        entries = ledger.rows(since)
        summary = ledger.summary(since)
    except LedgerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if as_json:
        print(json.dumps({**summary, "rows": [r.as_dict() for r in entries]}, indent=2))
        return 0
    if not entries:
        print("nothing prevented yet. install the guard with `burnlens install-hooks`.")
        return 0
    print(f"prevented {summary['tokens_prevented']:,} tokens across {summary['events']} stopped calls")
    print(f"evidence: {summary['basis']}")
    print()
    for rule, bucket in summary["by_rule"].items():
        print(f"{rule:<16}{bucket['events']:>5} calls   {bucket['tokens']:>12,} tokens")
    if rows:
        print()
        for entry in entries:
            print(f"{entry.at:%Y-%m-%d %H:%M}  {entry.rule:<12} {entry.tokens_prevented:>9,} tok  {entry.target}")
            print(f"{'':<20}{entry.basis}")
    return 0


def _session_detail(sessions, prefix: str, top: int, prices: PriceTable | None = None) -> int:  # noqa: ANN001 - list[Session]
    matches = [s for s in sessions if not s.is_subagent and s.session_id.startswith(prefix)]
    if len(matches) != 1:
        print(f"error: {len(matches)} sessions match '{prefix}'; give a longer prefix", file=sys.stderr)
        return 2
    session = matches[0]
    children = [s for s in sessions if s.parent_session_id == session.session_id]
    print(render_session_detail(session, children, top, prices))
    return 0


def _explain(sessions, prefix: str, refresh: bool, as_json: bool) -> int:  # noqa: ANN001 - list[Session]
    config = LLMConfig.from_env()
    if config is None:
        print("error: no model configured. Set BURNLENS_LLM_API_KEY (or OPENROUTER_API_KEY); see README.", file=sys.stderr)
        return 2
    matches = [s for s in sessions if not s.is_subagent and s.session_id.startswith(prefix)]
    if len(matches) != 1:
        print(f"error: {len(matches)} sessions match '{prefix}'; give a longer prefix", file=sys.stderr)
        return 2
    children = [s for s in sessions if s.parent_session_id == matches[0].session_id]
    print(f"asking {config.model} at {config.base_url} (digest only, no file contents)...", file=sys.stderr)
    try:
        explanation = SessionNarrator(config).explain(matches[0], children, refresh=refresh)
    except AnalysisError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(explanation.as_dict(), indent=2) if as_json else render_explanation(explanation))
    return 0


def _extras(args: argparse.Namespace) -> list[tuple[str, Path]]:
    extras = [("generic", p) for p in (args.generic_root or [])] + [("litellm", p) for p in (args.litellm_root or [])] + [("langfuse", p) for p in (args.langfuse_root or [])]
    extras += [("codex", p) for p in (args.codex_root or [])] + [("gemini-cli", p) for p in (args.gemini_root or [])]
    if args.discover_agents:
        extras += discover_sources()
    seen = {(args.source, args.root.resolve())}
    unique = []
    for source, path in extras:
        key = (source, path.resolve())
        if key not in seen:
            unique.append((source, path))
            seen.add(key)
    extras = unique
    missing = [f"{src}: {path}" for src, path in extras if not path.exists()]
    if missing:
        raise SystemExit("error: source path does not exist: " + "; ".join(missing))
    return extras


def _resolve_since(days: int | None, since: str | None) -> datetime | None:
    if days is not None and since is not None:
        raise SystemExit("error: pass --days or --since, not both")
    if days is not None:
        return datetime.now(timezone.utc) - timedelta(days=days)
    if since is not None:
        return datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
