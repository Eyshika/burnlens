"""Handoff brief: everything a fresh session needs, so "start a new session" is one click.

Built from the transcript alone, no model needed: the goal, the latest ask, the files
the agent edited and read most, the commands it leaned on, its last statements, and
a ready-to-paste opening prompt that scopes the work and asks for limited output.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .model import Session

MAX_FILES = 8
MAX_COMMANDS = 5
MAX_STATEMENTS = 5
STATEMENT_CHARS = 160
EDIT_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})


def build_handoff(session: Session, children: list[Session]) -> str:
    """Markdown brief for the next session."""
    edited: Counter[str] = Counter()
    read: Counter[str] = Counter()
    commands: Counter[str] = Counter()
    for turn in session.turns:
        for call in turn.tool_calls:
            if call.name in EDIT_TOOLS and call.file_path:
                edited[call.file_path] += 1
            elif call.name == "Read" and call.file_path:
                read[call.file_path] += 1
            elif call.name == "Bash" and call.command_head:
                commands[call.command_head] += 1
    statements = [t.text_preview for t in reversed(session.turns) if t.text_preview and not t.tool_calls][:MAX_STATEMENTS]
    statements.reverse()
    hot_files = [p for p, _ in (edited + read).most_common(MAX_FILES)]

    lines = [
        f"# Handoff from session {session.session_id[:8]} ({session.project})",
        "",
        f"**Goal:** {session.first_prompt or '(no prompt recorded)'}",
        f"**Latest ask:** {session.last_prompt or session.first_prompt or '(none)'}",
        f"**Where it stopped:** turn {len(session.turns)}, context {session.peak_context:,} tokens at peak, {len(children)} subagents.",
        "",
        "## Files in play",
    ]
    lines.extend(f"- `{_short(p)}` (edited {edited[p]}x, read {read[p]}x)" for p in hot_files) or lines.append("- (none)")
    lines += ["", "## Commands it relied on"]
    lines.extend(f"- `{c[:90]}` ({n}x)" for c, n in commands.most_common(MAX_COMMANDS)) or lines.append("- (none)")
    lines += ["", "## Last things it said"]
    lines.extend(f"- {t[:STATEMENT_CHARS]}" for t in statements) or lines.append("- (nothing recorded)")
    lines += ["", "## Paste this to start the new session", "", "```", _opening_prompt(session, hot_files), "```"]
    return "\n".join(lines)


def _opening_prompt(session: Session, hot_files: list[str]) -> str:
    files = ", ".join(_short(p) for p in hot_files[:4]) or "the files I paste below"
    ask = session.last_prompt or session.first_prompt or "continue the task"
    return (
        f"Continuing from a previous session. Goal: {session.first_prompt or ask}\n"
        f"Current step: {ask}\n"
        f"Relevant files: {files}. Read only the functions you need (use line ranges), not whole files.\n"
        "Keep shell output short (tail/grep). Use Sonnet for any subagents and do not nest them.\n"
        "Ask me before reading anything over 50 KB; I can paste the exact block."
    )


def _short(path: str) -> str:
    home = str(Path.home())
    return path.replace(home, "~", 1) if path.startswith(home) else path
