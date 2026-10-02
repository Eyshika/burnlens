"""Typed records parsed out of agent transcripts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"})
MUTATION_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})
# build and test runners whose logs are long and mostly irrelevant; the list is SoL-Pi's
DIAGNOSTIC_COMMAND = re.compile(
    r"(?:^|[;&|()\s])(?:lake\s+build|lake\s+env\s+lean|lean|coq|cargo(?:\s+(?:build|test|check))?|zig\s+build"
    r"|pytest|python3?\s+-m\s+(?:pytest|unittest|py_compile)|ctest|cmake\s+--build|ninja|make|npm\s+test"
    r"|pnpm\s+test|yarn\s+test|go\s+test|bazel\s+test|tsc|mvn\s+test|gradle(?:w)?\s+test)(?:\s|$)",
    re.I,
)


@dataclass(frozen=True)
class Usage:
    """Token counts for one API response."""

    input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0

    @property
    def context(self) -> int:
        """Everything the model had to read for this turn."""
        return self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens

    @property
    def total(self) -> int:
        return self.context + self.output_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.cache_creation_input_tokens + other.cache_creation_input_tokens,
            self.cache_read_input_tokens + other.cache_read_input_tokens,
            self.output_tokens + other.output_tokens,
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Usage":
        return cls(
            input_tokens=int(raw.get("input_tokens") or 0),
            cache_creation_input_tokens=int(raw.get("cache_creation_input_tokens") or 0),
            cache_read_input_tokens=int(raw.get("cache_read_input_tokens") or 0),
            output_tokens=int(raw.get("output_tokens") or 0),
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "output_tokens": self.output_tokens,
            "context": self.context,
            "total": self.total,
        }


@dataclass
class ToolCall:
    """One tool invocation and the size of what it pushed back into context."""

    tool_use_id: str
    name: str
    input: dict[str, Any] = field(default_factory=dict)
    result_bytes: int = 0
    result_has_image: bool = False

    @property
    def file_path(self) -> str | None:
        value = self.input.get("file_path")
        return str(value) if value else None

    @property
    def command_head(self) -> str | None:
        """First line of a shell command, whitespace-collapsed, for grouping."""
        command = self.input.get("command")
        if not command:
            return None
        return " ".join(str(command).strip().splitlines()[0].split())


@dataclass
class Turn:
    """One assistant message with its usage and the tools it called."""

    message_id: str
    timestamp: datetime
    model: str
    usage: Usage
    tool_calls: list[ToolCall] = field(default_factory=list)
    text_preview: str = ""


@dataclass
class Session:
    """One transcript file: a main session or a subagent run."""

    session_id: str
    project: str
    path: Path
    parent_session_id: str | None
    first_prompt: str
    last_prompt: str = ""
    turns: list[Turn] = field(default_factory=list)
    cwd: str = ""  # directory the session ran in, as the transcript recorded it
    git_branch: str = ""
    repo: str = ""  # resolved from cwd; the unit repo-level findings attach to
    agent: str = "claude-code"  # which application produced this session
    user: str = ""  # who ran it; empty on a single-user machine
    workflow: str = ""  # scheduled job / CI workflow name; empty for interactive work
    run_kind: str = "interactive"  # interactive | scheduled | ci
    prompts: list[str] = field(default_factory=list)  # every user prompt, truncated; the teacher reads these
    prompt_chars: list[int] = field(default_factory=list)  # original length of each prompt (pasted content shows here)

    @property
    def is_subagent(self) -> bool:
        return self.parent_session_id is not None

    @property
    def usage(self) -> Usage:
        total = Usage()
        for turn in self.turns:
            total = total + turn.usage
        return total

    @property
    def peak_context(self) -> int:
        return max((t.usage.context for t in self.turns), default=0)

    @property
    def start(self) -> datetime | None:
        return self.turns[0].timestamp if self.turns else None

    @property
    def end(self) -> datetime | None:
        return self.turns[-1].timestamp if self.turns else None
