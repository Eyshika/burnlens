"""Resolve which repository a session ran in, from the ``cwd`` the transcript records.

The repo is the unit every repo-level finding attaches to: CLAUDE.md, .mcp.json, agent
definitions and workflow files all live there, and a fix to any of them lands once for everyone.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

UNKNOWN = "unknown"
MAX_WALK_UP = 24  # a repo root deeper than this is a symlink loop, not a checkout
REMOTE_URL = re.compile(r"^\s*url\s*=\s*(\S+)\s*$", re.M)
SCP_SYNTAX = re.compile(r"^[\w.-]+@([\w.-]+):(.+)$")

_cache: dict[str, str] = {}


def resolve(cwd: str) -> str:
    """A stable key for the repo containing ``cwd``: ``host/org/name`` where a remote is readable.

    Degrades rather than fails: the git root's name when there is no remote, the directory's own
    name when there is no checkout, and ``unknown`` when there is no usable path. A session
    recorded on another machine will not resolve here, which is why the raw cwd is kept too.
    """
    if not cwd:
        return UNKNOWN
    if cwd in _cache:
        return _cache[cwd]
    _cache[cwd] = key = _resolve(cwd)
    return key


def _resolve(cwd: str) -> str:
    try:
        path = Path(cwd).expanduser()
    except (OSError, ValueError):
        return UNKNOWN
    root = _git_root(path)
    if root is None:
        return path.name or UNKNOWN
    remote = _remote_url(root / ".git" / "config")
    return normalise_remote(remote) if remote else root.name


def _git_root(start: Path) -> Path | None:
    current = start
    for _ in range(MAX_WALK_UP):
        try:
            if (current / ".git").exists():
                return current
        except OSError:
            return None
        if current.parent == current:
            break
        current = current.parent
    return None


def _remote_url(config: Path) -> str:
    try:
        text = config.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    match = REMOTE_URL.search(text)
    return match.group(1) if match else ""


def normalise_remote(url: str) -> str:
    """github.com/<owner>/<repo>, from either the ssh or the https form."""
    cleaned = url.strip().removesuffix(".git")
    scp = SCP_SYNTAX.match(cleaned)
    if scp:
        return f"{scp.group(1)}/{scp.group(2).lstrip('/')}"
    without_scheme = re.sub(r"^\w+://", "", cleaned)
    without_creds = without_scheme.split("@", 1)[-1]
    return without_creds or UNKNOWN
