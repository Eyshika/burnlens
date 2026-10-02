"""The prevented-spend ledger: one auditable row per load that was stopped before it ran.

This is the only place a savings number is allowed to come from. A row is written only when the
size of the thing that did not enter context was known at the moment it was stopped, so the total
is an observation rather than an estimate. Everything else stays `unmeasured`.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

STATE_DIR = Path(os.environ.get("BURNLENS_STATE_DIR", Path.home() / ".burnlens"))
LEDGER_PATH = STATE_DIR / "prevented.jsonl"
BYTES_PER_TOKEN = 4  # recorded on every row so a later correction can be applied to old rows


class LedgerError(RuntimeError):
    """Raised when the ledger cannot be read; never raised on a failed append."""


@dataclass(frozen=True)
class PreventedLoad:
    """One load that was stopped, and the measurement that makes it countable."""

    at: datetime
    rule: str
    target: str
    bytes_prevented: int
    divisor: int
    session_id: str
    repo: str
    decision: str
    basis: str

    @property
    def tokens_prevented(self) -> int:
        return self.bytes_prevented // self.divisor if self.divisor else 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "at": self.at.isoformat(),
            "rule": self.rule,
            "target": self.target,
            "bytes_prevented": self.bytes_prevented,
            "divisor": self.divisor,
            "tokens_prevented": self.tokens_prevented,
            "session_id": self.session_id,
            "repo": self.repo,
            "decision": self.decision,
            "basis": self.basis,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PreventedLoad":
        return cls(
            at=_parse_ts(raw.get("at")),
            rule=str(raw.get("rule") or ""),
            target=str(raw.get("target") or ""),
            bytes_prevented=int(raw.get("bytes_prevented") or 0),
            divisor=int(raw.get("divisor") or BYTES_PER_TOKEN),
            session_id=str(raw.get("session_id") or ""),
            repo=str(raw.get("repo") or ""),
            decision=str(raw.get("decision") or ""),
            basis=str(raw.get("basis") or ""),
        )


class Ledger:
    """Append-only JSONL. Appends never raise: a lost row must not break the user's tool call."""

    def __init__(self, path: Path = LEDGER_PATH) -> None:
        self.path = path

    def append(self, row: PreventedLoad) -> bool:
        if row.bytes_prevented <= 0:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row.as_dict(), sort_keys=True) + "\n")
        except OSError as exc:
            logger.warning("could not append to ledger %s: %s", self.path, exc)
            return False
        return True

    def rows(self, since: datetime | None = None) -> list[PreventedLoad]:
        return [r for r in self._iter_rows() if since is None or r.at >= since]

    def _iter_rows(self) -> Iterator[PreventedLoad]:
        if not self.path.is_file():
            return
        try:
            text = self.path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.error("cannot read ledger %s: %s", self.path, exc)
            raise LedgerError(f"cannot read ledger {self.path}: {exc}") from exc
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield PreventedLoad.from_dict(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue  # a torn line from a concurrent append is skipped, never fatal

    def summary(self, since: datetime | None = None) -> dict[str, Any]:
        rows = self.rows(since)
        by_rule: dict[str, dict[str, int]] = {}
        for row in rows:
            bucket = by_rule.setdefault(row.rule, {"events": 0, "tokens": 0, "bytes": 0})
            bucket["events"] += 1
            bucket["tokens"] += row.tokens_prevented
            bucket["bytes"] += row.bytes_prevented
        return {
            "events": len(rows),
            "tokens_prevented": sum(r.tokens_prevented for r in rows),
            "bytes_prevented": sum(r.bytes_prevented for r in rows),
            "by_rule": dict(sorted(by_rule.items(), key=lambda kv: -kv[1]["tokens"])),
            "evidence_class": "measured",
            "basis": "each row is a load that was stopped and whose size was known at that moment; "
                     "token counts are bytes divided by the divisor recorded on the row",
        }

    def rewrite(self, rows: list[PreventedLoad]) -> None:
        """Replace the file atomically. Only for pruning; the ledger is otherwise append-only."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".jsonl")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row.as_dict(), sort_keys=True) + "\n")
            os.replace(tmp, self.path)
        except OSError:
            Path(tmp).unlink(missing_ok=True)
            raise


def _parse_ts(raw: Any) -> datetime:
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)
