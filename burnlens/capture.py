"""Explicit command capture with complete diagnostics and a bounded preview."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import subprocess
import time
import uuid

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaptureConfig:
    output_root: Path
    preview_lines: int = 200
    max_preview_bytes: int = 16_000


@dataclass(frozen=True)
class CaptureResult:
    run_id: str
    exit_code: int
    duration_seconds: float
    output_bytes: int
    displayed_bytes: int
    output_path: str
    preview: str
    baseline: bool


class CaptureError(RuntimeError):
    """A command could not be started or its output could not be captured."""


class CaptureRunner:
    def __init__(self, cfg: CaptureConfig) -> None:
        if cfg.preview_lines <= 0 or cfg.max_preview_bytes <= 0:
            raise ValueError("Preview line and byte limits must be positive")
        self._cfg = cfg

    def run(self, command: list[str], *, baseline: bool = False) -> CaptureResult:
        """Capture argv without a shell; callers stream the artifact for baseline runs.

        Preview and displayed_bytes always describe the bounded preview. A caller
        displaying the full baseline artifact must account for its bytes itself.
        """
        if not command or not all(isinstance(arg, str) for arg in command):
            raise ValueError("Command must be a non-empty list of string arguments")
        run_id = uuid.uuid4().hex
        output_path = self._cfg.output_root.resolve() / f"{run_id}.log"
        started = time.monotonic()
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("xb") as output:
                completed = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
            duration_seconds = time.monotonic() - started
            output_bytes = output_path.stat().st_size
            with output_path.open("rb") as output:
                raw_preview = output.read(self._cfg.max_preview_bytes)
            preview = raw_preview.decode("utf-8", errors="replace")
            preview = "".join(preview.splitlines(keepends=True)[: self._cfg.preview_lines])
            # Replacement characters can expand invalid input beyond the byte limit.
            preview = preview.encode("utf-8")[: self._cfg.max_preview_bytes].decode(
                "utf-8", errors="ignore"
            )
        except (OSError, ValueError) as exc:
            logger.error("Command capture failed: executable=%r output=%s", command[0], output_path)
            raise CaptureError(f"Could not capture command {command[0]!r}: {exc}") from exc
        return CaptureResult(
            run_id=run_id,
            exit_code=completed.returncode,
            duration_seconds=duration_seconds,
            output_bytes=output_bytes,
            displayed_bytes=len(preview.encode("utf-8")),
            output_path=str(output_path),
            preview=preview,
            baseline=baseline,
        )
