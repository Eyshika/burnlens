from pathlib import Path
import sys

import pytest

from burnlens.capture import CaptureConfig, CaptureError, CaptureRunner


def test_failure_preserves_status_and_complete_diagnostics(tmp_path: Path) -> None:
    result = CaptureRunner(CaptureConfig(tmp_path)).run(
        [sys.executable, "-c", "import sys; print('diagnostic', file=sys.stderr); sys.exit(7)"]
    )
    assert result.exit_code == 7
    assert Path(result.output_path).read_text() == "diagnostic\n"
    assert result.preview == "diagnostic\n"
    assert result.duration_seconds >= 0


@pytest.mark.parametrize("payload", ["b'x' * 1000000", "b'line\\n' * 200000", "b'\\xff' * 1000000"])
def test_large_output_has_bounded_preview_and_complete_artifact(tmp_path: Path, payload: str) -> None:
    result = CaptureRunner(CaptureConfig(tmp_path, preview_lines=3, max_preview_bytes=100)).run(
        [sys.executable, "-c", f"import sys; sys.stdout.buffer.write({payload})"]
    )
    assert result.exit_code == 0
    assert result.output_bytes == 1_000_000
    assert Path(result.output_path).stat().st_size == result.output_bytes
    assert len(result.preview.splitlines()) <= 3
    assert result.displayed_bytes == len(result.preview.encode("utf-8")) <= 100


def test_argv_is_literal_and_does_not_invoke_shell(tmp_path: Path) -> None:
    marker = tmp_path / "should-not-exist"
    literal = f"$(touch {marker}); echo unsafe"
    result = CaptureRunner(CaptureConfig(tmp_path)).run(
        [sys.executable, "-c", "import sys; print(sys.argv[1])", literal]
    )
    assert result.preview == literal + "\n"
    assert not marker.exists()


def test_missing_executable_logs_and_raises(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(CaptureError, match="Could not capture"):
        CaptureRunner(CaptureConfig(tmp_path)).run([str(tmp_path / "missing-executable")])
    assert "Command capture failed" in caplog.text


def test_baseline_keeps_bounded_preview_and_full_artifact(tmp_path: Path) -> None:
    result = CaptureRunner(CaptureConfig(tmp_path, max_preview_bytes=10)).run(
        [sys.executable, "-c", "print('x' * 1000)"], baseline=True
    )
    assert result.baseline
    assert result.output_bytes == 1001
    assert result.displayed_bytes == 10
    assert len(result.preview.encode("utf-8")) == 10


def test_invalid_inputs_fail_before_execution(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        CaptureRunner(CaptureConfig(tmp_path, preview_lines=0))
    with pytest.raises(ValueError, match="non-empty"):
        CaptureRunner(CaptureConfig(tmp_path)).run([])
