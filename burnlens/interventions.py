"""Persistent, opt-in interventions and descriptive run measurements."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sqlite3
import shlex
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

from .findings import Thresholds
from .model import Session

logger = logging.getLogger(__name__)
MAX_TEXT_LENGTH = 4096
OUTCOMES = frozenset({'success', 'rework', 'failed', 'unknown'})


class InterventionError(RuntimeError):
    """An intervention operation cannot be completed."""


def default_state_dir() -> Path:
    return Path(os.environ.get('BURNLENS_STATE_DIR', str(Path.home() / '.burnlens')))


@dataclass(frozen=True)
class InterventionConfig:
    state_dir: Path = field(default_factory=default_state_dir)


@dataclass(frozen=True)
class RunMeasurement:
    run_id: str
    task_label: str
    baseline: bool
    exit_code: int
    duration_seconds: float
    output_bytes: int
    displayed_bytes: int
    output_path: str
    outcome: str = 'unknown'
    notes: str = ''
    policy_revision: str = ''


@dataclass
class Action:
    id: str
    revision: str
    project: str
    agent: str
    state: str
    policy_text: str
    observed_count: int
    observed_bytes: int
    transcript_dirs: list[str]
    runs: list[RunMeasurement] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        trials = [run for run in self.runs if not run.baseline]
        comparisons = []
        for label in sorted({run.task_label for run in self.runs}):
            baseline = [run for run in self.runs if run.task_label == label and run.baseline]
            trial = [run for run in trials if run.task_label == label]
            comparisons.append({
                'task_label': label, 'baseline_runs': len(baseline), 'trial_runs': len(trial),
                'baseline_mean_output_bytes': sum(r.output_bytes for r in baseline) / len(baseline) if baseline else None,
                'trial_mean_output_bytes': sum(r.output_bytes for r in trial) / len(trial) if trial else None,
                'trial_mean_displayed_bytes': sum(r.displayed_bytes for r in trial) / len(trial) if trial else None,
            })
        result['measurement'] = {
            'trial_runs': len(trials), 'baseline_runs': len(self.runs) - len(trials),
            'output_bytes_reduced': sum(max(0, r.output_bytes - r.displayed_bytes) for r in trials),
            'task_comparisons': comparisons, 'savings': 'unmeasured',
            'quality': 'User-reported outcomes; not independently verified.' if any(r.outcome != 'unknown' for r in self.runs) else 'Quality unmeasured; record task outcomes.',
        }
        return result


class InterventionService:
    def __init__(self, config: InterventionConfig) -> None:
        self._config = config
        self._database = config.state_dir / 'interventions.sqlite3'

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self._config.state_dir.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._database, timeout=30)
        try:
            connection.execute('BEGIN IMMEDIATE')
            connection.execute('CREATE TABLE IF NOT EXISTS actions (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            logger.exception('intervention operation failed in %s', self._database)
            raise
        finally:
            connection.close()

    def _fail(self, message: str) -> None:
        logger.error('%s', message)
        raise InterventionError(message)

    def _read(self, connection: sqlite3.Connection, action_id: str) -> Action:
        row = connection.execute('SELECT payload FROM actions WHERE id = ?', (action_id,)).fetchone()
        if row is None:
            self._fail('Unknown intervention ID')
        payload = json.loads(row[0])
        payload['runs'] = [RunMeasurement(**run) for run in payload['runs']]
        return Action(**payload)

    def _save(self, connection: sqlite3.Connection, action: Action) -> None:
        connection.execute('INSERT OR REPLACE INTO actions VALUES (?, ?)', (action.id, json.dumps(asdict(action))))

    def refresh(self, sessions: list[Session], thresholds: Thresholds) -> list[Action]:
        grouped: dict[tuple[str, str], list[Session]] = {}
        for session in sessions:
            grouped.setdefault((session.project, session.agent), []).append(session)
        with self._connect() as connection:
            for (project, agent), group in grouped.items():
                calls = [call for session in group for turn in session.turns for call in turn.tool_calls
                         if call.name.lower() == 'bash' and call.result_bytes >= thresholds.large_payload_bytes]
                if not calls:
                    continue
                action_id = hashlib.sha256(json.dumps([project, agent]).encode()).hexdigest()[:24]
                directories = sorted({str(session.path.parent) for session in group if not session.is_subagent})
                policy = (
                    'For commands expected to produce large output, explicitly opt into full-log capture: '
                    f'{shlex.join([sys.executable, "-m", "burnlens", "capture", "--action", action_id])} --task <same-task-label> -- COMMAND. '
                    'Full logs are retained and command exit status is preserved. Inspect the full log when needed. '
                    'Record baseline runs with --baseline using the same task label, then record each task outcome. '
                    'Do not rerun destructive commands just to collect a baseline. No command is rewritten automatically.'
                )
                revision = hashlib.sha256(json.dumps([policy, directories]).encode()).hexdigest()
                row = connection.execute('SELECT id FROM actions WHERE id = ?', (action_id,)).fetchone()
                existing = self._read(connection, action_id) if row else None
                state = existing.state if existing and existing.revision == revision else 'proposed'
                action = Action(action_id, revision, project, agent, state, policy, len(calls),
                                sum(call.result_bytes for call in calls), directories, existing.runs if existing else [])
                self._save(connection, action)
        return self.list_actions()

    def list_actions(self) -> list[Action]:
        if not self._database.exists():
            return []
        with self._connect() as connection:
            ids = [row[0] for row in connection.execute('SELECT id FROM actions ORDER BY id')]
            return [self._read(connection, action_id) for action_id in ids]

    def get_action(self, action_id: str) -> Action:
        with self._connect() as connection:
            return self._read(connection, action_id)

    def apply(self, action_id: str, revision: str) -> Action:
        with self._connect() as connection:
            action = self._read(connection, action_id)
            if action.revision != revision:
                self._fail('Proposal changed; review the current revision before applying')
            action.state = 'applied'
            self._save(connection, action)
            return action

    def revert(self, action_id: str) -> Action:
        with self._connect() as connection:
            action = self._read(connection, action_id)
            action.state = 'reverted'
            self._save(connection, action)
            return action

    def record_run(self, action_id: str, measurement: RunMeasurement) -> Action:
        for value in (measurement.run_id, measurement.task_label, measurement.output_path):
            if not isinstance(value, str) or not value.strip() or len(value) > MAX_TEXT_LENGTH:
                self._fail('Run strings must be nonempty and bounded')
        if (not isinstance(measurement.baseline, bool) or not isinstance(measurement.exit_code, int)
                or not math.isfinite(measurement.duration_seconds) or measurement.duration_seconds < 0
                or not isinstance(measurement.output_bytes, int) or measurement.output_bytes < 0
                or not isinstance(measurement.displayed_bytes, int) or measurement.displayed_bytes < 0
                or measurement.outcome not in OUTCOMES or len(measurement.notes) > MAX_TEXT_LENGTH):
            self._fail('Invalid run measurement')
        with self._connect() as connection:
            action = self._read(connection, action_id)
            if not measurement.baseline and action.state != 'applied' and measurement.policy_revision != action.revision:
                self._fail('Apply the reviewed intervention before recording a trial')
            if any(run.run_id == measurement.run_id for run in action.runs):
                self._fail('Run ID already recorded')
            action.runs.append(measurement)
            self._save(connection, action)
            return action

    def record_outcome(self, action_id: str, run_id: str, outcome: str, notes: str) -> Action:
        if not isinstance(outcome, str) or outcome not in OUTCOMES or not isinstance(notes, str) or len(notes) > MAX_TEXT_LENGTH:
            self._fail('Invalid outcome or notes')
        with self._connect() as connection:
            action = self._read(connection, action_id)
            for index, run in enumerate(action.runs):
                if run.run_id == run_id:
                    payload = asdict(run)
                    payload.update(outcome=outcome, notes=notes)
                    action.runs[index] = RunMeasurement(**payload)
                    self._save(connection, action)
                    return action
            self._fail('Unknown run ID')
        raise AssertionError('unreachable')

    def advice_for(self, transcript_path: str) -> str:
        if not isinstance(transcript_path, str) or '..' in transcript_path.split('/'):
            return ''
        for action in self.list_actions():
            if action.state == 'applied' and action.agent == 'claude-code':
                if transcript_path.rsplit('/', 1)[0] in action.transcript_dirs:
                    return action.policy_text
        return ''
