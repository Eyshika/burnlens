from datetime import datetime, timezone
from pathlib import Path

import pytest

from burnlens.findings import Thresholds
from burnlens.interventions import InterventionConfig, InterventionError, InterventionService, RunMeasurement
from burnlens.model import Session, ToolCall, Turn, Usage


def make_session(path: Path) -> Session:
    return Session('session', 'project', path, None, 'task', turns=[
        Turn('turn', datetime.now(timezone.utc), 'configured-model', Usage(), [
            ToolCall('tool', 'Bash', {}, result_bytes=100000)])])


def test_review_apply_measure_revert(tmp_path: Path) -> None:
    service = InterventionService(InterventionConfig(tmp_path / 'state'))
    assert service.list_actions() == []
    assert not (tmp_path / 'state').exists()
    session = make_session(tmp_path / 'transcripts' / 'session.jsonl')
    action = service.refresh([session], Thresholds())[0]
    assert action.observed_bytes == 100000
    assert service.advice_for(str(session.path)) == ''
    with pytest.raises(InterventionError):
        service.apply(action.id, 'old-revision')
    service.record_run(action.id, RunMeasurement('base', 'task', True, 0, 1, 1000, 1000, '/generated/log'))
    with pytest.raises(InterventionError):
        service.record_run(action.id, RunMeasurement('trial', 'task', False, 0, 1, 1000, 100, '/generated/log'))
    service.apply(action.id, action.revision)
    assert '--action' in service.advice_for(str(session.path))
    assert service.advice_for(str(session.path.parent / 'other' / 'session.jsonl')) == ''
    action = service.record_run(action.id, RunMeasurement('trial', 'task', False, 0, 1, 1000, 100, '/generated/log'))
    assert action.as_dict()['measurement']['output_bytes_reduced'] == 900
    assert action.as_dict()['measurement']['savings'] == 'unmeasured'
    action = service.record_outcome(action.id, 'trial', 'success', 'checked task result')
    assert 'User-reported' in action.as_dict()['measurement']['quality']
    reopened = InterventionService(InterventionConfig(tmp_path / 'state'))
    assert len(reopened.get_action(action.id).runs) == 2
    reopened.revert(action.id)
    assert reopened.advice_for(str(session.path)) == ''


def test_changed_scope_requires_review_and_unknown_ids_are_rejected(tmp_path: Path) -> None:
    service = InterventionService(InterventionConfig(tmp_path / 'state'))
    action = service.refresh([make_session(tmp_path / 'one' / 'session')], Thresholds())[0]
    service.apply(action.id, action.revision)
    updated = service.refresh([make_session(tmp_path / 'two' / 'session')], Thresholds())[0]
    assert updated.state == 'proposed'
    with pytest.raises(InterventionError):
        service.apply(action.id, action.revision)
    with pytest.raises(InterventionError):
        service.get_action('../../unknown')


def test_duplicate_and_invalid_measurement_rejected(tmp_path: Path) -> None:
    service = InterventionService(InterventionConfig(tmp_path))
    action = service.refresh([make_session(tmp_path / 'session')], Thresholds())[0]
    run = RunMeasurement('base', 'task', True, 0, 1, 1000, 1000, '/generated/log')
    service.record_run(action.id, run)
    with pytest.raises(InterventionError):
        service.record_run(action.id, run)
    with pytest.raises(InterventionError):
        service.record_outcome(action.id, 'base', 'verified', '')
    with pytest.raises(InterventionError):
        service.record_run(action.id, RunMeasurement('bad', 'task', True, 0, float('nan'), 1, 1, '/generated/log'))
