import json
from types import SimpleNamespace

import pytest

import collect
import task_checkpoint
from core import Records, SafetyStop, record_configuration


def test_collection_reserve_defaults_on_and_can_be_disabled(monkeypatch):
    monkeypatch.setattr(collect.shutil, 'disk_usage', lambda path: SimpleNamespace(free=1))
    with pytest.raises(SafetyStop, match='Shared disk reserve'):
        collect.disk_guard({'minimum_free_gib': 6})
    assert collect.disk_guard({'minimum_free_gib': 6, 'disk_reserve_checks_enabled': False}) == 1


@pytest.mark.parametrize('enabled', [True, False])
def test_image_preparation_receives_run_reserve_policy(monkeypatch, enabled):
    commands = []
    monkeypatch.setattr(collect.subprocess, 'run', lambda command, **kwargs: commands.append(command))
    collect.prepare_task_image({'instance_id': 'sample'}, {'disk_reserve_checks_enabled': enabled})
    assert ('--disable-disk-reserve-checks' in commands[0]) is (not enabled)


@pytest.mark.parametrize('enabled', [True, False])
def test_image_preparation_time_limits_follow_run_policy(monkeypatch, enabled):
    observed = {}

    def capture(command, **kwargs):
        observed.update(command=command, **kwargs)

    monkeypatch.setattr(collect.subprocess, 'run', capture)
    collect.prepare_task_image({'instance_id': 'sample'}, {'image_preparation_time_limits_enabled': enabled})
    assert ('--disable-build-timeout' in observed['command']) is (not enabled)
    assert observed['timeout'] == (3900 if enabled else None)


def test_preparation_time_policy_change_is_audited(tmp_path):
    records = Records(tmp_path)
    record_configuration(records, {})
    updated = {'image_preparation_time_limits_enabled': False}
    record_configuration(records, updated, runtime_change_reason='User requested no arbitrary task limits')
    event = json.loads((tmp_path / 'runtime-policy-changes.jsonl').read_text())
    assert event['previous_config'] == {}
    assert event['api_sampling_changed'] is False


def test_reserve_policy_change_requires_review_and_preserves_history(tmp_path):
    records = Records(tmp_path)
    original = {'minimum_free_gib': 6}
    updated = {**original, 'disk_reserve_checks_enabled': False}
    record_configuration(records, original)
    with pytest.raises(SafetyStop, match='Configuration changed'):
        record_configuration(records, updated)
    record_configuration(records, updated, runtime_change_reason='User explicitly disabled disk reserves')
    event = json.loads((tmp_path / 'runtime-policy-changes.jsonl').read_text())
    assert event['previous_config'] == original
    assert event['updated_config'] == updated
    assert event['api_sampling_changed'] is False
    assert event['previous_attempts_preserved'] is True


@pytest.mark.parametrize('enabled', [True, False])
def test_checkpoint_disk_reserves_follow_run_policy(tmp_path, monkeypatch, enabled):
    records = Records(tmp_path)
    records.write('trajectory.json', {'messages': []})
    environment = SimpleNamespace(container_id='sample', run_config={'disk_reserve_checks_enabled': enabled})
    monkeypatch.setattr(task_checkpoint.shutil, 'disk_usage', lambda path: SimpleNamespace(free=1))
    monkeypatch.setattr(task_checkpoint.time, 'sleep', lambda duration: None)
    observed = []

    def export(command, *, stdout, stderr):
        observed.append(command)
        stdout.write(b'checkpoint payload')
        results = iter([None, 0])
        return SimpleNamespace(poll=lambda: next(results), returncode=0)

    monkeypatch.setattr(task_checkpoint.subprocess, 'Popen', export)
    task_checkpoint.save_checkpoint(environment, records, {'instance_id': 'sample'}, 'sha256:image')
    if enabled:
        assert observed == []
        assert 'Insufficient free disk' in json.loads((tmp_path / 'checkpoint-error.json').read_text())['error']
    else:
        assert len(observed) == 1
        assert json.loads((tmp_path / 'checkpoint.json').read_text())['status'] == 'ready'
        assert not (tmp_path / 'checkpoint-error.json').exists()
