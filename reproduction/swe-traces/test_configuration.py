import json

import pytest

from core import Budget, Records, SafetyStop, record_configuration


def initial_config(tmp_path):
    config = {'per_task_stop_cny': 3, 'budget_stop_cny': 270, 'model': 'deepseek-flash'}
    record_configuration(Records(tmp_path), config)
    return config


def test_same_configuration_needs_no_review(tmp_path):
    config = initial_config(tmp_path)
    record_configuration(Records(tmp_path), config)
    assert not (tmp_path / 'budget-policy-changes.jsonl').exists()


def test_reviewed_budget_change_preserves_original_configuration(tmp_path):
    config = initial_config(tmp_path)
    updated = {**config, 'per_task_stop_cny': 8}
    record_configuration(Records(tmp_path), updated, 'Reviewed per-task budget stop')
    event = json.loads((tmp_path / 'budget-policy-changes.jsonl').read_text())
    assert event['previous_config'] == config
    assert event['updated_config'] == updated
    assert event['api_sampling_changed'] is False
    assert json.loads((tmp_path / 'config.snapshot.json').read_text()) == updated


def test_user_can_remove_all_spending_caps_with_an_audit_record(tmp_path):
    config = initial_config(tmp_path)
    updated = {**config, 'per_task_stop_cny': None, 'budget_stop_cny': None,
               'budget_authorized_cny': None}
    record_configuration(Records(tmp_path), updated, 'User explicitly removed all spending caps')
    event = json.loads((tmp_path / 'budget-policy-changes.jsonl').read_text())
    assert event['spending_caps_disabled'] is True
    assert event['prior_costs_reset'] is False


def test_unlimited_spending_still_records_reservations_and_usage(tmp_path):
    config = {'per_task_stop_cny': None, 'budget_stop_cny': None,
              'price_input_cny_per_million': 2, 'price_output_cny_per_million': 8}
    budget = Budget(Records(tmp_path), config)
    budget.state['charged_upper_estimate_cny'] = 500
    budget.reserve(1000, 1000, 100, 'long-task')
    assert budget.settle({'prompt_tokens': 1000, 'completion_tokens': 100}) == pytest.approx(0.0028)
    assert budget.state['charged_upper_estimate_cny'] == pytest.approx(500.0028)
    assert budget.state['pending'] is None


def test_reviewed_runtime_change_is_separately_audited(tmp_path):
    config = initial_config(tmp_path)
    updated = {**config, 'max_completion_tokens': None, 'api_max_completion_tokens': 393216,
               'max_steps': 0, 'job_timeout_seconds': 0, 'max_format_corrections': None,
               'api_timeout_seconds': 600}
    record_configuration(Records(tmp_path), updated, runtime_change_reason='User requested continuous execution')
    event = json.loads((tmp_path / 'runtime-policy-changes.jsonl').read_text())
    assert event['previous_attempts_preserved'] is True
    assert event['api_sampling_changed'] is True
    assert event['previous_config'] == config


def test_runtime_review_cannot_change_the_model(tmp_path):
    config = initial_config(tmp_path)
    with pytest.raises(SafetyStop, match='Configuration changed'):
        record_configuration(Records(tmp_path), {**config, 'model': 'different'}, runtime_change_reason='review')


@pytest.mark.parametrize('updates,reason', [
    ({'per_task_stop_cny': 8}, None),
    ({'model': 'different'}, 'review'),
    ({'budget_stop_cny': 300}, 'review'),
    ({'per_task_stop_cny': 9}, 'review'),
    ({'per_task_stop_cny': -1}, 'review'),
    ({'per_task_stop_cny': float('nan')}, 'review'),
    ({'per_task_stop_cny': True}, 'review'),
])
def test_unreviewed_or_unrelated_changes_are_rejected(tmp_path, updates, reason):
    config = initial_config(tmp_path)
    with pytest.raises(SafetyStop, match='Configuration changed'):
        record_configuration(Records(tmp_path), {**config, **updates}, reason)
    assert json.loads((tmp_path / 'config.snapshot.json').read_text()) == config
    assert not (tmp_path / 'budget-policy-changes.jsonl').exists()
