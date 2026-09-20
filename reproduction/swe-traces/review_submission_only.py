import fcntl
import hashlib
import json
import subprocess
import time
from pathlib import Path

from core import Records, SafetyStop
from quality import audit_case, events


def main():
    root = Path(__file__).resolve().parent
    run = root / 'runs/nowcoding-mixed-100-20260918'
    images = root.parent / 'swe-own-image/provenance'
    records = Records(run)
    with (run / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget_bytes = (run / 'budget.json').read_bytes()
        if json.loads(budget_bytes)['pending']:
            raise SafetyStop('Pending request must be reconciled first')
        manifest = json.loads((images / 'validated-images.json').read_text())
        releases = events(images / 'image-cache-releases.jsonl')
        for task_id in ['django__django-13212', 'django__django-12936']:
            case = run / 'instances' / task_id
            if not case.exists():
                raise SafetyStop('Already archived or missing task; do not duplicate this migration')
            audit = audit_case(case)
            if (audit['non_submission_tool_calls'] != 0 or audit['patch_bytes'] != 0
                    or audit['exit_status'] != 'Submitted' or 'response_format_errors' not in audit['quality_flags']):
                raise SafetyStop('Only structurally invalid submission-only attempts are eligible')
            entry = manifest['images'][task_id]
            present = subprocess.run(['docker', 'image', 'inspect', entry['image_id']],
                                     capture_output=True, timeout=30).returncode == 0
            if not present and not any(event.get('image_id') == entry['image_id'] and event.get('released') for event in releases):
                raise SafetyStop('Missing image is not explained by this run cache-release log')
            destination = run / 'invalid-attempts' / task_id / time.strftime('%Y%m%d-%H%M%S')
            if destination.parent.exists():
                raise SafetyStop('One structural-format restart per task; review before any further restart')
            response_cost = sum(event['conservative_cost_cny'] for event in events(case / 'responses.jsonl'))
            destination.parent.mkdir(parents=True)
            case.rename(destination)
            records.event('attempt-cost-carryovers.jsonl', {
                'task_id': task_id, 'completed_response_cost_cny': response_cost,
                'archive': str(destination), 'global_budget_unchanged': True,
            })
            Records(destination).write('structural-review.json', {
                'audit': audit, 'eligible_for_100_trace_target': False,
                'reason': 'No task tool command executed; only a submission following rejected simulated actions',
                'fixed_task_replaced': False, 'original_records_preserved': True,
                'next_attempt_policy': 'real-tool-feedback-v2',
            })
            if not present:
                del manifest['images'][task_id]
                image_record = images / 'automatic' / ('task-' + task_id.replace('__', '-')) / 'image.json'
                if image_record.exists():
                    if json.loads(image_record.read_text())['image_id'] != entry['image_id']:
                        raise SafetyStop('Build record image mismatch')
                    image_record.rename(image_record.with_name('image-before-structural-review.json'))
            Records(images).write('validated-images.json', manifest)
            records.event('reviewed-restarts.jsonl', {'instance_id': task_id,
                'reason': 'Structural invalidity, not answer correctness or desired paper statistics',
                'archived_attempt': str(destination), 'prior_response_cost_cny': response_cost,
                'sample_replacements': 0})
        if (run / 'budget.json').read_bytes() != budget_bytes:
            raise SafetyStop('Budget must not be changed by structural review')
        records.event('agent-policy-changes.jsonl', {
            'policy_version': 'real-tool-feedback-v2',
            'reason': 'Rejected multiple-action responses were mistaken for executed commands by the model',
            'format_error_feedback_changed': True, 'submission_requires_real_task_command': True,
            'system_and_task_templates_changed': False, 'forced_target_turns_or_delays': False,
            'collector_sha256': hashlib.sha256((root / 'collect.py').read_bytes()).hexdigest(),
            'original_completed_trace_preserved': 'django__django-13413',
        })
        print('STRUCTURAL_REVIEW_COMPLETE_BUDGET_AND_ORIGINAL_ATTEMPTS_RETAINED')


if __name__ == '__main__':
    main()
