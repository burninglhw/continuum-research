import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRACE_ROOT = ROOT.parent / 'swe-traces'
sys.path.insert(0, str(TRACE_ROOT))

from core import Records, timestamp


def checked(command):
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=90).strip()


def main():
    provenance = ROOT / 'provenance'
    manifest = json.loads((provenance / 'validated-images.json').read_text())
    entry = manifest['images']['django__django-13413']
    validation = json.loads(Path(entry['validation_report']).read_text())
    assert validation['validation_status'] == 'passed'
    groups = json.loads((provenance / 'dependency-groups-retry.json').read_text())
    images = [json.loads((provenance / name).read_text()) for name in
              ['base-image.json', 'django-env-image.json', 'django-13413-image.json']]
    details = [json.loads(checked(['docker', 'image', 'inspect', record['image_id']]))[0]
               for record in images]
    for index in [1, 2]:
        parent = details[index - 1]['RootFS']['Layers']
        assert details[index]['RootFS']['Layers'][:len(parent)] == parent
    donor = checked(['docker', 'create', '--network=none', '--read-only',
                     '--label=continuum.trace.owner=ext.luohaowen1',
                     '--entrypoint=/bin/true', entry['image_id']])
    try:
        locks = provenance / 'software-locks'
        locks.mkdir(exist_ok=True)
        checked(['docker', 'cp', donor + ':/opt/continuum-build/.', str(locks)])
    finally:
        checked(['docker', 'rm', donor])
    tests = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'test_safety.py', 'test_local_images.py'],
                           cwd=TRACE_ROOT, capture_output=True, text=True, timeout=120)
    (provenance / 'collector-tests.txt').write_text(tests.stdout + tests.stderr)
    assert tests.returncode == 0
    first = subprocess.run([sys.executable, str(TRACE_ROOT / 'collect.py'), '--limit', '1', '--check-images-only'],
                           capture_output=True, text=True, timeout=60)
    assert first.returncode == 0 and '"api_called": false' in first.stdout
    third = subprocess.run([sys.executable, str(TRACE_ROOT / 'collect.py'), '--limit', '3', '--check-images-only'],
                           capture_output=True, text=True, timeout=60)
    assert third.returncode != 0 and 'No validated self-built image for django__django-10426' in third.stderr
    assert 'AUTH_CHECK_READY' not in third.stdout
    first_group = groups['tasks'][0]['dependency_group']
    run = TRACE_ROOT / 'runs/nowcoding-mixed-100-20260918'
    ledger = json.loads((run / 'budget.json').read_text())
    report = {
        'checked_at': timestamp(), 'status': 'self_built_first_task_ready',
        'validated_tasks': list(manifest['images']), 'selected_tasks': groups['task_count'],
        'candidate_dependency_groups': groups['candidate_dependency_group_count'],
        'unresolved_dependency_recipes': groups['unresolved_count'],
        'first_dependency_group_tasks': groups['group_members'][first_group],
        'images': images, 'shared_parent_layers_verified': True,
        'shared_image_chain_bytes': images[-1]['size_bytes'],
        'dependency_increment_bytes': images[1]['size_bytes'] - images[0]['size_bytes'],
        'task_increment_bytes': images[2]['size_bytes'] - images[1]['size_bytes'],
        'disk_free_bytes': shutil.disk_usage(ROOT).free,
        'image_check_limit_1': 'passed', 'image_check_limit_3': 'stops_before_credentials_or_api',
        'collector_test_output': tests.stdout.strip(), 'task_validation': str(Path(entry['validation_report'])),
        'actual_swebench_agent_traces': len(list((run / 'instances').glob('*/responses.jsonl'))),
        'model_calls_this_setup': 0, 'gpu_used_this_setup': False, 'budget_ledger': ledger,
        'original_continuum_checkout_clean': checked(['git', '-C', str(ROOT.parents[1] / 'vllm-continuum'),
                                                      'status', '--porcelain']) == '',
        'source_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in ROOT.iterdir() if path.is_file() and
                          (path.suffix in ['.py', '.sh', '.txt'] or path.name.startswith('Dockerfile'))},
    }
    Records(provenance).write('setup-status.json', report)
    if not (run / 'status-before-self-built.json').exists() and (run / 'status.json').exists():
        shutil.copyfile(run / 'status.json', run / 'status-before-self-built.json')
    Records(run).write('status.json', {'state': report['status'], 'validated_tasks': report['validated_tasks'],
                                     'collected_traces': report['actual_swebench_agent_traces'],
                                     'remaining_image_setup_tasks': 100 - len(manifest['images']),
                                     'setup_report': str(provenance / 'setup-status.json'),
                                     'paid_collection_started': False})
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
