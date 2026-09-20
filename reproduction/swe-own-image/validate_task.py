import hashlib
import json
import os
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
TRACE_ROOT = ROOT.parent / 'swe-traces'
sys.path.insert(0, str(TRACE_ROOT))

from collect import TimedDocker, checked
from core import Records


def execute_checked(environment, command, timeout=60):
    result = environment.execute(command, timeout=timeout)
    if result['returncode']:
        raise RuntimeError('Validation command failed: ' + command + '\n' + result['output'][-2500:])
    return result['output']


def main():
    os.umask(0o077)
    task = json.loads((TRACE_ROOT / 'tasks-100.jsonl').read_text().splitlines()[0])
    assert task['instance_id'] == 'django__django-13413'
    build = json.loads((ROOT / 'provenance/django-13413-image.json').read_text())
    image = json.loads(checked(['docker', 'image', 'inspect', build['image_id']]))[0]
    assert image['Config']['Labels']['continuum.image.owner'] == 'ext.luohaowen1'
    assert image['Config']['Labels']['continuum.image.base_commit'] == task['base_commit']
    config = json.loads((TRACE_ROOT / 'run_config.json').read_text())
    source = TRACE_ROOT / 'source/mini-swe-agent/src/minisweagent/config/extra/swebench.yaml'
    container_env = yaml.safe_load(source.read_text())['environment']['env']
    run = ROOT / 'provenance' / ('validation-' + time.strftime('%Y%m%d-%H%M%S'))
    run.mkdir()
    records = Records(run)
    environment = None
    report = {'instance_id': task['instance_id'], 'image_id': image['Id'],
              'validation_status': 'failed', 'paid_model_called': False, 'gpu_used': False,
              'swebench_resolved': None, 'is_agent_trace': False}
    try:
        environment = TimedDocker(records, config, image=image['Id'], cwd='/testbed', env=container_env)
        report['python_version'] = execute_checked(environment, 'python --version').strip()
        assert report['python_version'].startswith('Python 3.6.')
        report['head'] = execute_checked(environment, 'git rev-parse HEAD').strip()
        assert report['head'] == task['base_commit']
        assert not execute_checked(environment, 'git status --porcelain').strip()
        execute_checked(environment, 'python -m pip check && test ! -e /dev/nvidia0')
        report['django_version'] = execute_checked(environment, 'python -m django --version').strip()
        test_command = ('python tests/runtests.py basic admin_filters --verbosity 1 '
                        '--settings=test_sqlite --parallel 1')
        report['test_command'] = test_command
        report['test_output'] = execute_checked(environment, test_command)
        report['pip_freeze'] = execute_checked(environment, 'python -m pip freeze')
        execute_checked(environment, "printf 'isolated\\n' > /testbed/.continuum-isolation-proof")
        report['first_container_id'] = environment.container_id
        environment.cleanup()
        environment = TimedDocker(Records(run / 'second-container'), config,
                                  image=image['Id'], cwd='/testbed', env=container_env)
        execute_checked(environment, 'test ! -e /testbed/.continuum-isolation-proof')
        assert not execute_checked(environment, 'git status --porcelain').strip()
        report.update(validation_status='passed', fresh_workspace_verified=True,
                      second_container_id=environment.container_id)
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        if environment:
            environment.cleanup()
        records.write('validation.json', report)
    manifest_path = ROOT / 'provenance/validated-images.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'schema_version': 1, 'images': {}}
    entry = {'reference': build['tag'], 'image_id': image['Id'], 'base_commit': task['base_commit'],
             'validation_status': 'passed', 'owner': 'ext.luohaowen1',
             'validation_report': str(run / 'validation.json'),
             'validation_report_sha256': hashlib.sha256((run / 'validation.json').read_bytes()).hexdigest()}
    manifest['images'][task['instance_id']] = entry
    Records(ROOT / 'provenance').write('validated-images.json', manifest)
    print(json.dumps({key: report[key] for key in ['instance_id', 'validation_status', 'python_version',
                                                  'django_version', 'fresh_workspace_verified']}, indent=2))
    print('REPORT=' + str(run / 'validation.json'))


if __name__ == '__main__':
    main()
