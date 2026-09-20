import argparse
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
TRACE_ROOT = ROOT.parent / 'swe-traces'
sys.path.insert(0, str(TRACE_ROOT))

from collect import TimedDocker, checked
from core import Records, SafetyStop, timestamp
from local_images import inspect_image, load_manifest


def stage_source_cache(context, recipe):
    destination = context / 'source-cache'
    destination.mkdir(exist_ok=True)
    source = ROOT / 'source-cache' / (recipe['base_commit'] + '.json')
    if not source.exists():
        return
    metadata = json.loads(source.read_text())
    archive = source.with_suffix('.tar.gz')
    if (metadata.get('repo') != recipe['repo'] or metadata.get('base_commit') != recipe['base_commit']
            or hashlib.sha256(archive.read_bytes()).hexdigest() != metadata.get('sha256')):
        raise SafetyStop('Prepared source cache identity or checksum mismatch')
    shutil.copyfile(archive, destination / 'repository.tar.gz')
    shutil.copyfile(source, destination / 'source.json')


def image_build(recipe, mode, parent, *, disk_reserve_checks_enabled=True, build_timeout_enabled=True):
    identity = recipe['dependency_group'][:16] if mode == 'env' else recipe['instance_id'].replace('__', '-')
    tag = 'continuum-luohaowen/swe-auto-' + mode + ':' + identity
    context = ROOT / 'contexts' / (mode + '-' + identity)
    context.mkdir(parents=True, exist_ok=True)
    provenance = ROOT / 'provenance/automatic' / (mode + '-' + identity)
    provenance.mkdir(parents=True, exist_ok=True)
    result_path = provenance / 'image.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        inspection = subprocess.run(['docker', 'image', 'inspect', result['image_id']], capture_output=True, text=True)
        if inspection.returncode == 0:
            details = json.loads(inspection.stdout)[0]
            if details['Config']['Labels'].get('continuum.image.owner') != 'ext.luohaowen1':
                raise SafetyStop('Cached image ownership mismatch')
            return result
        Records(provenance).write('released-image-' + uuid.uuid4().hex[:8] + '.json', result)
    if subprocess.run(['docker', 'image', 'inspect', tag], capture_output=True).returncode == 0:
        raise SafetyStop('Unrecorded tag already exists; refusing to overwrite')
    for source in ['Dockerfile.generic-' + mode, 'configure_image.py', 'activate-testbed.sh', 'matplotlib-compat.json']:
        shutil.copyfile(ROOT / source, context / source)
    Records(context).write('recipe.json', recipe)
    if mode == 'task':
        stage_source_cache(context, recipe)
    if disk_reserve_checks_enabled and shutil.disk_usage(ROOT).free < 12 * 1024 ** 3:
        raise SafetyStop('Need 12 GiB free before the next image build')
    build_memory_gib = 16 if mode == 'env' and recipe.get('repo') == 'pydata/xarray' else 4
    build_resources = {'memory_gib': build_memory_gib, 'memory_swap_gib': build_memory_gib, 'cpus': 2}
    command = ['docker', 'build', '--force-rm', f'--memory={build_memory_gib}g', f'--memory-swap={build_memory_gib}g',
               '--cpu-period=100000', '--cpu-quota=200000', '--network=default',
               '--build-arg', 'BASE_IMAGE=' + parent, '--build-arg', 'TASK_ID=' + recipe['instance_id'],
               '--build-arg', 'BASE_COMMIT=' + recipe['base_commit'],
               '--file', 'Dockerfile.generic-' + mode, '--tag', tag, '.']
    started = time.monotonic()
    log_path = provenance / 'build.log'
    if log_path.exists():
        shutil.copyfile(log_path, provenance / ('build-previous-' + uuid.uuid4().hex[:8] + '.log'))
    invocation_path = provenance / 'build-invocation.json'
    if invocation_path.exists():
        shutil.copyfile(invocation_path, provenance / ('build-invocation-previous-' + uuid.uuid4().hex[:8] + '.json'))
    Records(provenance).write('build-invocation.json', {
        'started_at': timestamp(), 'command': command, 'build_resources': build_resources,
        'disk_reserve_checks_enabled': disk_reserve_checks_enabled, 'build_timeout_enabled': build_timeout_enabled,
    })
    with log_path.open('w') as log:
        process = subprocess.Popen(command, cwd=context, env={**os.environ, 'DOCKER_BUILDKIT': '0'},
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                if disk_reserve_checks_enabled and shutil.disk_usage(ROOT).free < 8 * 1024 ** 3:
                    raise SafetyStop('Build stopped at shared disk reserve')
                if build_timeout_enabled and time.monotonic() - started > 1800:
                    raise SafetyStop('Build exceeded 30 minutes')
                time.sleep(2)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if process.returncode:
        print(log_path.read_text()[-4500:], flush=True)
        raise SafetyStop('Image build failed: ' + str(log_path))
    details = json.loads(checked(['docker', 'image', 'inspect', tag]))[0]
    result = {'reference': tag, 'image_id': details['Id'], 'parent_id': parent,
              'size_bytes': details['Size'], 'build_seconds': time.monotonic() - started,
              'dependency_group': recipe['dependency_group'], 'instance_id': recipe['instance_id'],
              'disk_reserve_checks_enabled': disk_reserve_checks_enabled,
              'build_timeout_enabled': build_timeout_enabled,
              'build_resources': build_resources,
              'source_sha256': {str(path.relative_to(context)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in context.rglob('*') if path.is_file()}}
    Records(provenance).write('image.json', result)
    print(json.dumps({'image_ready': tag, 'size_bytes': details['Size']}), flush=True)
    return result


def smoke_command(repository):
    commands = {
        'django/django': 'python tests/runtests.py basic --verbosity 1 --settings=test_sqlite --parallel 1',
        'matplotlib/matplotlib': "MPLBACKEND=Agg python -c \"import matplotlib.pyplot as plot; plot.plot([0,1]); plot.savefig('/tmp/continuum-smoke.png')\"",
        'sympy/sympy': "python -c \"import sympy; variable=sympy.Symbol('x'); assert sympy.expand((variable+1)**2)==variable**2+2*variable+1\"",
        'scikit-learn/scikit-learn': "python -c \"from sklearn.linear_model import LinearRegression; model=LinearRegression().fit([[1],[2]],[1,2]); assert abs(model.predict([[3]])[0]-3)<1e-6\"",
        'sphinx-doc/sphinx': 'python -m sphinx --version',
        'pytest-dev/pytest': 'python -m pytest --version',
        'psf/requests': "python -c \"import requests; request=requests.Request('GET','https://example.com').prepare(); assert request.method=='GET'\"",
        'mwaskom/seaborn': "MPLBACKEND=Agg python -c \"import seaborn,matplotlib.pyplot as plot; seaborn.lineplot(x=[1,2],y=[1,2]); plot.savefig('/tmp/continuum-smoke.png')\"",
        'astropy/astropy': "python -c \"from astropy import units; assert (1*units.m).to_value(units.cm)==100\"",
        'pylint-dev/pylint': 'python -m pylint --version',
        'pydata/xarray': "python -c \"import xarray; assert xarray.DataArray([1,2,3]).mean().item()==2\"",
    }
    return commands[repository]


def validate(recipe, build, *, disk_reserve_checks_enabled=True):
    config = json.loads((TRACE_ROOT / 'run_config.json').read_text())
    config['disk_reserve_checks_enabled'] = disk_reserve_checks_enabled
    source = TRACE_ROOT / 'source/mini-swe-agent/src/minisweagent/config/extra/swebench.yaml'
    container_env = yaml.safe_load(source.read_text())['environment']['env']
    destination = ROOT / 'provenance/automatic' / ('validation-' + recipe['instance_id'] + '-' + uuid.uuid4().hex[:8])
    records = Records(destination)
    environment = None
    report = {'instance_id': recipe['instance_id'], 'image_id': build['image_id'],
              'validation_status': 'failed', 'swebench_resolved': None,
              'is_agent_trace': False, 'gpu_used': False, 'paid_model_called': False,
              'disk_reserve_checks_enabled': disk_reserve_checks_enabled}
    try:
        environment = TimedDocker(records, config, image=build['image_id'], cwd='/testbed', env=container_env)

        def execute(command):
            result = environment.execute(command)
            if result['returncode']:
                raise SafetyStop('Environment smoke failed: ' + command + '\n' + result['output'][-2000:])
            return result['output']

        report['head'] = execute('git rev-parse HEAD').strip()
        assert report['head'] == recipe['base_commit']
        report['python_version'] = execute('python --version').strip()
        assert report['python_version'].startswith('Python ' + recipe['python'] + '.')
        report['setup_diff'] = execute('git diff HEAD --binary')
        expected = execute('cat /opt/continuum-build/setup-diff.patch')
        assert report['setup_diff'] == expected
        execute('python -m pip check && test ! -e /dev/nvidia0')
        report['test_command'] = smoke_command(recipe['repo'])
        report['test_output'] = execute(report['test_command'])
        report['pip_freeze'] = execute('python -m pip freeze')
        report['conda_explicit'] = execute('cat /opt/continuum-build/conda-explicit.txt')
        report['validation_status'] = 'passed'
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        if environment:
            environment.cleanup()
        records.write('validation.json', report)
    path = ROOT / 'provenance/validated-images.json'
    manifest = load_manifest(path)
    entry = {'reference': build['reference'], 'image_id': build['image_id'], 'base_commit': recipe['base_commit'],
             'validation_status': 'passed', 'owner': 'ext.luohaowen1',
             'validation_report': str(destination / 'validation.json'),
             'validation_report_sha256': hashlib.sha256((destination / 'validation.json').read_bytes()).hexdigest()}
    manifest['images'][recipe['instance_id']] = entry
    Records(path.parent).write(path.name, manifest)
    print(json.dumps({'task_validated': recipe['instance_id']}), flush=True)


def ensure(instance_id, *, disk_reserve_checks_enabled=True, build_timeout_enabled=True):
    if (TRACE_ROOT / 'pause-collection.json').exists():
        raise SafetyStop('Collection paused at task boundary for a recorded quality review')
    tasks = [json.loads(line) for line in (TRACE_ROOT / 'tasks-100.jsonl').read_text().splitlines()]
    task = next(record for record in tasks if record['instance_id'] == instance_id)
    path = ROOT / 'provenance/validated-images.json'
    manifest = load_manifest(path)
    if instance_id in manifest['images']:
        image_id = manifest['images'][instance_id]['image_id']
        if subprocess.run(['docker', 'image', 'inspect', image_id], capture_output=True).returncode == 0:
            inspect_image(task, manifest)
            return
    audit = json.loads((ROOT / 'provenance/dependency-groups-retry.json').read_text())
    recipe = next(record for record in audit['tasks'] if record['instance_id'] == instance_id)
    assert recipe['base_commit'] == task['base_commit'] and recipe['repo'] == task['repo']
    with (ROOT / 'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if recipe['dependency_group'] == audit['tasks'][0]['dependency_group']:
            parent = json.loads((ROOT / 'provenance/django-env-image.json').read_text())['image_id']
        else:
            base = json.loads((ROOT / 'provenance/base-image.json').read_text())['image_id']
            parent = image_build(recipe, 'env', base, disk_reserve_checks_enabled=disk_reserve_checks_enabled,
                                 build_timeout_enabled=build_timeout_enabled)['image_id']
        build = image_build(recipe, 'task', parent, disk_reserve_checks_enabled=disk_reserve_checks_enabled,
                            build_timeout_enabled=build_timeout_enabled)
        validate(recipe, build, disk_reserve_checks_enabled=disk_reserve_checks_enabled)


def release_own_reference(reference, image_id):
    if not reference.startswith('continuum-luohaowen/swe-auto-'):
        return {'reference': reference, 'released': False, 'reason': 'protected_nonautomatic_image'}
    result = subprocess.run(['docker', 'image', 'inspect', reference], capture_output=True, text=True)
    if result.returncode:
        return {'reference': reference, 'released': True, 'already_absent': True}
    details = json.loads(result.stdout)[0]
    labels = details['Config'].get('Labels') or {}
    if details['Id'] != image_id or labels.get('continuum.image.owner') != 'ext.luohaowen1' or labels.get('continuum.image.managed') != 'swe100':
        raise SafetyStop('Refusing to release an image with mismatched ownership or identity')
    removal = subprocess.run(['docker', 'image', 'rm', reference], capture_output=True, text=True, timeout=90)
    return {'reference': reference, 'image_id': image_id, 'released': removal.returncode == 0,
            'forced': False, 'detail': (removal.stdout + removal.stderr)[-1200:]}


def archive_task_environment(instance_id, run):
    case = run / 'instances' / instance_id
    recorded_image = json.loads((case / 'image.json').read_text())
    entry = recorded_image['validation']
    image_id = recorded_image['image_id']
    task = json.loads((case / 'task.json').read_text())
    report_bytes = Path(entry['validation_report']).read_bytes()
    report = json.loads(report_bytes)
    if (entry['image_id'] != image_id or task['instance_id'] != instance_id
            or hashlib.sha256(report_bytes).hexdigest() != entry['validation_report_sha256']
            or report['image_id'] != image_id or report['instance_id'] != instance_id
            or report['head'] != task['base_commit'] or report['validation_status'] != 'passed'):
        raise SafetyStop('Trace environment evidence identity or checksum mismatch')
    archive_path = case / 'environment.json'
    if archive_path.exists():
        previous = json.loads(archive_path.read_text())
        if previous['image_id'] != image_id or previous['instance_id'] != instance_id:
            raise SafetyStop('Existing environment archive identity mismatch')
        if previous.get('image_capture_status') == 'captured':
            return previous
    audit = json.loads((ROOT / 'provenance/dependency-groups-retry.json').read_text())
    recipe = next(record for record in audit['tasks'] if record['instance_id'] == instance_id)
    if recipe['base_commit'] != task['base_commit'] or recipe['repo'] != task['repo']:
        raise SafetyStop('Environment recipe does not match the recorded task')
    archive = {'schema_version': 1, 'captured_at': timestamp(), 'instance_id': instance_id,
               'image_id': image_id, 'recorded_image': recorded_image, 'recipe': recipe,
               'validation_report': report, 'validation_report_sha256': entry['validation_report_sha256'],
               'trace_execution': {name: json.loads((case / name).read_text())
                                   for name in ['execution-provenance.json', 'isolation.json', 'baseline.json']
                                   if (case / name).exists()},
               'image_capture_status': 'image_already_absent', 'image_files': {},
               'missing_evidence': ['live Docker metadata and in-image build records unavailable'],
               'bit_identical_rebuild_guaranteed': False}
    identity = instance_id.replace('__', '-')
    build_path = ROOT / 'provenance/automatic' / ('task-' + identity) / 'image.json'
    if build_path.exists():
        build = json.loads(build_path.read_text())
        if build['image_id'] == image_id:
            archive['build_record'] = build
            archive['build_context_text'] = {}
            context = ROOT / 'contexts' / ('task-' + identity)
            for name, expected in build['source_sha256'].items():
                source = context / name
                if not source.resolve().is_relative_to(context.resolve()):
                    raise SafetyStop('Build provenance points outside its own context')
                if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                    raise SafetyStop('Recorded build source is missing or changed: ' + name)
                if not name.endswith('.tar.gz'):
                    archive['build_context_text'][name] = source.read_text()
    inspection = subprocess.run(['docker', 'image', 'inspect', image_id], capture_output=True, text=True)
    if inspection.returncode == 0:
        details = json.loads(inspection.stdout)[0]
        labels = details['Config'].get('Labels') or {}
        if (details['Id'] != image_id or labels.get('continuum.image.owner') != 'ext.luohaowen1'
                or labels.get('continuum.image.task') != instance_id
                or labels.get('continuum.image.base_commit') != task['base_commit']):
            raise SafetyStop('Refusing environment capture from a mismatched image')
        files = ['/etc/os-release'] + ['/opt/continuum-build/' + name for name in [
            'recipe.json', 'environment-commands.json', 'task-commands.json',
            'dependency-pip-freeze.txt', 'pip-freeze.txt', 'conda-explicit.txt',
            'dpkg-versions.txt', 'setup-diff.patch', 'setup-status.txt']]
        capture = ('import json; from pathlib import Path; paths=' + repr(files)
                   + '; print(json.dumps({name:Path(name).read_text() if Path(name).is_file() else None for name in paths}))')
        payload = checked(['docker', 'run', '--rm', '--pull=never', '--network', 'none', '--read-only',
                           '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                           '--memory', '128m', '--memory-swap', '128m', '--cpus', '1', '--pids-limit', '32',
                           '--label', 'continuum.trace.owner=ext.luohaowen1',
                           '--entrypoint', '/opt/miniconda3/bin/python', image_id, '-c', capture], timeout=120)
        captured = json.loads(payload)
        required = ['/etc/os-release', '/opt/continuum-build/pip-freeze.txt',
                    '/opt/continuum-build/conda-explicit.txt', '/opt/continuum-build/dpkg-versions.txt']
        missing = [name for name in files if captured.get(name) is None]
        if entry['reference'].startswith('continuum-luohaowen/swe-auto-') and any(name in missing for name in required):
            raise SafetyStop('Required environment records missing; image will not be released')
        archive.update(image_capture_status='captured', docker_inspect=details, image_files=captured,
                       image_file_sha256={name: hashlib.sha256(value.encode()).hexdigest()
                                          for name, value in captured.items() if value is not None},
                       missing_evidence=missing)
    Records(case).write('environment.json', archive)
    return archive


def release_completed(instance_id, group_id, run_dir=None):
    run = Path(run_dir).resolve() if run_dir else TRACE_ROOT / 'runs/nowcoding-mixed-100-20260918'
    if not run.is_relative_to((TRACE_ROOT / 'runs').resolve()):
        raise SafetyStop('Run directory must belong to this trace collector')
    metrics = json.loads((run / 'instances' / instance_id / 'metrics.json').read_text())
    if not metrics.get('collection_completed'):
        raise SafetyStop('An incomplete task image must be retained for recovery')
    archive = archive_task_environment(instance_id, run)
    entry = archive['recorded_image']['validation']
    if archive['image_capture_status'] == 'captured':
        result = release_own_reference(entry['reference'], entry['image_id'])
    else:
        result = {'reference': entry['reference'], 'image_id': entry['image_id'],
                  'released': True, 'already_absent': True}
    archive_path = run / 'instances' / instance_id / 'environment.json'
    result.update(environment_archive=str(archive_path),
                  environment_archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest())
    Records(ROOT / 'provenance').event('image-cache-releases.jsonl', result)
    if group_id:
        audit = json.loads((ROOT / 'provenance/dependency-groups-retry.json').read_text())
        if instance_id not in audit['group_members'].get(group_id, []):
            raise SafetyStop('Group membership mismatch')
        for member in audit['group_members'][group_id]:
            member_metrics = run / 'instances' / member / 'metrics.json'
            if not member_metrics.exists() or not json.loads(member_metrics.read_text()).get('collection_completed'):
                return
        path = ROOT / 'provenance/automatic' / ('env-' + group_id[:16]) / 'image.json'
        if path.exists():
            parent = json.loads(path.read_text())
            if parent['dependency_group'] != group_id:
                raise SafetyStop('Group image provenance mismatch')
            Records(ROOT / 'provenance').event('image-cache-releases.jsonl',
                release_own_reference(parent['reference'], parent['image_id']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('instance_id')
    parser.add_argument('--release-task', action='store_true')
    parser.add_argument('--release-group')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--disable-disk-reserve-checks', action='store_true')
    parser.add_argument('--disable-build-timeout', action='store_true')
    arguments = parser.parse_args()
    os.umask(0o077)
    if arguments.release_task:
        release_completed(arguments.instance_id, arguments.release_group, arguments.run_dir)
    else:
        ensure(arguments.instance_id, disk_reserve_checks_enabled=not arguments.disable_disk_reserve_checks,
               build_timeout_enabled=not arguments.disable_build_timeout)
