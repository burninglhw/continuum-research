import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE_TAG = 'continuum-luohaowen/swe-base:ubuntu22.04-20260918'
ENV_TAG = 'continuum-luohaowen/swe-env:django-py36-13413deps-20260918'
TASK_TAG = 'continuum-luohaowen/swe-django:13413-20260918'


def build(stage, tag, recipe, extra):
    if shutil.disk_usage(ROOT).free < 12 * 1024 ** 3:
        raise RuntimeError('At least 12 GiB free required before an isolated build')
    existing = subprocess.run(['docker', 'image', 'inspect', tag], capture_output=True)
    if existing.returncode == 0:
        raise RuntimeError('Tag already exists; refusing to overwrite: ' + tag)
    command = ['docker', 'build', '--force-rm', '--network=default', '--memory=4g',
               '--memory-swap=4g', '--cpu-period=100000', '--cpu-quota=200000',
               '--label=continuum.image.owner=ext.luohaowen1', '--tag', tag,
               '--file', recipe, *extra, '.']
    started = time.monotonic()
    log_path = ROOT / 'provenance' / (stage + '-build.log')
    environment = {**os.environ, 'DOCKER_BUILDKIT': '0'}
    with log_path.open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                if shutil.disk_usage(ROOT).free < 8 * 1024 ** 3:
                    raise RuntimeError('Shared disk reserve reached')
                if time.monotonic() - started > 1800:
                    raise RuntimeError('Build exceeded 30-minute limit')
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
        print(log_path.read_text()[-7000:], flush=True)
        raise RuntimeError('Build failed; see ' + str(log_path))
    details = json.loads(subprocess.check_output(['docker', 'image', 'inspect', tag]))[0]
    result = {'tag': tag, 'image_id': details['Id'], 'size_bytes': details['Size'],
              'build_seconds': time.monotonic() - started, 'recipe': recipe,
              'recipe_sha256': hashlib.sha256((ROOT / recipe).read_bytes()).hexdigest(),
              'build_cpus': 2, 'build_memory_gib': 4, 'uses_other_users_images': False,
              'gpu_used': False, 'paid_model_called': False}
    (ROOT / 'provenance' / (stage + '-image.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['base', 'env', 'task'])
    arguments = parser.parse_args()
    os.umask(0o077)
    (ROOT / 'provenance').mkdir(exist_ok=True)
    with (ROOT / 'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if arguments.stage == 'base':
            build('base', BASE_TAG, 'Dockerfile.base', [])
        elif arguments.stage == 'env':
            image = json.loads((ROOT / 'provenance/base-image.json').read_text())
            build('django-env', ENV_TAG, 'Dockerfile.env-django',
                  ['--build-arg', 'BASE_IMAGE=' + image['image_id']])
        else:
            image = json.loads((ROOT / 'provenance/django-env-image.json').read_text())
            build('django-13413', TASK_TAG, 'Dockerfile.django-13413',
                  ['--build-arg', 'BASE_IMAGE=' + image['image_id']])


if __name__ == '__main__':
    main()
