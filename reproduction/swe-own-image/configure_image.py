import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


def run_shell(commands):
    subprocess.run(['/bin/bash', '-lc', 'set -euo pipefail\n' + '\n'.join(commands)], check=True)


def task_compatibility_commands(recipe):
    if (recipe['repo'] == 'mwaskom/seaborn' and recipe['version'] == '0.12'
            and 'pandas==2.0.0' in recipe['official_spec'].get('pip_packages', [])):
        return ['python -m pip install --no-deps pandas==1.5.3']
    return []


def xarray_environment_constraints(recipe):
    if (recipe['repo'] == 'pydata/xarray' and recipe['version'] == '2022.06'
            and recipe['official_spec'].get('packages') == 'environment.yml'
            and 'dask==2022.8.1' in recipe['official_spec'].get('pip_packages', [])):
        return {
            'conda': {'distributed': '2022.8.1', 'xarray': '2022.3.0', 'wheel': '0.45.1', 'rasterio': '1.3.9'},
            'pip': ['distributed==2022.8.1', 'xarray==2022.3.0', 'wheel==0.45.1', 'rasterio==1.3.9'],
        }
    return None


def constrain_environment(command, recipe, constraints):
    forced = {}
    for package in recipe['official_spec'].get('pip_packages', []):
        match = re.fullmatch(r'([A-Za-z0-9_-]+)==([0-9.]+)', package)
        if match:
            forced[match[1].replace('-', '_').lower()] = match[2]
    forced.update({name.replace('-', '_'): version for name, version in constraints['conda'].items()})
    present = set()
    lines = []
    for line in command.splitlines():
        match = re.match(r'^  - ([A-Za-z0-9_-]+)(.*)$', line)
        if match:
            normalized = match[1].replace('-', '_').lower()
            present.add(normalized)
            if normalized in forced:
                line = '  - ' + match[1] + '=' + forced[normalized]
        lines.append(line)
    aliases = {'python_dateutil': 'python-dateutil'}
    additions = ['  - ' + aliases.get(name, name) + '=' + version
                 for name, version in forced.items() if name not in present]
    return '\n'.join(lines).replace('dependencies:\n', 'dependencies:\n' + '\n'.join(additions) + '\n', 1)


def main():
    if os.environ.get('CONTINUUM_CONTAINER_BUILD') != '1' or not Path('/.dockerenv').exists():
        raise RuntimeError('This installer may only run inside the isolated Docker build')
    root = Path('/opt/continuum-build')
    recipe = json.loads((root / 'recipe.json').read_text())
    mode = sys.argv[1]
    if mode == 'environment':
        commands = []
        constraints = None
        if recipe['repo'] == 'matplotlib/matplotlib' and recipe['official_spec'].get('packages') == 'environment.yml':
            constraints = json.loads((root / 'matplotlib-compat.json').read_text())
        else:
            constraints = xarray_environment_constraints(recipe)
        if constraints:
            pins = [package for package in recipe['official_spec'].get('pip_packages', []) if '==' in package]
            (root / 'pip-constraints.txt').write_text('\n'.join(pins + constraints['pip']) + '\n')
            commands.append('export PIP_CONSTRAINT=/opt/continuum-build/pip-constraints.txt')
        for original in recipe['env_setup_commands']:
            command = original
            if 'name: testbed\n' in command and 'dependencies:\n' in command:
                if constraints:
                    command = constrain_environment(command, recipe, constraints)
                command = command.replace('dependencies:\n', 'dependencies:\n  - python=' + recipe['python'] + '\n', 1)
            commands.append(command)
        commands.extend([
            'python -m pip check',
            'python -m pip freeze > /opt/continuum-build/dependency-pip-freeze.txt',
            'conda list -n testbed --explicit > /opt/continuum-build/conda-explicit.txt',
            'conda clean --all --yes',
            'rm -rf /root/.cache/pip /var/lib/apt/lists/*',
        ])
    elif mode == 'task':
        repository = recipe['repo']
        commit = recipe['base_commit']
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository) or not re.fullmatch(r'[0-9a-f]{40}', commit):
            raise RuntimeError('Invalid source coordinates')
        commands = ['source /opt/miniconda3/etc/profile.d/conda.sh', 'conda activate testbed']
        source_metadata = root / 'source-cache/source.json'
        if source_metadata.exists():
            metadata = json.loads(source_metadata.read_text())
            archive = root / 'source-cache/repository.tar.gz'
            if (metadata.get('repo') != repository or metadata.get('base_commit') != commit
                    or hashlib.sha256(archive.read_bytes()).hexdigest() != metadata.get('sha256')):
                raise RuntimeError('Prepared source provenance mismatch')
            commands.extend([
                'mkdir -p /testbed/.git',
                'tar --extract --gzip --no-same-owner --file=/opt/continuum-build/source-cache/repository.tar.gz --directory=/testbed/.git',
                'git --git-dir=/testbed/.git config core.bare false',
                'git -C /testbed remote add origin ' + shlex.quote('https://github.com/' + repository + '.git'),
                'git -C /testbed cat-file -e ' + commit + '^{commit}',
            ])
        else:
            commands.extend([
            'git init /testbed',
            'git -C /testbed remote add origin ' + shlex.quote('https://github.com/' + repository + '.git'),
            'for attempt in 1 2 3; do '
            'git -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60 '
            '-C /testbed fetch --depth=1 origin ' + commit + ' && break; '
            'test "$attempt" -lt 3 || exit 1; sleep 2; done',
            ])
        commands.extend([
            'git -C /testbed checkout --detach ' + commit,
            'git -C /testbed remote remove origin', 'cd /testbed',
            *recipe['official_spec'].get('pre_install', []),
            *task_compatibility_commands(recipe),
        ])
        if recipe['official_spec'].get('install'):
            commands.append(recipe['official_spec']['install'])
        commands.extend([
            'python -m pip check',
            'python -m pip freeze > /opt/continuum-build/pip-freeze.txt',
            'conda list -n testbed --explicit > /opt/continuum-build/conda-explicit.txt',
            'dpkg-query -W > /opt/continuum-build/dpkg-versions.txt',
            'git diff HEAD --binary > /opt/continuum-build/setup-diff.patch',
            'git status --porcelain > /opt/continuum-build/setup-status.txt',
            'conda clean --all --yes',
            'rm -rf /root/.cache/pip /var/lib/apt/lists/*',
        ])
    else:
        raise RuntimeError('Unsupported build mode')
    (root / (mode + '-commands.json')).write_text(json.dumps(commands, indent=2) + '\n')
    run_shell(commands)


if __name__ == '__main__':
    main()
