import json
from types import SimpleNamespace

import pytest

import pipeline_images


@pytest.mark.parametrize('repository,mode,memory_gib', [
    ('pydata/xarray', 'env', 16),
    ('pydata/xarray', 'task', 4),
    ('pytest-dev/pytest', 'env', 4),
    ('django/django', 'task', 4),
    (None, 'env', 4),
])
def test_build_memory_scope_and_provenance(tmp_path, monkeypatch, repository, mode, memory_gib):
    monkeypatch.setattr(pipeline_images, 'ROOT', tmp_path)
    monkeypatch.setattr(pipeline_images.shutil, 'disk_usage', lambda path: SimpleNamespace(free=20 * 2**30))
    monkeypatch.setattr(pipeline_images.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=1))
    monkeypatch.setattr(pipeline_images, 'stage_source_cache', lambda *args: None)
    for filename in ['Dockerfile.generic-' + mode, 'configure_image.py', 'activate-testbed.sh', 'matplotlib-compat.json']:
        (tmp_path / filename).write_text('test build source')
    commands = []

    def build(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(poll=lambda: 0, returncode=0)

    monkeypatch.setattr(pipeline_images.subprocess, 'Popen', build)
    monkeypatch.setattr(pipeline_images, 'checked', lambda command: json.dumps([{'Id': 'sha256:test', 'Size': 1}]))
    recipe = {'dependency_group': 'test-group', 'instance_id': 'sample', 'base_commit': 'abc'}
    if repository:
        recipe['repo'] = repository
    result = pipeline_images.image_build(recipe, mode, 'sha256:parent')
    assert f'--memory={memory_gib}g' in commands[0]
    assert f'--memory-swap={memory_gib}g' in commands[0]
    assert '--cpu-quota=200000' in commands[0]
    assert result['build_resources'] == {'memory_gib': memory_gib, 'memory_swap_gib': memory_gib, 'cpus': 2}
    identity = 'test-group' if mode == 'env' else 'sample'
    invocation = json.loads((tmp_path / 'provenance/automatic' / (mode + '-' + identity) / 'build-invocation.json').read_text())
    assert invocation['command'] == commands[0]
    assert invocation['build_resources'] == result['build_resources']
