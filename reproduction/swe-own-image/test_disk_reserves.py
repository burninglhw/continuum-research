import json
from types import SimpleNamespace

import pytest

import pipeline_images
from core import SafetyStop


@pytest.mark.parametrize('enabled', [True, False])
def test_image_build_reserves_default_on_and_can_be_disabled(tmp_path, monkeypatch, enabled):
    monkeypatch.setattr(pipeline_images, 'ROOT', tmp_path)
    monkeypatch.setattr(pipeline_images.shutil, 'disk_usage', lambda path: SimpleNamespace(free=1))
    monkeypatch.setattr(pipeline_images.time, 'sleep', lambda duration: None)
    monkeypatch.setattr(pipeline_images.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=1))
    for filename in ['Dockerfile.generic-env', 'configure_image.py', 'activate-testbed.sh', 'matplotlib-compat.json']:
        (tmp_path / filename).write_text('test build source')
    commands = []

    def build(command, **kwargs):
        commands.append(command)
        results = iter([None, 0])
        return SimpleNamespace(poll=lambda: next(results), returncode=0)

    monkeypatch.setattr(pipeline_images.subprocess, 'Popen', build)
    monkeypatch.setattr(pipeline_images, 'checked', lambda command: json.dumps([{'Id': 'sha256:test', 'Size': 1}]))
    recipe = {'dependency_group': 'test-group', 'instance_id': 'sample', 'base_commit': 'abc'}
    if enabled:
        with pytest.raises(SafetyStop, match='Need 12 GiB'):
            pipeline_images.image_build(recipe, 'env', 'sha256:parent')
        assert commands == []
    else:
        result = pipeline_images.image_build(recipe, 'env', 'sha256:parent', disk_reserve_checks_enabled=False)
        assert len(commands) == 1
        assert result['disk_reserve_checks_enabled'] is False
        assert '--memory=4g' in commands[0]


@pytest.mark.parametrize('enabled', [True, False])
def test_build_timeout_defaults_on_and_can_be_disabled(tmp_path, monkeypatch, enabled):
    monkeypatch.setattr(pipeline_images, 'ROOT', tmp_path)
    monkeypatch.setattr(pipeline_images.shutil, 'disk_usage', lambda path: SimpleNamespace(free=20 * 2**30))
    monkeypatch.setattr(pipeline_images.time, 'sleep', lambda duration: None)
    clock = iter([0, 1900])
    monkeypatch.setattr(pipeline_images.time, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(pipeline_images.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=1))
    for filename in ['Dockerfile.generic-env', 'configure_image.py', 'activate-testbed.sh', 'matplotlib-compat.json']:
        (tmp_path / filename).write_text('test build source')
    results = iter([None, 0])
    process = SimpleNamespace(poll=lambda: next(results), returncode=0,
                              terminate=lambda: None, wait=lambda **kwargs: None)
    monkeypatch.setattr(pipeline_images.subprocess, 'Popen', lambda *args, **kwargs: process)
    monkeypatch.setattr(pipeline_images, 'checked', lambda command: json.dumps([{'Id': 'sha256:test', 'Size': 1}]))
    recipe = {'dependency_group': 'test-group', 'instance_id': 'sample', 'base_commit': 'abc'}
    if enabled:
        with pytest.raises(SafetyStop, match='Build exceeded 30 minutes'):
            pipeline_images.image_build(recipe, 'env', 'sha256:parent')
    else:
        result = pipeline_images.image_build(recipe, 'env', 'sha256:parent', build_timeout_enabled=False)
        assert result['build_seconds'] == 1900
        assert result['build_timeout_enabled'] is False
