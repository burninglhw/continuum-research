import hashlib
import json
from types import SimpleNamespace

import pytest

import local_images
from core import Records, SafetyStop


@pytest.fixture
def image_data(tmp_path):
    task = {'instance_id': 'django__django-13413', 'base_commit': 'a' * 40}
    image_id = 'sha256:' + 'b' * 64
    report = tmp_path / 'validation.json'
    report.write_text(json.dumps({'validation_status': 'passed', 'image_id': image_id,
                                 'instance_id': task['instance_id'], 'head': task['base_commit']}))
    entry = {'validation_status': 'passed', 'owner': 'ext.luohaowen1',
             'base_commit': task['base_commit'], 'image_id': image_id,
             'reference': 'continuum-luohaowen/swe-django:test',
             'validation_report': str(report),
             'validation_report_sha256': hashlib.sha256(report.read_bytes()).hexdigest()}
    manifest = {'schema_version': 1, 'images': {task['instance_id']: entry}}
    details = {'Id': image_id, 'Size': 1234, 'Config': {'Labels': {
        'continuum.image.owner': 'ext.luohaowen1', 'continuum.image.task': task['instance_id'],
        'continuum.image.base_commit': task['base_commit']}}}
    return task, manifest, details


def test_missing_task_does_not_invoke_docker(image_data, monkeypatch):
    task, manifest, details = image_data
    manifest['images'].clear()
    monkeypatch.setattr(local_images.subprocess, 'run', lambda *args, **kwargs: pytest.fail('Docker must not run'))
    with pytest.raises(SafetyStop, match='No validated'):
        local_images.inspect_image(task, manifest)


@pytest.mark.parametrize('field,value', [('owner', 'someone-else'), ('validation_status', 'failed'),
                                        ('base_commit', 'c' * 40), ('image_id', 'mutable-tag')])
def test_provenance_mismatch_stops(image_data, monkeypatch, field, value):
    task, manifest, details = image_data
    manifest['images'][task['instance_id']][field] = value
    monkeypatch.setattr(local_images.subprocess, 'run', lambda *args, **kwargs: pytest.fail('Docker must not run'))
    with pytest.raises(SafetyStop, match='provenance mismatch'):
        local_images.inspect_image(task, manifest)


def test_modified_validation_report_stops(image_data):
    task, manifest, details = image_data
    manifest['images'][task['instance_id']]['validation_report_sha256'] = '0' * 64
    with pytest.raises(SafetyStop, match='evidence mismatch'):
        local_images.inspect_image(task, manifest)


def test_absent_image_never_pulls(image_data, monkeypatch):
    task, manifest, details = image_data
    calls = []

    def invoke(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1, stdout='')

    monkeypatch.setattr(local_images.subprocess, 'run', invoke)
    with pytest.raises(SafetyStop, match='automatic pulls are disabled'):
        local_images.inspect_image(task, manifest)
    assert calls == [['docker', 'image', 'inspect', details['Id']]]


def test_wrong_docker_owner_stops(image_data, monkeypatch):
    task, manifest, details = image_data
    details['Config']['Labels']['continuum.image.owner'] = 'someone-else'
    monkeypatch.setattr(local_images.subprocess, 'run',
                        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=json.dumps([details])))
    with pytest.raises(SafetyStop, match='identity mismatch'):
        local_images.inspect_image(task, manifest)


def test_collector_preserves_validated_shared_image(image_data, tmp_path, monkeypatch):
    import collect

    task, manifest, details = image_data
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(collect, 'disk_guard', lambda config: None)
    monkeypatch.setattr(local_images.subprocess, 'run',
                        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=json.dumps([details])))
    reference, image_id, created = collect.ensure_image(task, Records(tmp_path / 'records'),
                                                       {'local_image_manifest': str(path)})
    assert reference.startswith('continuum-luohaowen/')
    assert image_id == details['Id']
    assert created is False


def test_malformed_manifest_stops(tmp_path):
    path = tmp_path / 'manifest.json'
    path.write_text('{"schema_version": 2, "images": []}')
    with pytest.raises(SafetyStop, match='Invalid'):
        local_images.load_manifest(path)


def test_missing_image_stops_before_key_prompt_or_api(tmp_path, monkeypatch):
    import collect

    run = tmp_path / 'run'
    run.mkdir()
    model = 'claude-sonnet-4-6'
    config = {'api_base': 'https://nowcoding.ai/v1', 'requested_model': model,
              'budget_stop_cny': 270, 'budget_authorized_cny': 300, 'workers': 1}
    (tmp_path / 'run_config.json').write_text(json.dumps(config))
    (tmp_path / 'tasks-100.jsonl').write_text(json.dumps({'instance_id': 'missing', 'base_commit': 'a' * 40}) + '\n')
    (run / 'interface-smoke.json').write_text(json.dumps({'http_status': 200,
        'nonempty_content': True, 'requested_model': model}))
    manifest = tmp_path / 'images.json'
    manifest.write_text(json.dumps({'schema_version': 1, 'images': {}}))
    monkeypatch.setattr(collect, 'ROOT', tmp_path)
    monkeypatch.setattr(collect.sys, 'argv', ['collect.py', '--limit', '1', '--run-dir', str(run),
                                            '--local-images', str(manifest)])
    monkeypatch.setattr(collect.getpass, 'getpass', lambda *args: pytest.fail('No credential prompt permitted'))
    monkeypatch.setattr(collect.requests, 'post', lambda *args, **kwargs: pytest.fail('No API request permitted'))
    with pytest.raises(SafetyStop, match='No validated'):
        collect.main()
