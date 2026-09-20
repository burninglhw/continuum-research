import hashlib
import json
from types import SimpleNamespace

import pytest

import pipeline_images
from core import SafetyStop


def test_missing_source_cache_preserves_network_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_images, 'ROOT', tmp_path)
    context = tmp_path / 'context'
    context.mkdir()
    pipeline_images.stage_source_cache(context, {'repo': 'sample/repo', 'base_commit': 'abc'})
    assert list((context / 'source-cache').iterdir()) == []


def test_source_cache_requires_matching_identity_and_checksum(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_images, 'ROOT', tmp_path)
    source = tmp_path / 'source-cache'
    source.mkdir()
    payload = b'known repository archive'
    (source / 'abc.tar.gz').write_bytes(payload)
    metadata = {'repo': 'sample/repo', 'base_commit': 'abc', 'sha256': hashlib.sha256(payload).hexdigest()}
    (source / 'abc.json').write_text(json.dumps(metadata))
    context = tmp_path / 'context'
    context.mkdir()
    pipeline_images.stage_source_cache(context, {'repo': 'sample/repo', 'base_commit': 'abc'})
    assert (context / 'source-cache/repository.tar.gz').read_bytes() == payload
    with pytest.raises(SafetyStop, match='identity or checksum'):
        pipeline_images.stage_source_cache(context, {'repo': 'different/repo', 'base_commit': 'abc'})
    (source / 'abc.tar.gz').write_bytes(b'corrupted')
    with pytest.raises(SafetyStop, match='identity or checksum'):
        pipeline_images.stage_source_cache(context, {'repo': 'sample/repo', 'base_commit': 'abc'})


def test_release_rejects_runs_outside_this_collector(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_images, 'TRACE_ROOT', tmp_path / 'collector')
    with pytest.raises(SafetyStop, match='Run directory'):
        pipeline_images.release_completed('sample', None, tmp_path / 'someone-else')


def test_nonautomatic_images_are_never_removed(monkeypatch):
    monkeypatch.setattr(pipeline_images.subprocess, 'run', lambda *args, **kwargs: pytest.fail('Docker must not be called'))
    result = pipeline_images.release_own_reference('another-user/model:latest', 'sha256:123')
    assert result['released'] is False


def test_mismatched_image_ownership_stops_before_removal(monkeypatch):
    observed = []

    def run(command, **kwargs):
        observed.append(command)
        return SimpleNamespace(returncode=0, stdout=json.dumps([{
            'Id': 'sha256:123', 'Config': {'Labels': {
                'continuum.image.owner': 'someone-else', 'continuum.image.managed': 'swe100'}}}]))

    monkeypatch.setattr(pipeline_images.subprocess, 'run', run)
    with pytest.raises(SafetyStop, match='mismatched ownership'):
        pipeline_images.release_own_reference('continuum-luohaowen/swe-auto-task:sample', 'sha256:123')
    assert len(observed) == 1 and observed[0][:3] == ['docker', 'image', 'inspect']


def test_release_uses_explicit_new_run_completion(tmp_path, monkeypatch):
    trace_root = tmp_path / 'collector'
    run = trace_root / 'runs/new-protocol'
    case = run / 'instances/sample'
    case.mkdir(parents=True)
    (case / 'metrics.json').write_text(json.dumps({'collection_completed': True}))
    provenance = tmp_path / 'images/provenance'
    provenance.mkdir(parents=True)
    reference = 'continuum-luohaowen/swe-auto-task:sample'
    (provenance / 'validated-images.json').write_text(json.dumps({
        'schema_version': 1, 'images': {'sample': {'reference': reference, 'image_id': 'sha256:123'}}}))
    monkeypatch.setattr(pipeline_images, 'TRACE_ROOT', trace_root)
    monkeypatch.setattr(pipeline_images, 'ROOT', tmp_path / 'images')
    observed = []

    def release(reference, image_id):
        observed.append((reference, image_id))
        return {'released': True}

    def archive(instance_id, run):
        result = {'image_capture_status': 'captured', 'recorded_image': {'validation': {
            'reference': reference, 'image_id': 'sha256:123'}}}
        (run / 'instances' / instance_id / 'environment.json').write_text(json.dumps(result))
        observed.append('archived')
        return result

    monkeypatch.setattr(pipeline_images, 'archive_task_environment', archive)
    monkeypatch.setattr(pipeline_images, 'release_own_reference', release)
    pipeline_images.release_completed('sample', None, run)
    assert observed == ['archived', (reference, 'sha256:123')]


@pytest.fixture
def environment_case(tmp_path, monkeypatch):
    trace_root = tmp_path / 'collector'
    run = trace_root / 'runs/current'
    case = run / 'instances/sample'
    case.mkdir(parents=True)
    image_root = tmp_path / 'images'
    provenance = image_root / 'provenance'
    provenance.mkdir(parents=True)
    task = {'instance_id': 'sample', 'repo': 'sample/repo', 'base_commit': 'abc'}
    report = {'image_id': 'sha256:123', 'instance_id': 'sample', 'head': 'abc',
              'validation_status': 'passed', 'python_version': 'Python 3.11.0',
              'pip_freeze': 'example==1.0', 'conda_explicit': '@EXPLICIT'}
    report_path = provenance / 'validation.json'
    report_path.write_text(json.dumps(report))
    entry = {'image_id': 'sha256:123', 'reference': 'continuum-luohaowen/swe-auto-task:sample',
             'validation_report': str(report_path),
             'validation_report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest()}
    (case / 'image.json').write_text(json.dumps({'image_id': 'sha256:123', 'validation': entry}))
    (case / 'task.json').write_text(json.dumps(task))
    (case / 'metrics.json').write_text(json.dumps({'collection_completed': True}))
    (provenance / 'dependency-groups-retry.json').write_text(json.dumps({'tasks': [task]}))
    monkeypatch.setattr(pipeline_images, 'TRACE_ROOT', trace_root)
    monkeypatch.setattr(pipeline_images, 'ROOT', image_root)
    return run, case, report_path


def test_absent_image_preserves_existing_validation_and_marks_gaps(environment_case, monkeypatch):
    run, case, report_path = environment_case
    monkeypatch.setattr(pipeline_images.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=1))
    archive = pipeline_images.archive_task_environment('sample', run)
    assert archive['image_capture_status'] == 'image_already_absent'
    assert archive['missing_evidence']
    assert archive['validation_report']['pip_freeze'] == 'example==1.0'
    assert json.loads((case / 'environment.json').read_text()) == archive


def test_archive_checksum_failure_prevents_image_removal(environment_case, monkeypatch):
    run, case, report_path = environment_case
    report_path.write_text('{}')
    monkeypatch.setattr(pipeline_images.subprocess, 'run', lambda *args, **kwargs: pytest.fail('No Docker call allowed'))
    with pytest.raises(SafetyStop, match='checksum'):
        pipeline_images.release_completed('sample', None, run)


def test_live_image_is_captured_without_network_or_host_mounts(environment_case, monkeypatch):
    run, case, report_path = environment_case
    details = {'Id': 'sha256:123', 'Config': {'Labels': {
        'continuum.image.owner': 'ext.luohaowen1', 'continuum.image.task': 'sample',
        'continuum.image.base_commit': 'abc'}}}
    monkeypatch.setattr(pipeline_images.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=json.dumps([details])))
    files = {'/etc/os-release': 'NAME=Ubuntu', '/opt/continuum-build/pip-freeze.txt': 'example==1.0',
             '/opt/continuum-build/conda-explicit.txt': '@EXPLICIT',
             '/opt/continuum-build/dpkg-versions.txt': 'bash 1.0'}
    commands = []

    def checked(command, **kwargs):
        commands.append(command)
        return json.dumps(files)

    monkeypatch.setattr(pipeline_images, 'checked', checked)
    archive = pipeline_images.archive_task_environment('sample', run)
    assert archive['image_capture_status'] == 'captured'
    assert archive['image_files'] == files
    assert commands[0][commands[0].index('--network') + 1] == 'none'
    assert '--read-only' in commands[0] and '--mount' not in commands[0] and '-v' not in commands[0]
    assert archive['image_file_sha256']['/etc/os-release'] == hashlib.sha256(b'NAME=Ubuntu').hexdigest()


def test_missing_package_inventory_prevents_release(environment_case, monkeypatch):
    run, case, report_path = environment_case
    details = {'Id': 'sha256:123', 'Config': {'Labels': {
        'continuum.image.owner': 'ext.luohaowen1', 'continuum.image.task': 'sample',
        'continuum.image.base_commit': 'abc'}}}
    monkeypatch.setattr(pipeline_images.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=json.dumps([details])))
    monkeypatch.setattr(pipeline_images, 'checked', lambda *args, **kwargs: '{}')
    monkeypatch.setattr(pipeline_images, 'release_own_reference', lambda *args, **kwargs: pytest.fail('No deletion allowed'))
    with pytest.raises(SafetyStop, match='Required environment records missing'):
        pipeline_images.release_completed('sample', None, run)
    assert not (case / 'environment.json').exists()
