import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

from core import SafetyStop


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def messages_digest(path):
    messages = json.loads(path.read_text())['messages']
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def save_checkpoint(environment, records, task, image_id):
    temporary = records.root / 'workspace-checkpoint.tar.gz.partial'
    final = records.root / 'workspace-checkpoint.tar.gz'
    process = None
    disk_reserve_checks_enabled = environment.run_config.get('disk_reserve_checks_enabled', True)
    try:
        if disk_reserve_checks_enabled and shutil.disk_usage(records.root).free < 9 * 1024 ** 3:
            raise SafetyStop('Insufficient free disk for a bounded checkpoint')
        with temporary.open('wb') as output, (records.root / 'checkpoint-export.log').open('w') as errors:
            process = subprocess.Popen(['docker', 'exec', environment.container_id,
                'tar', '--create', '--gzip', '--file=-', '--directory=/testbed', '.'],
                stdout=output, stderr=errors)
            started = time.monotonic()
            while process.poll() is None:
                if (temporary.stat().st_size > 2 * 1024 ** 3
                        or (disk_reserve_checks_enabled and shutil.disk_usage(records.root).free < 6 * 1024 ** 3)):
                    raise SafetyStop('Checkpoint disk limit reached')
                if time.monotonic() - started > 120:
                    raise SafetyStop('Checkpoint export timed out')
                time.sleep(0.2)
            if process.returncode:
                raise SafetyStop('Checkpoint export failed')
        temporary.replace(final)
        records.write('checkpoint.json', {'status': 'ready', 'instance_id': task['instance_id'],
            'image_id': image_id, 'archive': final.name, 'sha256': digest(final),
            'trajectory_sha256': digest(records.root / 'trajectory.json'),
            'messages_sha256': messages_digest(records.root / 'trajectory.json'),
            'size_bytes': final.stat().st_size})
    except Exception as error:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if temporary.exists():
            temporary.unlink()
        records.write('checkpoint-error.json', {'error': str(error), 'resumption_ready': False})


def restore_checkpoint(environment, records, task, image_id, previous_tools):
    marker = records.root / 'checkpoint.json'
    if marker.exists():
        checkpoint = json.loads(marker.read_text())
        if checkpoint['status'] != 'ready' or checkpoint['instance_id'] != task['instance_id'] or checkpoint['image_id'] != image_id:
            raise SafetyStop('Checkpoint provenance mismatch')
        archive = records.root / 'workspace-checkpoint.tar.gz'
        if digest(archive) != checkpoint['sha256'] or messages_digest(records.root / 'trajectory.json') != checkpoint['messages_sha256']:
            raise SafetyStop('Checkpoint integrity mismatch')
        subprocess.run(['docker', 'exec', environment.container_id, 'find', '/testbed',
                        '-mindepth', '1', '-maxdepth', '1', '-exec', 'rm', '-rf', '--', '{}', '+'],
                       check=True, capture_output=True, timeout=90)
        with archive.open('rb') as source:
            subprocess.run(['docker', 'exec', '-i', environment.container_id, 'tar', '--extract',
                            '--gzip', '--file=-', '--directory=/testbed', '--no-same-owner'],
                           stdin=source, check=True, capture_output=True, timeout=180)
        basis = 'integrity_checked_workspace_archive'
    elif previous_tools:
        review = json.loads((Path(__file__).parent / 'readonly-recovery-review.json').read_text())
        if (review['instance_id'] != task['instance_id'] or review['image_id'] != image_id
                or [event['command'] for event in previous_tools] != review['reviewed_commands']
                or any(event['returncode'] != 0 for event in previous_tools)):
            raise SafetyStop('No valid workspace checkpoint or reviewed read-only history')
        basis = 'explicitly_reviewed_readonly_history_same_image'
    else:
        basis = 'no_previous_tool_execution'
    records.event('resume-events.jsonl', {'instance_id': task['instance_id'], 'basis': basis,
        'previous_tool_commands_reexecuted': False, 'previous_outputs_and_timings_preserved': True})
