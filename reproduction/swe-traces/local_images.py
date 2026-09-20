import hashlib
import json
import re
import subprocess
from pathlib import Path

from core import SafetyStop


def load_manifest(path):
    try:
        manifest = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise SafetyStop('Self-built image manifest unavailable: ' + str(error)) from None
    if manifest.get('schema_version') != 1 or not isinstance(manifest.get('images'), dict):
        raise SafetyStop('Invalid self-built image manifest')
    return manifest


def inspect_image(task, manifest):
    entry = manifest['images'].get(task['instance_id'])
    if not isinstance(entry, dict):
        raise SafetyStop('No validated self-built image for ' + task['instance_id'] + '; no pull or API request')
    if (entry.get('validation_status') != 'passed' or entry.get('owner') != 'ext.luohaowen1'
            or entry.get('base_commit') != task['base_commit']
            or not str(entry.get('reference', '')).startswith('continuum-luohaowen/')
            or not re.fullmatch(r'sha256:[0-9a-f]{64}', str(entry.get('image_id', '')))):
        raise SafetyStop('Self-built image provenance mismatch')
    try:
        report = Path(entry['validation_report']).read_bytes()
        evidence = json.loads(report)
    except (KeyError, OSError, ValueError) as error:
        raise SafetyStop('Image validation evidence unavailable: ' + str(error)) from None
    if (hashlib.sha256(report).hexdigest() != entry.get('validation_report_sha256')
            or evidence.get('validation_status') != 'passed'
            or evidence.get('image_id') != entry['image_id']
            or evidence.get('instance_id') != task['instance_id']
            or evidence.get('head') != task['base_commit']):
        raise SafetyStop('Image validation evidence mismatch')
    result = subprocess.run(['docker', 'image', 'inspect', entry['image_id']],
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise SafetyStop('Validated local image is absent; automatic pulls are disabled')
    details = json.loads(result.stdout)[0]
    labels = details.get('Config', {}).get('Labels', {}) or {}
    if (details.get('Id') != entry['image_id']
            or labels.get('continuum.image.owner') != 'ext.luohaowen1'
            or labels.get('continuum.image.task') != task['instance_id']
            or labels.get('continuum.image.base_commit') != task['base_commit']):
        raise SafetyStop('Local Docker image identity mismatch')
    return entry, details
