import argparse
import concurrent.futures
import hashlib
import json
from collections import Counter
from pathlib import Path

import requests
from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
from swebench.harness.test_spec.python import make_env_script_list_py

ROOT = Path(__file__).resolve().parent
TRACE_ROOT = ROOT.parent / 'swe-traces'
ORIGINAL_REQUEST = requests.sessions.Session.request


def bounded_request(session, method, url, **kwargs):
    kwargs.setdefault('timeout', (15, 40))
    return ORIGINAL_REQUEST(session, method, url, **kwargs)


requests.sessions.Session.request = bounded_request


def describe(task):
    spec = MAP_REPO_VERSION_TO_SPECS[task['repo']][task['version']]
    record = {name: task[name] for name in ['instance_id', 'repo', 'base_commit', 'version']}
    record.update(python=spec['python'], official_spec=spec)
    try:
        commands = make_env_script_list_py(task, spec, 'testbed')
        signature = {'repo': task['repo'], 'python': spec['python'],
                     'env_setup_commands': commands}
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        record.update(dependency_group=digest, env_setup_commands=commands,
                      status='recipe_audited_not_built')
    except Exception as error:
        record.update(status='unresolved', error=str(error))
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'provenance/dependency-groups.json')
    arguments = parser.parse_args()
    tasks = [json.loads(line) for line in (TRACE_ROOT / 'tasks-100.jsonl').read_text().splitlines()]
    output = arguments.output
    if output.exists():
        raise RuntimeError('Audit already exists; preserve provenance rather than overwrite')
    previous = {}
    if arguments.previous:
        prior = json.loads(arguments.previous.read_text())
        if prior['source_dataset_sha256'] != hashlib.sha256((TRACE_ROOT / 'tasks-100.jsonl').read_bytes()).hexdigest():
            raise RuntimeError('Previous audit used a different task selection')
        previous = {record['instance_id']: record for record in prior['tasks']
                    if record['status'] != 'unresolved'}
    remaining = [task for task in tasks if task['instance_id'] not in previous]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        previous.update((record['instance_id'], record) for record in executor.map(describe, remaining))
    records = [previous[task['instance_id']] for task in tasks]
    groups = {}
    for record in records:
        if 'dependency_group' in record:
            groups.setdefault(record['dependency_group'], []).append(record['instance_id'])
    result = {'task_count': len(tasks), 'candidate_dependency_group_count': len(groups),
              'unresolved_count': sum(record['status'] == 'unresolved' for record in records),
              'python_versions': dict(Counter(record['python'] for record in records)),
              'group_members': groups, 'tasks': records,
              'method': 'same repo, Python and exact official environment setup commands',
              'limitation': 'Candidate shared dependency layers only. Per-task repo/pre-install/build '
                            'steps and tests remain necessary; unpinned packages must be locked after building.',
              'source_dataset_sha256': hashlib.sha256((TRACE_ROOT / 'tasks-100.jsonl').read_bytes()).hexdigest()}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ['task_count', 'candidate_dependency_group_count',
                                                 'unresolved_count', 'python_versions']}, indent=2))


if __name__ == '__main__':
    main()
