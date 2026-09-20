import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as parquet


def main():
    base = Path('/export/home/ext.luohaowen1/continuum')
    output = base / 'reproduction/swe-traces'
    dataset = base / 'datasets/swe_bench/data/test-00000-of-00001.parquet'
    fields = ['instance_id', 'repo', 'base_commit', 'problem_statement', 'version', 'environment_setup_commit']
    rows = sorted(parquet.read_table(dataset, columns=fields).to_pylist(), key=lambda row: row['instance_id'])
    assert len(rows) == 2294
    selected = random.Random(42).sample(rows, 100)
    assert len({row['instance_id'] for row in selected}) == 100
    serialized = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in selected)
    sample = output / 'tasks-100.jsonl'
    if sample.exists() and sample.read_text() != serialized:
        raise SystemExit('Refusing to overwrite a different task selection')
    sample.write_text(serialized)
    manifest = {
        'dataset': 'princeton-nlp/SWE-bench',
        'split': 'test',
        'revision': 'e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6',
        'source_rows': len(rows),
        'source_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
        'selection': 'random.Random(42).sample(sorted(rows, key=instance_id), 100)',
        'sample_sha256': hashlib.sha256(serialized.encode()).hexdigest(),
        'repository_counts': dict(Counter(row['repo'] for row in selected)),
        'gold_patch_and_test_patch_excluded': True,
        'paper_sample_ids_available': False,
        'agent_source_commit': '316a58794a6ff86b216e579b74fd56ed0c5a911f',
        'paper_targets': {
            'turns': {'mean': 10.9, 'std': 2.1},
            'tool_time_ms': {'mean': 925, 'std': 3550},
            'tokens_per_program': {'mean': 70126, 'std': 19732},
            'token_aggregation_definition': 'not specified precisely in paper; report multiple definitions',
        },
    }
    (output / 'dataset-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({'count': len(selected), 'first_three': [row['instance_id'] for row in selected[:3]],
                      'repository_counts': manifest['repository_counts']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
