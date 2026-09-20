import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import collect
from core import SafetyStop


def test_math_threads_follow_container_cpu_allocation(monkeypatch):
    observed = {}

    def capture(environment, **kwargs):
        observed.update(kwargs)

    monkeypatch.setattr(collect.DockerEnvironment, '__init__', capture)
    collect.TimedDocker(None, {'container_cpus': 4}, env={'PAGER': 'cat', 'OPENBLAS_NUM_THREADS': '128'})
    assert observed['env']['PAGER'] == 'cat'
    for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        assert observed['env'][name] == '4'


def test_container_lifetime_is_unbounded_without_relaxing_isolation(monkeypatch):
    environment = collect.TimedDocker.__new__(collect.TimedDocker)
    environment.config = SimpleNamespace(image='sha256:test-image')
    environment.run_config = json.loads((Path(__file__).parent / 'run_config.json').read_text())
    environment.container_id = None
    environment.donor = None
    commands = []

    def capture(command, **kwargs):
        commands.append(command)
        raise SafetyStop('Captured before execution')

    monkeypatch.setattr(collect, 'checked', capture)
    with pytest.raises(SafetyStop, match='Captured before execution'):
        environment._start_container()
    command = commands[0]
    assert command[-3:] == ['/bin/sleep', 'sha256:test-image', 'infinity']
    assert command[command.index('--network') + 1] == 'none'
    assert command[command.index('--cap-drop') + 1] == 'ALL'
    assert '--read-only' in command and '--memory' in command and '--cpus' in command
    assert '--mount' not in command and '-v' not in command and '--privileged' not in command
