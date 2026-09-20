import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

import collect
from core import Records, SafetyStop


@pytest.fixture
def environment(tmp_path, monkeypatch):
    runner = collect.TimedDocker.__new__(collect.TimedDocker)
    runner.container_id = 'test-container'
    runner.config = SimpleNamespace(env={})
    runner.run_config = json.loads((Path(__file__).parent / 'run_config.json').read_text())
    runner.records = Records(tmp_path)
    runner.phase = 'agent'
    runner.tool_events = []
    monkeypatch.setattr(collect, 'disk_guard', lambda settings: None)
    yield runner
    runner.container_id = None


@pytest.mark.parametrize('tool_command', [
    'pkill -9 -f runtests.py; python tests/runtests.py forms_tests --parallel 1',
    'printf "中文\\n"; printf "%s" "quoted input"',
])
def test_tool_command_is_stdin_not_timer_ancestor_argv(environment, monkeypatch, tool_command):
    def capture(command, *, stdin, stdout, stderr):
        assert command[:3] == ['docker', 'exec', '-i']
        assert stdin.read().decode('utf-8') == tool_command
        assert tool_command not in ' '.join(command)
        invocation = shlex.split(command[-1])
        assert len(invocation) == 6
        assert 'command=sys.stdin.read()' in invocation[3]
        assert 'stdin=subprocess.DEVNULL' in invocation[3]
        assert invocation[-2] == str(environment.run_config['tool_timeout_seconds'])
        stdout.write(('real tool output\n' + invocation[-1]
                      + json.dumps({'duration_ms': 12.5, 'returncode': -9})).encode())
        return SimpleNamespace(poll=lambda: 0)

    monkeypatch.setattr(collect.subprocess, 'Popen', capture)
    result = environment.execute(tool_command)
    assert result == {'output': 'real tool output', 'returncode': -9}
    assert environment.tool_events[0]['duration_ms'] == 12.5
    assert environment.tool_events[0]['command'] == tool_command


def test_missing_timer_still_stops_without_invented_duration(environment, monkeypatch):
    def capture(command, *, stdin, stdout, stderr):
        stdout.write(b'partial output without a timer')
        return SimpleNamespace(poll=lambda: 0)

    monkeypatch.setattr(collect.subprocess, 'Popen', capture)
    with pytest.raises(SafetyStop, match='Internal tool timer unavailable'):
        environment.execute('echo partial')
    record = json.loads((environment.records.root / 'tools.jsonl').read_text())
    assert record['duration_ms'] is None
    assert record['error'] == 'missing_internal_timer'
    assert environment.tool_events == []
