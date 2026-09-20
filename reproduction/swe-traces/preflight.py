import json
from pathlib import Path

from collect import ROOT, TimedDocker, checked
from core import Records


class TimerPreflight(TimedDocker):
    def _start_container(self):
        self.container_id = checked([
            'docker', 'run', '-d', '--rm', '--pull=never',
            '--label', 'continuum.trace.owner=ext.luohaowen1',
            '--network', 'none', '--read-only', '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges', '--cpus', '1',
            '--memory', '1g', '--memory-swap', '1g', '--pids-limit', '64',
            '--tmpfs', '/testbed:rw,nosuid,nodev,size=16m',
            '--tmpfs', '/tmp:rw,nosuid,nodev,size=32m',
            '--env', 'NVIDIA_VISIBLE_DEVICES=void', '--entrypoint', '/bin/sleep',
            self.config.image, '120',
        ])
        state = json.loads(checked(['docker', 'inspect', self.container_id]))[0]
        assert state['HostConfig']['NetworkMode'] == 'none'
        assert state['HostConfig']['ReadonlyRootfs']
        assert not state['HostConfig'].get('Binds')
        assert not state['HostConfig'].get('DeviceRequests')
        self.records.write('isolation.json', {
            'network': 'none', 'read_only': True, 'host_binds': False,
            'gpu_requests': False, 'purpose': 'timer validation only, NOT a SWE-bench trace',
        })


def main():
    config = json.loads((ROOT / 'run_config.json').read_text())
    records = Records(ROOT / 'preflight')
    image_id = checked(['docker', 'image', 'inspect', 'nvcr.io/nvidia/pytorch:24.04-py3', '--format', '{{.Id}}'])
    environment = TimerPreflight(records, config, image=image_id, cwd='/testbed')
    try:
        first = environment.execute("printf 'timer-preflight-ok\\n'")
        assert first['returncode'] == 0
        assert first['output'] == 'timer-preflight-ok\n'
        second = environment.execute("sleep 0.1; printf 'timer-delay-ok\\n'")
        assert second['output'] == 'timer-delay-ok\n'
        events = [json.loads(line) for line in (records.root / 'tools.jsonl').read_text().splitlines()]
        assert events[-1]['duration_ms'] >= 90
        assert events[-1]['docker_wall_ms'] >= events[-1]['duration_ms']
        records.write('result.json', {'timer_preflight': 'passed', 'actual_swebench_environment_validated': False,
                                     'agent_model_called': False, 'gpu_used': False,
                                     'internal_duration_ms': events[-1]['duration_ms'],
                                     'docker_wall_ms': events[-1]['docker_wall_ms']})
        print('TIMER_PREFLIGHT_PASSED')
    finally:
        environment.cleanup()


if __name__ == '__main__':
    main()
