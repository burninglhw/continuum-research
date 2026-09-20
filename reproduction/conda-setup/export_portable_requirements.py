import datetime
import importlib.metadata
import json
import subprocess
from pathlib import Path


base = Path("/export/home/ext.luohaowen1/continuum")
setup = base / "reproduction/conda-setup"
packages = json.loads(subprocess.check_output([str(base / "envs/continuum/bin/python"), "-m", "pip", "list", "--format=json"], text=True))
requirements = []
for package in packages:
    name = package["name"].lower().replace("_", "-")
    if name in {"vllm", "mini-swe-agent"}:
        continue
    requirements.append(f"{name}=={package['version']}")
requirements.append("vllm==0.10.2")
(setup / "requirements-pypi-locked.txt").write_text("\n".join(sorted(requirements)) + "\n")
report = {
    "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "status": "installed_and_cpu_validated",
    "gpu_experiment_started": False,
    "system_python_or_cuda_changed": False,
    "original_repository_modified": False,
    "environment_prefix": str(base / "envs/continuum"),
    "activation_script": str(setup / "activate_continuum.sh"),
    "source_copy": str(base / "workspaces/continuum-316a587"),
    "miniforge_release": "26.7.2-0",
    "miniforge_installer_sha256": "281b0ac7d550802efc81af633225a5e6116d29ae72f3ab4eae7168c3931a4c05",
    "paper_sha256": "d78aa8f0415311063158ebdd383f8ebde25414af5c28fdcf400968e7bb606ee2",
    "paper_explicit_versions": {"vllm_baseline": "0.10.2", "lmcache": "0.3.7"},
    "portable_requirements_note": "Installs the recorded PyPI runtime with vanilla vLLM 0.10.2 first. Then run install_fork.sh to install the fixed author source and bundled mini-swe-agent. This file alone does not install Continuum.",
    "environment_yml_note": "State snapshot; not a standalone one-command rebuild because the custom local vLLM version is not published on PyPI.",
    "cli_checks": {"vllm_version": (setup / "vllm-version.txt").read_text().splitlines()[-1], "mini_swebench_help": "Usage: mini-extra swebench" in (setup / "mini-swebench-help.txt").read_text()},
    "installed_package_count": len(packages),
}
(setup / "setup-summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
