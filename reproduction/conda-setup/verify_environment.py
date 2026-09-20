import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path


base = Path("/export/home/ext.luohaowen1/continuum")
setup = base / "reproduction/conda-setup"
checks = []
packages = ["torch", "torchaudio", "torchvision", "xformers", "transformers", "tokenizers", "numpy", "vllm", "lmcache", "mini-swe-agent", "litellm", "datasets", "matplotlib"]
versions = {package: importlib.metadata.version(package) for package in packages}


def check(name, operation):
    try:
        detail = operation()
        checks.append({"name": name, "passed": True, "detail": detail})
    except Exception as error:
        checks.append({"name": name, "passed": False, "error": repr(error)})


def import_module(name):
    module = importlib.import_module(name)
    return str(getattr(module, "__file__", None))


def check_scheduler():
    from vllm.config import SchedulerConfig
    scheduler = SchedulerConfig(max_model_len=8192, is_encoder_decoder=False, policy="continuum")
    assert scheduler.policy == "continuum"
    return scheduler.policy


def check_fork_origin():
    import vllm
    origin = Path(vllm.__file__).resolve()
    assert origin.is_relative_to(base / "workspaces/continuum-316a587"), str(origin)
    return str(origin)


assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Run CPU-only verification with GPUs hidden"
assert Path(sys.prefix) == base / "envs/continuum"
import torch

check("torch_cpu_tensor", lambda: float(torch.tensor([1.0, 2.0]).sum()))
check("vllm_fork_import", check_fork_origin)
check("vllm_native_ops_load", lambda: import_module("vllm._C"))
check("vllm_moe_ops_load", lambda: import_module("vllm._moe_C"))
check("flash_attention_2_ops_load", lambda: import_module("vllm.vllm_flash_attn._vllm_fa2_C"))
check("flash_attention_3_ops_load", lambda: import_module("vllm.vllm_flash_attn._vllm_fa3_C"))
check("continuum_scheduler_config", check_scheduler)
check("lmcache_native_ops_load", lambda: import_module("lmcache.c_ops"))
check("mini_swe_agent_import", lambda: import_module("minisweagent"))
check("datasets_import", lambda: import_module("datasets"))
check("matplotlib_import", lambda: import_module("matplotlib"))
check("pip_check", lambda: subprocess.check_output([sys.executable, "-m", "pip", "check"], text=True).strip())
report = {
    "python_version": sys.version,
    "python_executable": sys.executable,
    "environment_prefix": sys.prefix,
    "versions": versions,
    "torch_cuda_runtime": torch.version.cuda,
    "cuda_initialized": torch.cuda.is_initialized(),
    "gpu_forward_executed": False,
    "model_weights_downloaded_or_loaded": False,
    "source_commit": (base / "workspaces/continuum-316a587/.source-commit").read_text().strip(),
    "original_repository_clean": not subprocess.check_output(["git", "-C", str(base / "vllm-continuum"), "status", "--porcelain"], text=True).strip(),
    "verification_scope": "CPU imports, native extension loading, scheduler config and dependency consistency only; GPU inference and end-to-end reproduction NOT verified",
    "checks": checks,
    "all_checks_passed": all(record["passed"] for record in checks),
}
(setup / "environment-verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
assert report["all_checks_passed"]
assert not report["cuda_initialized"]
assert report["original_repository_clean"]
