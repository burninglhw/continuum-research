import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def command(arguments):
    result = subprocess.run(arguments, capture_output=True, text=True, check=False)
    return {"command": arguments, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fork", type=Path, required=True)
    parser.add_argument("--installed-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    packages = sorted({(entry.metadata["Name"], entry.version)
                       for entry in importlib.metadata.distributions()})
    result = {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": packages,
        "nvidia_smi": command(["nvidia-smi", "-q"]),
        "cpu": command(["lscpu"]),
        "disk": command(["df", "-h", str(args.root)]),
        "fork_commit": command(["git", "-C", str(args.fork), "rev-parse", "HEAD"]),
        "fork_status": command(["git", "-C", str(args.fork), "status", "--porcelain"]),
        "installed_commit": command(["git", "-C", str(args.installed_source), "rev-parse", "HEAD"]),
        "installed_status": command(["git", "-C", str(args.installed_source), "status", "--porcelain"]),
        "runtime_sha256": {path.name: digest(path) for path in sorted(args.root.glob("*.py"))},
        "input_sha256": {str(path.relative_to(args.root)): digest(path) for path in [
            args.root / "data/replay.pkl.gz", args.root / "data/manifest.json",
            args.root / "data/lengths.json", args.root / "profile-v2/curve.json"]},
        "model": {"path": str(args.model), "files": []},
        "installed_vllm_python_sha256": {},
    }
    for path in sorted(args.model.iterdir()):
        if path.is_file() and path.suffix in {".json", ".safetensors"}:
            result["model"]["files"].append({"file": path.name, "bytes": path.stat().st_size,
                                           "sha256": digest(path)})
    for path in sorted((args.installed_source / "vllm").rglob("*.py")):
        result["installed_vllm_python_sha256"][str(path.relative_to(args.installed_source))] = digest(path)
    (args.output / "environment.json").write_text(json.dumps(result, indent=2))
    (args.output / "packages.txt").write_text("".join(f"{name}=={version}\n" for name, version in packages))
    print(json.dumps({"output": str(args.output), "model_files": len(result["model"]["files"]),
                      "python_source_files": len(result["installed_vllm_python_sha256"])}), flush=True)


if __name__ == "__main__":
    main()
