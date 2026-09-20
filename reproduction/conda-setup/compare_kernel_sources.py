import hashlib
import json
import subprocess
import tarfile
from pathlib import Path


base = Path("/export/home/ext.luohaowen1/continuum")
setup = base / "reproduction/conda-setup"
repo = base / "vllm-continuum"
matched = []
different = []
missing = []
bundled_only = []
archive_paths = set()
prefixes = ("csrc/", "cmake/", "vllm/vllm_flash_attn/")
with tarfile.open(setup / "vllm-0.10.2.tar.gz", "r:gz") as archive:
    for member in archive.getmembers():
        if not member.isfile():
            continue
        relative = member.name.split("/", 1)[1]
        archive_paths.add(relative)
        if not (relative.startswith(prefixes) or relative in {"CMakeLists.txt", "vllm/_custom_ops.py"}):
            continue
        target = repo / relative
        if not target.is_file():
            if "__pycache__/" in relative or relative.startswith(("csrc/quantization/machete/generated/", "vllm/vllm_flash_attn/")):
                bundled_only.append(relative)
            else:
                missing.append(relative)
            continue
        expected = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual == expected:
            matched.append(relative)
        else:
            different.append(relative)
tracked = subprocess.check_output(["git", "-C", str(repo), "ls-files", "csrc", "cmake", "CMakeLists.txt", "vllm/_custom_ops.py"], text=True).splitlines()
fork_only = [path for path in tracked if path not in archive_paths]
report = {"reference": "PyPI vllm 0.10.2 sdist", "matched_count": len(matched), "different": different, "missing": missing, "fork_only": fork_only, "release_only_generated_and_bundled": bundled_only, "matched": matched}
(setup / "kernel-source-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({key: value for key, value in report.items() if key != "matched"}, indent=2))
assert matched and not different and not missing and not fork_only
