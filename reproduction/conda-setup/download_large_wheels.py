import concurrent.futures
import hashlib
import json
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

from pip._vendor.packaging.tags import sys_tags
from pip._vendor.packaging.utils import parse_wheel_filename


setup = Path("/export/home/ext.luohaowen1/continuum/reproduction/conda-setup")
rank = {tag: index for index, tag in enumerate(sys_tags())}
planned = re.findall(r"^\s+\+ ([a-zA-Z0-9_.-]+)==([^\s]+)", (setup / "dependency-plan.log").read_text(), re.MULTILINE)
names = {"vllm", "torch", "xformers", "triton", "ray", "pyarrow", "llvmlite", "scipy", "numpy"}
targets = [(name, version) for name, version in planned if name in names or name.startswith("nvidia-")]


def download_ranges(url, temporary, total_bytes):
    offset = temporary.stat().st_size if temporary.exists() else 0
    assert offset <= total_bytes
    if offset == total_bytes:
        return
    remaining = total_bytes - offset
    span = (remaining + 3) // 4
    directory = temporary.parent / ".chunks" / temporary.name
    directory.mkdir(parents=True, exist_ok=True)
    ranges = [(start, min(start + span, total_bytes) - 1) for start in range(offset, total_bytes, span)]

    def chunk_download(bounds):
        start, end = bounds
        chunk = directory / f"{start}-{end}"
        expected = end - start + 1
        if not chunk.exists() or chunk.stat().st_size != expected:
            result = subprocess.run(["curl", "--fail", "--location", "--retry", "2", "--connect-timeout", "20", "--max-time", "900", "--silent", "--show-error", "--range", f"{start}-{end}", "--max-filesize", str(expected), "--write-out", "%{http_code}", url, "--output", str(chunk)], capture_output=True, text=True, check=True)
            assert result.stdout.strip() == "206", result.stdout
        assert chunk.stat().st_size == expected
        return chunk

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        chunks = list(pool.map(chunk_download, ranges))
    with temporary.open("ab") as destination:
        for chunk in chunks:
            with chunk.open("rb") as source:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
    assert temporary.stat().st_size == total_bytes
    for chunk in chunks:
        chunk.unlink()
    directory.rmdir()


def fetch(package):
    name, version = package
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=30) as response:
        metadata = json.load(response)
    compatible = []
    for artifact in metadata["urls"]:
        if artifact["packagetype"] != "bdist_wheel" or artifact["yanked"]:
            continue
        _, _, _, tags = parse_wheel_filename(artifact["filename"])
        scores = [rank[tag] for tag in tags if tag in rank]
        if scores:
            compatible.append((min(scores), artifact))
    assert compatible, package
    artifact = min(compatible, key=lambda candidate: candidate[0])[1]
    if artifact["size"] < 30000000 and name != "vllm":
        return None
    target = setup / "downloads" / artifact["filename"]
    assert target.name == artifact["filename"]
    start = time.monotonic()
    if not target.exists():
        temporary = target.with_suffix(target.suffix + ".part")
        download_ranges(artifact["url"], temporary, artifact["size"])
        temporary.replace(target)
    assert target.stat().st_size == artifact["size"], target.name
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == artifact["digests"]["sha256"], target.name
    print(name, version, artifact["size"], f"{time.monotonic() - start:.1f}s", "SHA256 verified", flush=True)
    return {"name": name, "version": version, "filename": target.name, "url": artifact["url"], "bytes": artifact["size"], "sha256": digest.hexdigest(), "path": str(target)}


with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    records = [record for record in pool.map(fetch, targets) if record is not None]
(setup / "large-wheel-manifest.json").write_text(json.dumps(records, indent=2) + "\n")
(setup / "large-wheel-paths.txt").write_text("\n".join(record["path"] for record in records) + "\n")
print("Total wheel bytes:", sum(record["bytes"] for record in records), flush=True)
