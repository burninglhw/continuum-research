import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from prepare_data import normalize, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    manifest = json.loads(arguments.manifest.read_text())
    assert sum(record["bytes"] for record in manifest["records"]) <= 12000000
    for record in manifest["records"]:
        target = arguments.output / "raw" / (record["instance_id"] + ".traj.json")
        if target.exists():
            content = target.read_bytes()
        else:
            request = urllib.request.Request(record["source_url"], headers={"If-Match": record["etag"]})
            with urllib.request.urlopen(request, timeout=45) as response:
                content = response.read(record["bytes"] + 1)
        assert len(content) == record["bytes"]
        assert hashlib.sha256(content).hexdigest() == record["sha256"]
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        normalized = normalize(json.loads(content), record)
        save_json(arguments.output / "normalized" / (record["instance_id"] + ".json"), normalized)
        print(record["instance_id"], "verified", flush=True)
    save_json(arguments.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
