import argparse
import collections
import hashlib
import json
import random
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def normalize(payload, record):
    assert payload["instance_id"] == record["instance_id"]
    assert payload["trajectory_format"] == "mini-swe-agent-1"
    messages = payload["messages"]
    turns = []
    pattern = payload["info"]["config"]["agent"]["action_regex"]
    for index, message in enumerate(messages):
        assert message["role"] in {"system", "user", "assistant"}
        assert isinstance(message["content"], str)
        if message["role"] != "assistant":
            continue
        following = messages[index + 1] if index + 1 < len(messages) else None
        actions = re.findall(pattern, message["content"], re.DOTALL)
        usage = message.get("extra", {}).get("response", {}).get("usage", {})
        gap = None
        kind = "terminal_or_missing"
        if following and following["role"] == "user":
            gap = following["timestamp"] - message["timestamp"]
            assert gap >= 0
            if len(actions) != 1:
                kind = "format_error_feedback"
            elif "<returncode>" in following["content"]:
                kind = "action_observation"
            elif "timed out" in following["content"]:
                kind = "action_timeout"
            else:
                kind = "terminal_or_other_feedback"
        turns.append({
            "turn_id": len(turns), "assistant_message_index": index,
            "feedback_kind": kind, "action_count": len(actions),
            "action_feedback_gap_s": gap,
            "original_prompt_tokens": usage.get("prompt_tokens"),
            "original_completion_tokens": usage.get("completion_tokens"),
        })
    return {
        **record, "trajectory_format": payload["trajectory_format"],
        "actual_mini_version": payload["info"].get("mini_version"),
        "exit_status": payload["info"].get("exit_status"),
        "messages": [{key: message[key] for key in ("role", "content", "timestamp")} for message in messages],
        "turns": turns,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--verified", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    namespace = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    tree = ET.parse(arguments.audit / "swe-devstral-s3-all.xml").getroot()
    assert tree.find("s:IsTruncated", namespace).text == "false"
    official = {json.loads(line)["instance_id"] for line in arguments.verified.read_text().splitlines()}
    records = []
    for entry in tree.findall("s:Contents", namespace):
        key = entry.find("s:Key", namespace).text
        if not key.endswith(".traj.json"):
            continue
        instance_id = key.split("/")[-1].removesuffix(".traj.json")
        size = int(entry.find("s:Size", namespace).text)
        if instance_id in official:
            records.append({
                "instance_id": instance_id, "repo": instance_id.rsplit("-", 1)[0],
                "source_url": "https://swe-bench-submissions.s3.amazonaws.com/" + urllib.parse.quote(key),
                "etag": entry.find("s:ETag", namespace).text, "bytes": size,
            })
    selected = []
    for repo in ["astropy__astropy", "django__django", "scikit-learn__scikit-learn", "sympy__sympy"]:
        candidates = sorted([record for record in records if record["repo"] == repo and 32768 <= record["bytes"] <= 1000000], key=lambda record: record["instance_id"])
        random.Random(42).shuffle(candidates)
        assert len(candidates) >= 3
        for index, record in enumerate(candidates[:3]):
            selected.append({**record, "split": "development" if index == 0 else "evaluation"})
    assert sum(record["bytes"] for record in selected) <= 12000000
    manifest = {
        "source_index": "https://github.com/SWE-bench/experiments/tree/main/evaluation/verified/20251209_mini-v1.17.2_devstral-small-2512",
        "selection_seed": 42, "selection_rule": "3 per named repo; sorted IDs then seeded shuffle; 32768..1000000 bytes; no outcomes used; first is development",
        "matched_official_instances": len(records),
        "source_model": "Devstral Small 2512", "declared_mini_version": "1.17.2",
        "gap_semantics": "local client-observed assistant-finish to following action/feedback message; includes overhead, NOT pure tool runtime",
        "license_note": "Public trajectory access verified; a separate trajectory redistribution license has not been established. Raw content is not included in the report bundle.",
        "records": selected,
    }
    save_json(arguments.output / "selection_before_download.json", manifest)
    previous_path = arguments.output / "manifest.json"
    previous = {record["instance_id"]: record for record in json.loads(previous_path.read_text())["records"]} if previous_path.exists() else {}
    normalized = []
    for record in selected:
        path = arguments.output / "raw" / (record["instance_id"] + ".traj.json")
        if not path.exists():
            request = urllib.request.Request(record["source_url"], headers={"If-Match": record["etag"]})
            with urllib.request.urlopen(request, timeout=45) as response:
                content = response.read(1000001)
            assert len(content) == record["bytes"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        content = path.read_bytes()
        assert len(content) == record["bytes"]
        record["sha256"] = hashlib.sha256(content).hexdigest()
        if record["instance_id"] in previous:
            assert record["sha256"] == previous[record["instance_id"]]["sha256"], "Previously downloaded source changed"
        trajectory = normalize(json.loads(content), record)
        save_json(arguments.output / "normalized" / (record["instance_id"] + ".json"), trajectory)
        normalized.append(trajectory)
        print(record["instance_id"], record["split"], len(content), len(trajectory["turns"]), flush=True)
    manifest["total_bytes"] = sum(record["bytes"] for record in selected)
    manifest["trajectory_count"] = len(selected)
    manifest["assistant_turns"] = sum(len(trajectory["turns"]) for trajectory in normalized)
    manifest["feedback_counts"] = dict(collections.Counter(turn["feedback_kind"] for trajectory in normalized for turn in trajectory["turns"]))
    manifest["actual_mini_versions"] = sorted({trajectory["actual_mini_version"] for trajectory in normalized})
    manifest["exit_status_counts"] = dict(collections.Counter(trajectory["exit_status"] for trajectory in normalized))
    save_json(arguments.output / "manifest.json", manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
