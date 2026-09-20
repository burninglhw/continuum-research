import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


def load_json_records(path):
    text = path.read_text(encoding="utf-8")
    try:
        decoded = json.loads(text)
        return decoded if isinstance(decoded, list) else [decoded]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference_metrics(repository, dataset_ids):
    summaries = {}
    for mode in ("vllm", "continuum"):
        result_dir = repository / "continuum_exp" / f"swebench_llama70B_{mode}"
        result_path = result_dir / "results.json"
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text())
        verification = json.loads((result_dir / "verification.json").read_text())
        durations = list(result["job_durations"].values())
        require(len(durations) == result["num_jobs"], f"Reference job count mismatch: {mode}")
        require(math.isclose(statistics.mean(durations), result["average_duration"], rel_tol=1e-12),
                f"Reference average mismatch: {mode}")
        referenced_ids = set()
        for field, value in verification.items():
            if field.endswith("_ids") and isinstance(value, list):
                referenced_ids.update(value)
        require(referenced_ids <= dataset_ids, f"Reference SWE-bench IDs missing from downloaded split: {mode}")
        summaries[mode] = {
            "source": "pre-existing author result JSON; NOT a newly executed benchmark",
            "num_jobs": result["num_jobs"],
            "average_duration_seconds": result["average_duration"],
            "p95_duration_seconds": result["percentile_95"],
            "total_instances": verification["total_instances"],
            "submitted_instances": verification["submitted_instances"],
            "resolved_instances": verification["resolved_instances"],
            "resolved_rate_over_all_500": verification["resolved_instances"] / verification["total_instances"],
            "reference_instance_ids_checked": len(referenced_ids),
        }
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Verify prepared data offline using only Python's standard library.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    project_root = arguments.project_root.resolve()
    data_root = project_root / "datasets"
    manifest = json.loads((data_root / "MANIFEST.json").read_text())
    byte_count = 0
    for entry in manifest["files"]:
        file_path = (data_root / entry["path"]).resolve()
        require(file_path.is_relative_to(data_root), "Manifest path escapes the data directory")
        content = file_path.read_bytes()
        require(len(content) == entry["bytes"], f"File-size mismatch: {entry['path']}")
        require(hashlib.sha256(content).hexdigest() == entry["sha256"], f"SHA-256 mismatch: {entry['path']}")
        if "git_blob_sha1" in entry:
            actual_blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
            require(actual_blob == entry["git_blob_sha1"], f"Git blob mismatch: {entry['path']}")
        if "row_count" in entry and file_path.suffix == ".jsonl":
            require(len(load_json_records(file_path)) == entry["row_count"], f"Row-count mismatch: {entry['path']}")
        byte_count += len(content)
    require(byte_count == manifest["total_file_bytes"], "Total file-size mismatch")
    swe_full = load_json_records(data_root / "swe_bench_verified/data/test.jsonl")
    swe_small = load_json_records(data_root / "small/swe_bench_verified_100/test.jsonl")
    swe_smoke = load_json_records(data_root / "smoke/swe_bench_verified_10/test.jsonl")
    dataset_ids = {record["instance_id"] for record in swe_full}
    require(len(swe_full) == len(dataset_ids) == 500, "Expected 500 unique SWE-bench Verified test instances")
    required_fields = {"instance_id", "repo", "base_commit", "problem_statement", "patch", "test_patch", "version", "FAIL_TO_PASS", "PASS_TO_PASS"}
    for record in swe_full:
        require(required_fields <= record.keys(), "SWE-bench schema is incomplete")
        for field in ("FAIL_TO_PASS", "PASS_TO_PASS"):
            require(isinstance(json.loads(record[field]), list), f"Invalid SWE-bench test list: {field}")
    expected_small = random.Random(42).sample(sorted(swe_full, key=lambda record: record["instance_id"]), 100)
    require(swe_small == expected_small, "SWE-bench 100-instance selection is not reproducible")
    require(swe_smoke == expected_small[:10], "SWE-bench smoke sample does not match its parent sample")
    questions = load_json_records(data_root / "bfcl_v4_web_search/BFCL_v4_web_search.json")
    answers = load_json_records(data_root / "bfcl_v4_web_search/possible_answer/BFCL_v4_web_search.json")
    question_ids = {record["id"] for record in questions}
    answer_ids = {record["id"] for record in answers}
    require(len(questions) == len(answers) == len(question_ids) == len(answer_ids) == 100, "BFCL count/uniqueness mismatch")
    require(question_ids == answer_ids, "BFCL question/answer IDs differ")
    for question in questions:
        require(bool(question.get("question")) and "WebSearchAPI" in question.get("involved_classes", []), "Invalid BFCL web-search question")
    tools = load_json_records(data_root / "bfcl_v4_web_search/multi_turn_func_doc/web_search.json")
    require(bool(tools), "Missing BFCL tool definitions")
    smoke_questions = load_json_records(data_root / "smoke/bfcl_v4_web_search_10/questions.jsonl")
    smoke_answers = load_json_records(data_root / "smoke/bfcl_v4_web_search_10/answers.jsonl")
    expected_questions = random.Random(42).sample(sorted(questions, key=lambda record: record["id"]), 10)
    require(smoke_questions == expected_questions, "BFCL smoke selection is not reproducible")
    answer_lookup = {record["id"]: record for record in answers}
    require(smoke_answers == [answer_lookup[record["id"]] for record in smoke_questions], "BFCL smoke answer mapping mismatch")
    report = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "verified_files": len(manifest["files"]),
        "verified_bytes": byte_count,
        "swe_verified_test_instances": 500,
        "swe_small_instances": 100,
        "swe_smoke_instances": 10,
        "bfcl_unique_questions": 100,
        "bfcl_matching_answer_records": 100,
        "bfcl_smoke_instances": 10,
        "seed": 42,
        "checks": ["file size", "SHA-256", "BFCL Git blobs", "record schema", "unique IDs", "question-answer alignment", "deterministic subsetting"],
        "benchmark_executed": False,
        "models_downloaded": False,
        "container_images_downloaded": False,
        "author_reference_results": reference_metrics(project_root / "vllm-continuum", dataset_ids),
    }
    output = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
