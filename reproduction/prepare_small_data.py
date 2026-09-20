import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import duckdb
import requests


SWE_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
SWE_PARQUET_SHA256 = "a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd"
BFCL_REVISION = "7ad0134c665944819f88bc50862108d94015968b"
BFCL_PREFIX = "berkeley-function-call-leaderboard/bfcl_eval/data/"
BFCL_FILES = (
    ("BFCL_v4_web_search.json", "e4fd5c6543a8ba69823b6df9a9a3332daafe3c0a", 36984),
    ("possible_answer/BFCL_v4_web_search.json", "3af1a136e2e2d08f1cddeb546ee9e58364ade8dd", 75962),
    ("multi_turn_func_doc/web_search.json", "ecd4768683d8d6f102cbbadfcb41969b9bd322e3", 2252),
    ("README.md", "460ddf62835f5cc5c2ec86e254749ce86414e2c3", 20277),
)
MAX_FILE_BYTES = 10 * 1024 * 1024


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def git_blob_sha1(content):
    return hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()


def fetch(session, url, limit=MAX_FILE_BYTES):
    for attempt in range(3):
        try:
            with session.get(url, stream=True, timeout=(8, 25)) as response:
                response.raise_for_status()
                announced_size = int(response.headers.get("Content-Length", 0))
                if announced_size > limit:
                    raise ValueError(f"Download exceeds size cap: {url}")
                content = bytearray()
                for chunk in response.iter_content(65536):
                    content.extend(chunk)
                    if len(content) > limit:
                        raise ValueError(f"Download exceeds size cap: {url}")
                return bytes(content)
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def save_file(root, relative_path, content, entries, **details):
    destination = root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != content:
            raise FileExistsError(f"Refusing to overwrite a different file: {destination}")
    else:
        with destination.open("xb") as output_file:
            output_file.write(content)
    entries.append({"path": relative_path, "bytes": len(content), "sha256": sha256(content), **details})
    print(f"Verified {relative_path}: {len(content):,} bytes", flush=True)
    return destination


def read_json_records(path):
    content = path.read_text(encoding="utf-8")
    try:
        decoded = json.loads(content)
        return decoded if isinstance(decoded, list) else [decoded]
    except json.JSONDecodeError:
        return [json.loads(line) for line in content.splitlines() if line.strip()]


def json_lines(records):
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records).encode()


def fetch_blob(session, blob_id, expected_size):
    transport_url = f"https://api.github.com/repos/ShishirPatil/gorilla/git/blobs/{blob_id}"
    metadata = json.loads(fetch(session, transport_url))
    content = base64.b64decode(metadata["content"])
    if len(content) != expected_size or git_blob_sha1(content) != blob_id:
        raise ValueError(f"Git blob verification failed: {blob_id}")
    return content, transport_url


def main():
    parser = argparse.ArgumentParser(description="Download pinned small datasets without models or containers.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    arguments = parser.parse_args()
    root = arguments.project_root.resolve() / "datasets"
    entries = []
    session = requests.Session()
    session.headers["User-Agent"] = "continuum-small-data-preparation/1.0"
    dataset_id = "princeton-nlp/SWE-bench_Verified"
    source_base = f"https://huggingface.co/datasets/{dataset_id}/resolve/{SWE_REVISION}"
    transport_base = f"{arguments.hf_endpoint.rstrip('/')}/datasets/{dataset_id}/resolve/{SWE_REVISION}"
    parquet_relative_path = "data/test-00000-of-00001.parquet"
    cached_parquet = root / "_sources" / "swe_bench_verified" / parquet_relative_path
    parquet_content = cached_parquet.read_bytes() if cached_parquet.exists() else fetch(session, f"{transport_base}/{parquet_relative_path}")
    if len(parquet_content) != 2096679 or sha256(parquet_content) != SWE_PARQUET_SHA256:
        raise ValueError("SWE-bench Verified size/SHA-256 mismatch")
    parquet_path = save_file(
        root, f"_sources/swe_bench_verified/{parquet_relative_path}", parquet_content, entries,
        source_url=f"{source_base}/{parquet_relative_path}", transport_url=f"{transport_base}/{parquet_relative_path}",
        revision=SWE_REVISION, expected_lfs_sha256=SWE_PARQUET_SHA256, row_count=500,
    )
    readme_content = fetch(session, f"{transport_base}/README.md")
    if git_blob_sha1(readme_content) != "f362a6ce9579a3ee1922e3402463feef139b7d00":
        raise ValueError("SWE-bench dataset card blob mismatch")
    save_file(root, "_sources/swe_bench_verified/README.md", readme_content, entries,
              source_url=f"{source_base}/README.md", transport_url=f"{transport_base}/README.md", revision=SWE_REVISION)
    with duckdb.connect(":memory:") as connection:
        result = connection.execute("SELECT * FROM read_parquet(?)", [str(parquet_path)])
        columns = [column[0] for column in result.description]
        swe_records = [dict(zip(columns, record)) for record in result.fetchall()]
    swe_records.sort(key=lambda record: record["instance_id"])
    swe_ids = [record["instance_id"] for record in swe_records]
    if len(swe_records) != 500 or len(set(swe_ids)) != 500:
        raise ValueError("Expected 500 unique SWE-bench Verified test instances")
    for record in swe_records:
        for field in ("instance_id", "repo", "base_commit", "problem_statement", "patch", "test_patch", "version", "FAIL_TO_PASS", "PASS_TO_PASS"):
            if field not in record:
                raise ValueError(f"Missing SWE-bench field: {field}")
    small_swe = random.Random(42).sample(swe_records, 100)
    smoke_swe = small_swe[:10]
    for relative_path, records, subset_kind in (
        ("swe_bench_verified/data/test.jsonl", swe_records, "official test split, lossless parquet-to-jsonl conversion"),
        ("small/swe_bench_verified_100/test.jsonl", small_swe, "new seed-42 sample; NOT the authors' recorded traces"),
        ("smoke/swe_bench_verified_10/test.jsonl", smoke_swe, "first 10 of the new seed-42 sample; NOT paper-scale evaluation"),
    ):
        save_file(root, relative_path, json_lines(records), entries, row_count=len(records),
                  derived_from=f"_sources/swe_bench_verified/{parquet_relative_path}", selection=subset_kind, seed=42)
    for relative_path, blob_id, expected_size in BFCL_FILES:
        cached_path = root / "bfcl_v4_web_search" / relative_path
        if cached_path.exists() and git_blob_sha1(cached_path.read_bytes()) == blob_id:
            content = cached_path.read_bytes()
            transport_url = f"https://api.github.com/repos/ShishirPatil/gorilla/git/blobs/{blob_id}"
        else:
            content, transport_url = fetch_blob(session, blob_id, expected_size)
        source_path = BFCL_PREFIX + relative_path
        save_file(root, "bfcl_v4_web_search/" + relative_path, content, entries,
                  source_url=f"https://raw.githubusercontent.com/ShishirPatil/gorilla/{BFCL_REVISION}/{source_path}",
                  transport_url=transport_url, revision=BFCL_REVISION, git_blob_sha1=blob_id)
    license_content, license_url = fetch_blob(session, "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64", 11357)
    save_file(root, "bfcl_v4_web_search/LICENSE", license_content, entries,
              source_url=f"https://raw.githubusercontent.com/ShishirPatil/gorilla/{BFCL_REVISION}/LICENSE",
              transport_url=license_url, revision=BFCL_REVISION)
    questions = read_json_records(root / "bfcl_v4_web_search/BFCL_v4_web_search.json")
    answers = read_json_records(root / "bfcl_v4_web_search/possible_answer/BFCL_v4_web_search.json")
    question_ids = [record["id"] for record in questions]
    answer_ids = [record["id"] for record in answers]
    if len(question_ids) != 100 or len(set(question_ids)) != 100 or set(question_ids) != set(answer_ids) or len(answer_ids) != 100:
        raise ValueError("BFCL must have 100 unique questions and matching answer IDs")
    bfcl_smoke = random.Random(42).sample(sorted(questions, key=lambda record: record["id"]), 10)
    answers_by_id = {record["id"]: record for record in answers}
    for relative_path, records in (
        ("smoke/bfcl_v4_web_search_10/questions.jsonl", bfcl_smoke),
        ("smoke/bfcl_v4_web_search_10/answers.jsonl", [answers_by_id[record["id"]] for record in bfcl_smoke]),
    ):
        save_file(root, relative_path, json_lines(records), entries, row_count=10, seed=42,
                  selection="new deterministic smoke sample; NOT GPT-5 execution traces")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper": "Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live",
        "paper_sha256": "d78aa8f0415311063158ebdd383f8ebde25414af5c28fdcf400968e7bb606ee2",
        "continuum_code_revision": "316a58794a6ff86b216e579b74fd56ed0c5a911f",
        "swe_dataset": {"id": dataset_id, "revision": SWE_REVISION, "split": "test", "rows": 500},
        "bfcl_dataset": {"repository": "ShishirPatil/gorilla", "revision": BFCL_REVISION, "unique_questions": 100,
                         "evaluation_variants": ["with search snippets", "without search snippets"],
                         "version_note": "Pinned to a revision before the PDF date; authors' exact collection revision is not published."},
        "missing_for_paper_trace_replay": ["Authors' 100 GPT-5 SWE-bench execution traces", "Authors' 100 GPT-5 BFCL execution traces", "Exact trace replay/preprocessing and BFCL 0.4 workload scaling implementation"],
        "not_downloaded": ["LLM weights", "Docker/SWE-bench task images", "Full BFCL non-web-search categories", "Full SWE-bench train/dev corpus"],
        "total_file_bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
    }
    manifest_path = root / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "prepared", "swe_verified_rows": 500, "bfcl_questions": 100,
                      "files": len(entries), "total_file_bytes": manifest["total_file_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
