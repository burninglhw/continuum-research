import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ORDER = ["vllm-fcfs", "continuum-public", "continuum-paper", "elastic",
         "static-25", "static-50", "static-75", "conditional-whole"]
PRIMARY = ["vllm-fcfs", "continuum-paper", "elastic"]
LABELS = ["vLLM-fork FCFS", "Continuum public", "Continuum rebuilt", "Elastic prefix",
          "Static 25%", "Static 50%", "Static 75%", "Conditional whole"]
LABEL = dict(zip(ORDER, LABELS))
COLOR = dict(zip(ORDER, ["#718096", "#805ad5", "#2b6cb0", "#dd6b20",
                        "#38a169", "#2f855a", "#276749", "#b83280"]))
PLANNED = {(mode, seed) for mode in PRIMARY for seed in [42, 43, 44]}
PLANNED.update((mode, 42) for mode in ORDER if mode not in PRIMARY)


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def reduction(reference, alternative):
    return (1 - alternative / reference) * 100 if reference else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    workload = json.loads((args.root / "data/lengths.json").read_text())
    expected = {program["instance_id"]: len(program["steps"]) for program in workload}
    runs = {}
    configs = {}
    job_records = {}
    events = {}
    omitted = []
    sources = None
    shared_config = None
    seed_arrivals = {}
    for directory in sorted((args.root / "runs").iterdir()):
        if not directory.is_dir():
            continue
        if not (directory / "summary.json").exists():
            omitted.append({"directory": directory.name, "reason": "No completed summary"})
            continue
        config = json.loads((directory / "config.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        jobs = read_lines(directory / "jobs.jsonl")
        turns = read_lines(directory / "turns.jsonl")
        metrics = json.loads((directory / "scheduler-metrics.json").read_text())
        key = (config["mode"], config["seed"])
        assert key in PLANNED and key not in runs, ("Unexpected or duplicate run", key)
        assert config["status"] == "completed" and summary["complete"], directory
        assert len(jobs) == summary["programs"] == 100
        assert {job["instance_id"]: job["responses"] for job in jobs} == expected
        assert len(turns) == summary["responses"] == sum(expected.values()) == 3856
        assert len({(turn["instance_id"], turn["step"]) for turn in turns}) == len(turns)
        assert all(turn["forced_tokens_verified"] for turn in turns)
        assert summary["all_forced_tokens_verified"]
        assert sum(turn["output_tokens"] for turn in turns) == 579884
        assert sum(turn["prompt_tokens"] for turn in turns) == 99191508
        assert config["program_ids"] == [program["instance_id"] for program in workload]
        assert abs(statistics.mean(job["jct_seconds"] for job in jobs) - summary["mean_jct"]) < 1e-6
        if sources is None:
            sources = config["source_sha256"]
        assert sources == config["source_sha256"], ("Runtime source changed", directory)
        selected = {field: config[field] for field in ["model", "data", "cost_profile", "jps", "kv_gib",
                    "cascade_attention", "gpu_uuid", "reasoning_included", "tool_timing"]}
        if shared_config is None:
            shared_config = selected
        assert selected == shared_config, ("Experimental configuration differs", directory)
        previous_arrivals = seed_arrivals.setdefault(config["seed"], config["arrival_offsets"])
        assert previous_arrivals == config["arrival_offsets"], ("Arrivals differ", key)
        reclamation = metrics["reclamation"]
        net = [entry["net_raf"] for entry in reclamation]
        raw = [entry["raf"] for entry in reclamation]
        scheduled = metrics["scheduler_seconds"]
        queue_delays = [entry["queue_seconds"] for entry in metrics["admissions"]]
        runs[key] = {
            "mode": key[0], "seed": key[1], "directory": directory.name, **summary,
            "preemptions": metrics["metrics"].get("preemption_events", 0),
            "admissions": len(queue_delays),
            "mean_admission_queue_seconds": statistics.mean(queue_delays) if queue_delays else None,
            "p95_admission_queue_seconds": percentile(queue_delays, .95),
            "reported_cached_input_tokens": sum(turn["hit_tokens"] for turn in turns),
            "net_raf_median": percentile(net, .5), "net_raf_p95": percentile(net, .95),
            "net_raf_max": max(net) if net else None, "raw_raf_median": percentile(raw, .5),
            "net_endangered_blocks": sum(entry["net_endangered_blocks"] for entry in reclamation),
            "admission_deficit_blocks": sum(entry["needed_blocks"] for entry in reclamation),
            "unresolved_reclamations": sum(entry["remaining_deficit"] > 0 for entry in reclamation),
            "total_scheduler_seconds": sum(scheduled),
            "scheduler_fraction_of_makespan": sum(scheduled) / summary["makespan"],
            "mean_reclamation_ms": statistics.mean(entry["seconds"] for entry in reclamation) * 1000 if reclamation else None,
        }
        configs[key] = config
        job_records[key] = {job["instance_id"]: job for job in jobs}
        events[key] = reclamation
    comparisons = []
    per_job = []
    for seed in [42, 43, 44]:
        for reference_mode, alternative_mode in [("vllm-fcfs", "continuum-paper"),
                                                  ("vllm-fcfs", "continuum-public"),
                                                  ("vllm-fcfs", "elastic"),
                                                  ("continuum-paper", "elastic"),
                                                  ("conditional-whole", "elastic")]:
            reference_key = reference_mode, seed
            alternative_key = alternative_mode, seed
            if reference_key not in runs or alternative_key not in runs:
                continue
            reference = runs[reference_key]
            alternative = runs[alternative_key]
            wins = 0
            for instance_id in expected:
                before = job_records[reference_key][instance_id]["jct_seconds"]
                after = job_records[alternative_key][instance_id]["jct_seconds"]
                wins += after < before
                per_job.append({"seed": seed, "reference": reference_mode, "alternative": alternative_mode,
                                "instance_id": instance_id, "responses": expected[instance_id],
                                "reference_jct": before, "alternative_jct": after,
                                "jct_reduction_percent": reduction(before, after)})
            comparisons.append({
                "seed": seed, "reference": reference_mode, "alternative": alternative_mode,
                "mean_jct_reduction_percent": reduction(reference["mean_jct"], alternative["mean_jct"]),
                "p95_jct_reduction_percent": reduction(reference["p95_jct"], alternative["p95_jct"]),
                "historical_recompute_reduction_percent": reduction(reference["historical_prefix_recomputed_tokens"], alternative["historical_prefix_recomputed_tokens"]),
                "physical_eviction_reduction_percent": reduction(reference["physical_eviction_blocks"], alternative["physical_eviction_blocks"]),
                "tool_waiting_memory_time_reduction_percent": reduction(reference["tool_waiting_pinned_gib_seconds"], alternative["tool_waiting_pinned_gib_seconds"]),
                "faster_jobs": wins, "total_jobs": len(expected),
            })
    cohorts = defaultdict(list)
    for row in per_job:
        for lower, upper, label in [(1, 20, "1-20"), (21, 50, "21-50"),
                                     (51, 100, "51-100"), (101, 1000000, "101+")]:
            if lower <= row["responses"] <= upper:
                cohorts[(row["seed"], row["reference"], row["alternative"], label)].append(row)
    cohort_rows = []
    for (seed, reference, alternative, label), selected in sorted(cohorts.items()):
        before = statistics.mean(row["reference_jct"] for row in selected)
        after = statistics.mean(row["alternative_jct"] for row in selected)
        cohort_rows.append({"seed": seed, "reference": reference, "alternative": alternative,
                            "response_count_cohort": label, "jobs": len(selected),
                            "reference_mean_jct": before, "alternative_mean_jct": after,
                            "mean_jct_reduction_percent": reduction(before, after),
                            "faster_jobs": sum(row["alternative_jct"] < row["reference_jct"] for row in selected)})
    completed_turns = [done for count in expected.values() for done in range(1, count)]
    remaining_turns = [count - done for count in expected.values() for done in range(1, count)]
    workload_summary = {"programs": len(expected), "responses": sum(expected.values()),
                        "responses_per_program_mean": statistics.mean(expected.values()),
                        "responses_per_program_population_sd": statistics.pstdev(expected.values()),
                        "responses_per_program_median": statistics.median(expected.values()),
                        "max_responses_per_program": max(expected.values()),
                        "accepted_responses": sum(step["accepted"] for program in workload for step in program["steps"]),
                        "eta_unclipped_complete_corpus": -statistics.correlation(completed_turns, remaining_turns),
                        "definition": "Every observed response, including recovery, counts as one replay request."}
    aggregate = []
    for mode in PRIMARY:
        selected = [runs[(mode, seed)] for seed in [42, 43, 44] if (mode, seed) in runs]
        if selected:
            values = [row["mean_jct"] for row in selected]
            aggregate.append({"mode": mode, "seeds": [row["seed"] for row in selected],
                              "mean_jct": statistics.mean(values),
                              "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
                              "min_mean_jct": min(values), "max_mean_jct": max(values),
                              "mean_p95_jct": statistics.mean(row["p95_jct"] for row in selected)})
    evidence = {"complete_runs": len(runs), "planned_runs": len(PLANNED),
                "missing_runs": [{"mode": mode, "seed": seed} for mode, seed in sorted(PLANNED - runs.keys())],
                "omitted_incomplete_directories": omitted, "common_config": shared_config,
                "common_runtime_sha256": sources,
                "runs": [runs[key] for key in sorted(runs)], "primary_aggregate": aggregate,
                "paired_seed_comparisons": comparisons, "paired_response_cohorts": cohort_rows,
                "workload_summary": workload_summary,
                "exact_paper_reproduction": False, "real_gpu_replay": True,
                "inference": "Descriptive paired arrival-seed comparisons on one shared GPU; no independence or population-significance claim."}
    (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2))
    for filename, rows in [("run-metrics.csv", evidence["runs"]), ("paired-jobs.csv", per_job),
                           ("paired-seeds.csv", comparisons), ("paired-response-cohorts.csv", cohort_rows)]:
        if rows:
            with (args.output / filename).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    print(json.dumps({"completed": len(runs), "missing": evidence["missing_runs"],
                      "aggregate": aggregate, "comparisons": comparisons}, indent=2))
    if args.no_plots:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    if aggregate:
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
        for mode in PRIMARY:
            selected = [runs[(mode, seed)] for seed in [42, 43, 44] if (mode, seed) in runs]
            if not selected:
                continue
            for axis, field in zip(axes, ["mean_jct", "p95_jct"]):
                axis.plot([row["seed"] for row in selected], [row[field] for row in selected],
                          marker="o", color=COLOR[mode], label=LABEL[mode])
                axis.set_xticks([42, 43, 44])
                axis.set_xlabel("Arrival seed")
                axis.legend(fontsize=8)
        axes[0].set(ylabel="Mean JCT (seconds)", title="Primary methods: paired full-corpus runs")
        axes[1].set(ylabel="P95 JCT (seconds)", title="Shared H200; JPS 0.12; 24 GiB KV")
        figure.savefig(args.output / "primary-seeds.png", dpi=180)
        plt.close(figure)
    selected = [runs[(mode, 42)] for mode in ORDER if (mode, 42) in runs]
    if selected:
        figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
        positions = list(range(len(selected)))
        axes[0].bar(positions, [row["mean_jct"] for row in selected], color=[COLOR[row["mode"]] for row in selected])
        axes[0].set_xticks(positions, [LABEL[row["mode"]] for row in selected], rotation=40, ha="right")
        axes[0].set(ylabel="Mean JCT (seconds)", title="Same-seed controls (seed 42 only)")
        for row in selected:
            mode = row["mode"]
            values = sorted(job["jct_seconds"] for job in job_records[(mode, 42)].values())
            axes[1].plot(values, [(index + 1) / 100 for index in range(100)], color=COLOR[mode], label=LABEL[mode])
        axes[1].set(xlabel="Job completion time (seconds)", ylabel="CDF over all 100 jobs", title="No trajectory filtering or truncation")
        axes[1].legend(fontsize=8)
        figure.savefig(args.output / "same-seed-controls.png", dpi=180)
        plt.close(figure)
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
        for row in selected:
            mode = row["mode"]
            values = sorted(event["net_raf"] for event in events[(mode, 42)])
            if values:
                axes[0].plot(values, [(index + 1) / len(values) for index in range(len(values))],
                             color=COLOR[mode], label=f"{LABEL[mode]} ({len(values)})")
            axes[1].scatter(row["tool_waiting_pinned_gib_seconds"], row["mean_jct"], color=COLOR[mode], s=60, label=LABEL[mode])
        axes[0].set(xlabel="Net newly endangered blocks / admission deficit", ylabel="CDF over reclamations",
                    title="Net RAF (not physical eviction amplification)")
        axes[0].set_xscale("symlog", linthresh=1)
        axes[0].set_xlim(left=0)
        axes[0].legend(fontsize=7)
        axes[1].set(xlabel="Tool-waiting protected KV (GiB seconds)", ylabel="Mean JCT (seconds)",
                    title="Protection / latency trade-off, seed 42")
        axes[1].legend(fontsize=8)
        figure.savefig(args.output / "granularity-tradeoff.png", dpi=180)
        plt.close(figure)
if __name__ == "__main__":
    main()
