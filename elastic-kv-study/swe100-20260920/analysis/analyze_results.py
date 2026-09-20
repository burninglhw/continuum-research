import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from policy import CostCurve


LABELS = {"vllm-fcfs": "vLLM-fork FCFS", "continuum-public": "Continuum public / fixed",
          "continuum-paper": "Continuum paper / rebuilt", "elastic": "Elastic prefix",
          "static-25": "Static 25%", "static-50": "Static 50%", "static-75": "Static 75%",
          "conditional-whole": "Conditional whole"}
ORDER = list(LABELS)
COLORS = {mode: color for mode, color in zip(ORDER, ["#718096", "#805ad5", "#2b6cb0", "#dd6b20",
                                                    "#38a169", "#2f855a", "#276749", "#b83280"])}


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def rws(curve, total, epsilon):
    maximum = total // 16
    low, high = 0, maximum
    limit = curve.latency(maximum * 16, total) + epsilon
    while low < high:
        middle = (low + high) // 2
        if curve.latency(middle * 16, total) <= limit:
            high = middle
        else:
            low = middle + 1
    return low * 16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    curve = CostCurve(args.root / "profile-v2/curve.json")
    workload = json.loads((args.root / "data/lengths.json").read_text())
    characterization = []
    for program in workload:
        for step in program["steps"][:-1]:
            if step["tool"] is None:
                continue
            total = step["input_tokens"] + step["output_tokens"] - 1
            keep = rws(curve, total, .1)
            characterization.append({"instance_id": program["instance_id"], "turn": step["turn"],
                                     "tokens": total, "rws100ms": keep, "psr100ms": 1 - keep / total,
                                     "prediction_not_direct_per_trace_measurement": True})
    (args.output / "characterization.json").write_text(json.dumps(characterization))
    runs = []
    groups = defaultdict(list)
    expected_jobs = {program["instance_id"] for program in workload}
    for directory in sorted((args.root / "runs").glob("*")):
        if not (directory / "summary.json").exists():
            continue
        config = json.loads((directory / "config.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        jobs = read_lines(directory / "jobs.jsonl")
        turns = read_lines(directory / "turns.jsonl")
        assert summary["complete"] and summary["programs"] == 100
        assert {job["instance_id"] for job in jobs} == expected_jobs
        assert len(turns) == 3856 and all(turn["forced_tokens_verified"] for turn in turns)
        assert summary["all_forced_tokens_verified"]
        record = {"directory": directory.name, "mode": config["mode"], "seed": config["seed"],
                  "jps": config["jps"], "kv_gib": config["kv_gib"], **summary}
        runs.append(record)
        groups[config["mode"]].append(record)
    comparisons = []
    for seed in [42, 43, 44]:
        by_mode = {record["mode"]: record for record in runs if record["seed"] == seed}
        if "continuum-paper" not in by_mode or "elastic" not in by_mode:
            continue
        reference = by_mode["continuum-paper"]
        alternative = by_mode["elastic"]
        comparisons.append({"seed": seed,
                            "jct_reduction_percent": (1 - alternative["mean_jct"] / reference["mean_jct"]) * 100,
                            "p95_reduction_percent": (1 - alternative["p95_jct"] / reference["p95_jct"]) * 100,
                            "recompute_reduction_percent": (1 - alternative["historical_prefix_recomputed_tokens"] / max(1, reference["historical_prefix_recomputed_tokens"])) * 100,
                            "tool_waiting_memory_time_reduction_percent": (1 - alternative["tool_waiting_pinned_gib_seconds"] / max(1e-9, reference["tool_waiting_pinned_gib_seconds"])) * 100})
        configs = [json.loads((args.root / "runs" / by_mode[mode]["directory"] / "config.json").read_text())
                   for mode in ("continuum-paper", "elastic")]
        assert configs[0]["arrival_offsets"] == configs[1]["arrival_offsets"]
        assert configs[0]["program_ids"] == configs[1]["program_ids"]
        assert configs[0]["source_sha256"] == configs[1]["source_sha256"], "Run source versions differ"
    aggregate = []
    for mode in ORDER:
        trials = groups[mode]
        if not trials:
            continue
        values = [trial["mean_jct"] for trial in trials]
        aggregate.append({"mode": mode, "trials": len(trials), "mean_jct": statistics.mean(values),
                          "jct_sample_sd": statistics.stdev(values) if len(values) > 1 else None,
                          "p95_jct": statistics.mean(trial["p95_jct"] for trial in trials),
                          "throughput": statistics.mean(trial["throughput_jobs_per_second"] for trial in trials),
                          "waiting_pinned_gib_seconds": statistics.mean(trial["tool_waiting_pinned_gib_seconds"] for trial in trials),
                          "recomputed_tokens": statistics.mean(trial["historical_prefix_recomputed_tokens"] for trial in trials),
                          "scheduler_ms": statistics.mean(trial["mean_scheduler_ms"] for trial in trials)})
    evidence = {"complete_runs": len(runs), "runs": runs, "aggregate": aggregate,
                "elastic_vs_paper_by_seed": comparisons,
                "all_results_real_gpu_replay": True, "exact_paper_reproduction": False,
                "characterization": {"waiting_tool_events": len(characterization),
                    "median_psr100ms": statistics.median(row["psr100ms"] for row in characterization),
                    "fraction_with_psr_below_5percent": statistics.mean(row["psr100ms"] < .05 for row in characterization)}}
    (args.output / "results.json").write_text(json.dumps(evidence, indent=2))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for entry in curve.curves:
        positions = [point["cached"] / entry["total"] for point in entry["points"]]
        penalties = [(point["seconds"] - entry["points"][-1]["seconds"]) * 1000 for point in entry["points"]]
        axes[0].plot(positions, penalties, marker="o", markersize=3, label=f"{entry['total']/1024:.0f}K")
    axes[0].axhline(100, color="#444", linestyle="--", linewidth=.8)
    axes[0].set(xlabel="Cached prefix fraction", ylabel="Extra prefill latency (ms)", yscale="symlog", title="Measured H200 / 8B prefix costs (3 repeats)")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(ncol=2, fontsize=8)
    values = sorted(row["psr100ms"] * 100 for row in characterization)
    axes[1].plot(values, [(index + 1) / len(values) for index in range(len(values))], color="#dd6b20")
    axes[1].set(xlabel="Protection slack within +100 ms (%)", ylabel="CDF over tool-wait events",
                title="Interpolated characterization, not loaded JCT")
    figure.savefig(args.output / "prefix-characterization.png", dpi=170)
    plt.close(figure)
    if aggregate:
        figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
        positions = list(range(len(aggregate)))
        axes[0].bar(positions, [row["mean_jct"] for row in aggregate],
                    color=[COLORS[row["mode"]] for row in aggregate],
                    yerr=[row["jct_sample_sd"] or 0 for row in aggregate], capsize=3)
        axes[0].set_xticks(positions, [f"{LABELS[row['mode']]}\nn={row['trials']}" for row in aggregate], rotation=40, ha="right")
        axes[0].set(ylabel="Mean JCT (s); error bar = seed SD", title="100 full trajectories / H200 / JPS .12 / KV 24 GiB")
        for row in aggregate:
            axes[1].scatter(row["waiting_pinned_gib_seconds"], row["mean_jct"], color=COLORS[row["mode"]], s=65, label=LABELS[row["mode"]])
        axes[1].set(xlabel="Tool-waiting protected KV (GiB·s)", ylabel="Mean JCT (s)", title="Measured latency / memory-time trade-off")
        axes[1].legend(fontsize=8)
        figure.savefig(args.output / "replay-comparison.png", dpi=170)
        plt.close(figure)
        figure, axis = plt.subplots(figsize=(8, 4.5), layout="constrained")
        for mode in ORDER:
            events = []
            for run in groups[mode]:
                metrics = json.loads((args.root / "runs" / run["directory"] / "scheduler-metrics.json").read_text())
                events.extend(row["net_raf"] for row in metrics["reclamation"])
            if events:
                events.sort()
                axis.plot(events, [(index + 1) / len(events) for index in range(len(events))], label=f"{LABELS[mode]} ({len(events)})", color=COLORS[mode])
        axis.set(xlabel="Newly endangered blocks / admission deficit (net RAF)", ylabel="CDF over reclamation events", xscale="log", title="Granularity mismatch: actual scheduler events")
        if axis.lines:
            axis.legend(fontsize=8)
        figure.savefig(args.output / "reclamation-amplification.png", dpi=170)
        plt.close(figure)
    print(json.dumps({"runs": len(runs), "aggregate": aggregate, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
