import argparse
import collections
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

from prepare_data import normalize, save_json


def ecdf(axis, values, label, color):
    values = np.sort(values)
    axis.step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=label, color=color, linewidth=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    manifest = json.loads((arguments.data / "manifest.json").read_text())
    workload = json.loads(arguments.workload.read_text())
    selected = {record["instance_id"]: record for record in manifest["records"]}
    assert len(selected) == 12
    checks = []
    for record in manifest["records"]:
        content = (arguments.data / "raw" / (record["instance_id"] + ".traj.json")).read_bytes()
        assert len(content) == record["bytes"]
        assert hashlib.sha256(content).hexdigest() == record["sha256"]
        payload = json.loads(content)
        timestamps = [message["timestamp"] for message in payload["messages"]]
        assert timestamps == sorted(timestamps), "Nonmonotonic local timestamps"
        expected = normalize(payload, record)
        actual = json.loads((arguments.data / "normalized" / (record["instance_id"] + ".json")).read_text())
        assert actual == expected
        checks.append({"instance_id": record["instance_id"], "sha256_verified": True, "normalization_recomputed": True, "timestamps_monotonic": True, "redacted_marker_count": sum(message["content"].count("<redacted>") for message in payload["messages"])})
    assert {trace["instance_id"] for trace in workload["traces"]} == set(selected)
    all_turns = []
    pilot_turns = []
    for trace in workload["traces"]:
        assert trace["sha256"] == selected[trace["instance_id"]]["sha256"]
        assert trace["split"] == selected[trace["instance_id"]]["split"]
        last_end = 0
        for turn in trace["turns"]:
            assert 0 <= turn["reusable_prefix_tokens"] == last_end <= turn["prompt_tokens"] < turn["end_tokens"]
            assert turn["output_tokens"] == turn["end_tokens"] - turn["prompt_tokens"]
            assert turn["new_input_tokens"] == turn["prompt_tokens"] - last_end
            last_end = turn["end_tokens"]
        assert trace["pilot_turns"] == trace["turns"][:len(trace["pilot_turns"])], "Pilot must be an unchanged initial segment"
        assert all(turn["end_tokens"] <= 8192 for turn in trace["pilot_turns"])
        all_turns.extend(trace["turns"])
        if trace["pilot_eligible"]:
            pilot_turns.extend(trace["pilot_turns"])
    assert len(all_turns) == manifest["assistant_turns"] == 919
    assert len(pilot_turns) == 72
    development = [trace for trace in workload["traces"] if trace["split"] == "development"]
    evaluation = [trace for trace in workload["traces"] if trace["split"] == "evaluation"]
    assert len(development) == 4 and len(evaluation) == 8
    bytes_per_token = 2 * 36 * 8 * 128 * 2
    summary = {
        "gpu_experiment_status": "PAUSED_BY_USER; no GPU benchmark launched; no performance claims",
        "validation": checks,
        "hashes_and_normalization_valid": True,
        "split_sessions": {"development": len(development), "evaluation": len(evaluation)},
        "split_pilot_turns": {"development": sum(len(trace["pilot_turns"]) for trace in development), "evaluation": sum(len(trace["pilot_turns"]) for trace in evaluation)},
        "pilot_max_end_tokens": max(turn["end_tokens"] for turn in pilot_turns),
        "pilot_return_turns": sum(turn["turn_id"] > 0 for turn in pilot_turns),
        "pilot_feedback_counts": dict(collections.Counter(turn["feedback_kind"] for turn in pilot_turns)),
        "pilot_reusable_prefix_token_fraction": sum(turn["reusable_prefix_tokens"] for turn in pilot_turns) / sum(turn["prompt_tokens"] for turn in pilot_turns),
        "all_turns_exceeding_40960_total_tokens": sum(turn["end_tokens"] > 40960 for turn in all_turns),
        "kv_bytes_per_token_bf16_theoretical": bytes_per_token,
        "max_pilot_kv_gib_theoretical": max(turn["end_tokens"] for turn in pilot_turns) * bytes_per_token / 2 ** 30,
        "eval_final_kv_sum_gib_theoretical": sum(trace["pilot_turns"][-1]["end_tokens"] for trace in evaluation) * bytes_per_token / 2 ** 30,
        "memory_note": "Logical tensor size only, not measured allocated memory; no padding, cross-session sharing or allocator overhead.",
    }
    save_json(arguments.output / "validation.json", summary)
    save_json(arguments.output / "source_manifest.json", manifest)
    save_json(arguments.output / "workload_metadata.json", workload)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titleweight": "bold"})
    figure, axes = plt.subplots(2, 2, figsize=(12, 9.4))
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.13, top=0.88, hspace=0.4, wspace=0.28)
    figure.suptitle("SWE-bench trajectory pilot | DATA CHARACTERIZATION ONLY", fontsize=15, fontweight="bold", y=0.975)
    projects = ["astropy", "django", "scikit-learn", "sympy"]
    positions = np.arange(4)
    axes[0, 0].bar(positions, [1] * 4, color="#257f8b", label="Development: 4")
    axes[0, 0].bar(positions, [2] * 4, bottom=[1] * 4, color="#8dc5ca", label="Evaluation: 8")
    axes[0, 0].set(xticks=positions, xticklabels=projects, ylabel="Unique source sessions", ylim=(0, 4), title="12 fixed-seed traces; 3.60 MB raw")
    axes[0, 0].legend(frameon=False, loc="upper right")
    ecdf(axes[0, 1], [turn["prompt_tokens"] for turn in all_turns], "Full traces: 919 turns", "#257f8b")
    ecdf(axes[0, 1], [turn["prompt_tokens"] for turn in pilot_turns], "Initial 6 turns/session: 72", "#dc784c")
    axes[0, 1].axvline(8192, color="#a4a4a4", linestyle="--", linewidth=1)
    axes[0, 1].set(xscale="log", xlabel="Qwen3 serialized prompt tokens (log scale)", ylabel="Fraction of turns", title="Pilot excludes later long contexts")
    axes[0, 1].set_xticks([2000, 8000, 32000, 80000], ["2K", "8K", "32K", "80K"])
    axes[0, 1].xaxis.set_minor_formatter(NullFormatter())
    axes[0, 1].legend(frameon=False, loc="lower right")
    ecdf(axes[1, 0], [turn["action_feedback_gap_s"] for turn in all_turns if turn["feedback_kind"] == "action_observation"], "Action observation: 640", "#257f8b")
    ecdf(axes[1, 0], [turn["action_feedback_gap_s"] for turn in all_turns if turn["feedback_kind"] == "format_error_feedback"], "Format-error feedback: 268", "#dc784c")
    axes[1, 0].set(xscale="log", xlabel="Assistant-finish to feedback, seconds (log scale)", ylabel="Fraction within feedback class", title="Client-observed gaps, NOT pure tool runtime")
    axes[1, 0].legend(frameon=False, loc="lower right")
    for turn_number in range(6):
        current = [turn for turn in pilot_turns if turn["turn_id"] == turn_number]
        mean_old = np.mean([turn["reusable_prefix_tokens"] for turn in current])
        mean_new = np.mean([turn["new_input_tokens"] for turn in current])
        axes[1, 1].bar(turn_number + 1, mean_old, color="#257f8b", label="Potential reusable prefix" if turn_number == 0 else None)
        axes[1, 1].bar(turn_number + 1, mean_new, bottom=mean_old, color="#e6b494", label="New input" if turn_number == 0 else None)
    axes[1, 1].set(xlabel="Pilot turn (12 sessions per bar)", ylabel="Mean input tokens", title="Append-only potential, NOT actual cache hits", xticks=list(range(1, 7)))
    axes[1, 1].legend(frameon=False, loc="upper left")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
    figure.text(0.5, 0.018, "GPU tests paused by user. No measured/simulated JCT, latency, throughput, or Continuum speedup is shown.", ha="center", fontsize=10, color="#733a24")
    figure.savefig(arguments.output / "dataset_characterization.png", dpi=180, facecolor="white")
    figure.savefig(arguments.output / "dataset_characterization.svg", facecolor="white")
    print(json.dumps({key: value for key, value in summary.items() if key != "validation"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
