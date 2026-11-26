#!/usr/bin/env python3
"""Validate and package a completed, evidence-gated Master v1.9 run."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
from split_bundle_v1_9 import split_and_verify


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(root: Path, names: list[str]) -> None:
    missing = [name for name in names if not (root / name).exists()]
    if missing:
        raise FileNotFoundError("Cannot freeze incomplete v1.9 run; missing: " + ", ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(args.results, [
        "FEATURE_MANIFEST_v1_9.json", "CONFIG_v1_9.json", "run_manifest_v1_9.json", "HARDWARE_PROFILE_v1_9.json", "PRECHECK_v1_9.json",
        "synthetic_seed_results.csv", "synthetic_extra_metrics_per_seed.csv", "synthetic_extra_metrics_summary.csv", "synthetic_refresh_coverage_per_seed.csv",
        "faithfulness_per_window.csv", "faithfulness_summary.csv", "delta_faithfulness_per_window.csv", "delta_faithfulness_summary.csv", "delta_faithfulness_paired.csv", "paired_aopc_differences.csv",
        "ablation_per_seed.csv", "ablation_summary.csv", "efficiency_2x2_trials.csv", "efficiency_2x2.csv",
        "scaling_per_seed.csv", "scaling_summary.csv", "architecture_validation.json", "multi_policy_temperature.csv",
        "method_latency_trials.csv", "method_latency_summary.csv", "method_latency_tail_metrics.csv",
        "public_detection_summary.csv", "public_calibrated_metrics_summary.csv", "public_policy_results.csv",
        "public_full_attribution_subset.csv", "public_full_attribution_summary.csv", "CONSISTENCY_REPORT_v1_9.json",
        "PAPER_EVIDENCE_MAP_v1_9.json", "PAPER_TABLES_v1_9.md", "PAPER_RELEASE_CHECKLIST_v1_9.md",
        "figure_scaling_latency_v1_9.png", "figure_common_window_faithfulness_v1_9.png", "figure_method_latency_v1_9.png",
        "figure_faithfulness_vs_latency_v1_9.png", "figure_detection_refresh_timeline_v1_9.png", "figure_fresh_attribution_heatmap_v1_9.png",
    ])
    report = json.loads((args.results / "CONSISTENCY_REPORT_v1_9.json").read_text())
    if report.get("status") != "PASS" or not report.get("public_alignment_gate"):
        raise RuntimeError("Cannot freeze: v1.9 consistency and public-alignment gates must pass.")
    precheck = json.loads((args.results / "PRECHECK_v1_9.json").read_text())
    if precheck.get("status") != "PASS" or precheck.get("public_reconstruction_label_alignment") != "window_end_t_minus_1":
        raise RuntimeError("Cannot freeze: the durable public-alignment preflight gate did not pass.")
    hardware = json.loads((args.results / "HARDWARE_PROFILE_v1_9.json").read_text())
    if not hardware.get("cuda_available") or hardware.get("timing_device_label") != "GPU":
        raise RuntimeError("Cannot freeze: official v1.9 timing bundle requires CUDA/GPU provenance.")
    public = pd.read_csv(args.results / "public_full_attribution_subset.csv")
    policy = pd.read_csv(args.results / "public_policy_results.csv")
    detection = pd.read_csv(args.results / "public_detection_summary.csv")
    if public.empty or set(public["mode"]) != {"full_attribution_subset"}:
        raise RuntimeError("Cannot freeze: public fresh-attribution subset is absent or mislabelled.")
    if set(public.label_alignment) != {"window_end_t_minus_1"} or set(detection.label_alignment) != {"window_end_t_minus_1"}:
        raise RuntimeError("Cannot freeze: public label alignment gate failed.")
    if not policy.exact_budget_match.all() or "offline_score_change_oracle" not in set(policy.policy):
        raise RuntimeError("Cannot freeze: public policy budget/oracle labelling gate failed.")
    required_methods = {"aw_ig", "full_swing", "zero_ig", "kernel_shap", "lime"}
    if set(pd.read_csv(args.results / "faithfulness_summary.csv").method) != required_methods:
        raise RuntimeError("Cannot freeze: common-window faithfulness is incomplete.")
    if set(pd.read_csv(args.results / "method_latency_tail_metrics.csv").method) != required_methods:
        raise RuntimeError("Cannot freeze: latency-tail evidence is incomplete.")
    delta = pd.read_csv(args.results / "delta_faithfulness_summary.csv")
    if set(delta["method"]) != {"full_swing_delta", "zero_ig_delta", "aw_ig_delta", "kernel_shap_delta", "lime_delta"}:
        raise RuntimeError("Cannot freeze: direct prediction-difference faithfulness is incomplete.")
    if set(pd.read_csv(args.results / "synthetic_seed_results.csv").seed) != {42, 52, 62}:
        raise RuntimeError("Cannot freeze: all three synthetic seeds are required.")

    base = args.output.with_suffix("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite existing frozen bundle: {args.output}')
    shutil.make_archive(str(base), "zip", args.results)
    receipt = {"release": "Master v1.9", "bundle": str(args.output), "sha256": sha256(args.output),
               "seeds": [42, 52, 62], "status": "frozen", "hardware": hardware,
               "public_label_alignment": "window_end_t_minus_1"}
    args.output.with_suffix(".freeze_receipt.json").write_text(json.dumps(receipt, indent=2))
    parts_manifest = split_and_verify(args.output)
    print(json.dumps(receipt, indent=2))
    print('Verified 15 MiB parts manifest:', parts_manifest)


if __name__ == "__main__":
    main()
