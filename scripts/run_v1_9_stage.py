#!/usr/bin/env python3
"""Memory-bounded, auditable Master v1.9 experiment stages.

One invocation writes one durable stage artifact and exits.  This design is
intentional: Colab releases Python/PyTorch memory between stages, while the
assembler rejects incomplete or scientifically invalid result bundles.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import core_v1_9 as core
from adaptive_deltaxai.models import DataPlaneGRU, DataPlaneTransformer, ReconGRU
from adaptive_deltaxai.metrics_v1_8 import (
    aligned_public_labels,
    detector_metrics,
    latency_distribution,
    paired_aopc_bootstrap,
    positive_event_ids,
    refresh_coverage_metrics,
    validation_f1_threshold,
    normal_calibration_quantile,
)
from adaptive_deltaxai.swing import linear_completeness_check


ROOT = Path(__file__).parents[1]
SEEDS = (42, 52, 62)
METHODS = ("zero_ig", "aw_ig", "full_swing", "kernel_shap", "lime")


def read_config() -> dict:
    return json.loads((ROOT / "configs" / "master_v1_9.json").read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in {".py", ".json", ".md", ".txt"}):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def hardware_profile() -> dict:
    cpu = "unknown"
    memory_kb = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip(); break
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                memory_kb = int(line.split()[1]); break
    except OSError:
        pass
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_model": cpu,
        "memory_kb": memory_kb,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "timing_device_label": "GPU" if torch.cuda.is_available() else "CPU",
    }


def init(data_root: Path, output: Path, *, require_cuda: bool) -> tuple[dict, torch.device]:
    output.mkdir(parents=True, exist_ok=True)
    cfg = read_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    required = ["tier2_temperature_stream.npy", "tier2_power_stream.npy", "tier2_labels.npy"]
    missing = [x for x in required if not (data_root / x).exists()]
    if missing:
        raise FileNotFoundError(f"Missing synthetic input files: {missing}")
    data_hashes = {name: sha256_file(data_root / name) for name in required}
    split = cfg["split"]
    manifest = {
        "experiment_id": "AdaptiveDeltaXAI_Master_v1_9",
        "hardware": hardware_profile(),
        "code_tree_sha256": tree_hash(ROOT),
        "synthetic_data_sha256": data_hashes,
        "config": cfg,
        "execution_mode": "one_memory_bounded_process_per_durable_stage",
    }
    (output / "CONFIG_v1_9.json").write_text(json.dumps(cfg, indent=2))
    (output / "FEATURE_MANIFEST_v1_9.json").write_text((ROOT / "manifests" / "FEATURE_MANIFEST_v1_9.json").read_text())
    (output / "run_manifest_v1_9.json").write_text(json.dumps(manifest, indent=2))
    (output / "HARDWARE_PROFILE_v1_9.json").write_text(json.dumps(manifest["hardware"], indent=2))
    if split != {"type": "chronological", "train": 0.70, "validation": 0.10, "test": 0.20}:
        raise RuntimeError("v1.9 requires the frozen chronological 70/10/20 split.")
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("Official v1.9 stages require CUDA/T4. Use --allow-cpu-smoke only for preflight development.")
    return cfg, device


def stage_preflight(data_root: Path, output: Path, cfg: dict, device: torch.device) -> None:
    """Write a durable scientific preflight receipt before any public rerun.

    The notebook runs the complete pytest suite as well.  This stage repeats
    the two release-critical numerical contracts in the process that will run
    the experiments, and records the CUDA/data provenance alongside them.
    """
    target = output / "PRECHECK_v1_9.json"
    required = ["tier2_temperature_stream.npy", "tier2_power_stream.npy", "tier2_labels.npy"]
    missing = [name for name in required if not (data_root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing synthetic inputs during preflight: {missing}")
    toy_labels = np.asarray([0, 1, 0, 1, 1, 0, 0], dtype=int)
    aligned = aligned_public_labels(toy_labels, window=3, n_scores=4)
    if aligned.tolist() != [0, 1, 1, 0]:
        raise RuntimeError("Public reconstruction t-1 label-alignment regression gate failed.")
    residual = float(linear_completeness_check(str(device)))
    if residual >= 1e-5:
        raise RuntimeError(f"Full-SWING linear completeness preflight failed: {residual}")
    receipt = {
        "status": "PASS",
        "release": "Master v1.9",
        "public_reconstruction_label_alignment": "window_end_t_minus_1",
        "synthetic_classifier_label_alignment": "window_target_t",
        "full_swing_linear_completeness": residual,
        "hardware": hardware_profile(),
        "data_files": required,
    }
    target.write_text(json.dumps(receipt, indent=2))
    print(f"[PASS] preflight receipt written: {target}", flush=True)


def require_precheck(output: Path) -> None:
    """Prevent stale/unverified public result generation."""
    path = output / "PRECHECK_v1_9.json"
    if not path.exists():
        raise RuntimeError("Run the v1.9 preflight stage before public evaluation.")
    receipt = json.loads(path.read_text())
    if receipt.get("status") != "PASS" or receipt.get("public_reconstruction_label_alignment") != "window_end_t_minus_1":
        raise RuntimeError("The v1.9 public-alignment preflight receipt is invalid.")


def release_memory(*objects) -> None:
    del objects
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def split_with_audit(raw: np.ndarray, labels: np.ndarray, cfg: dict, output: Path, stream: str):
    parts, targets, scale = core.split_normalize(raw, labels)
    w = cfg["window"]
    audit = {
        "stream": stream,
        "n_total": int(len(raw)),
        "train_end": int(scale["train_end"]),
        "validation_end": int(scale["val_end"]),
        "test_start": int(scale["test_start"]),
        "raw_split_sizes": [int(len(x)) for x in parts],
        "valid_window_counts": [int(len(x) - w) for x in parts],
        "positive_window_counts": [int(y[w:].sum()) for y in targets],
        "normalization_fit": "training_observations_only",
        "window_policy": "within_split_only; no cross-boundary inputs",
    }
    if min(audit["valid_window_counts"]) <= 0 or sum(audit["raw_split_sizes"]) != len(raw):
        raise RuntimeError("Invalid chronological split/window alignment.")
    (output / f"split_audit_{stream}.json").write_text(json.dumps(audit, indent=2))
    return parts, targets, scale


def load_model(data_root: Path, output: Path, device: torch.device, seed: int, stream: str = "temperature"):
    streams, labels = core.load_synthetic(data_root)
    raw = streams[stream]
    cfg = read_config()
    parts, targets, scale = split_with_audit(raw, labels, cfg, output, stream)
    checkpoint = output / f"checkpoint_gru_{stream}_seed{seed}.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing {checkpoint.name}; run the corresponding synthetic stage first.")
    model = DataPlaneGRU(raw.shape[1]).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return streams, labels, raw, parts, targets, scale, model


def stage_synthetic(data_root: Path, output: Path, cfg: dict, device: torch.device, seed: int, stream: str):
    target = output / f"synthetic_{stream}_seed{seed}.json"
    if target.exists():
        print(f"[SKIP] {target.name}"); return
    streams, labels = core.load_synthetic(data_root)
    raw = streams[stream]
    parts, targets, scale = split_with_audit(raw, labels, cfg, output, stream)
    core.set_seed(seed)
    print(f"[RUN] synthetic stream={stream}, seed={seed}", flush=True)
    model = DataPlaneGRU(raw.shape[1]).to(device)
    model = core.train_binary(model, parts[0], targets[0], parts[1], targets[1], cfg, device, seed)
    checkpoint = output / f"checkpoint_gru_{stream}_seed{seed}.pt"
    torch.save(model.state_dict(), checkpoint)
    normalized = (raw - scale["mean"]) / scale["std"]
    records = core.run_adaptive(normalized, model, {**cfg, **cfg[stream]}, device, seed=seed)
    summary = core.summarize(records)
    if summary["fresh_completeness_residual_mean"] >= 0.05:
        raise RuntimeError("Synthetic fresh-SWING completeness gate failed.")
    cache = core.cache_fidelity(records, normalized, model, {**cfg, **cfg[stream]}, device, 100, seed)
    cache.to_csv(output / f"cache_fidelity_{stream}_seed{seed}.csv", index=False)
    core.save_controller_records(output / f"controller_{stream}_seed{seed}.npz", records)
    pd.DataFrame(core.grouped_context_rows(records, stream, raw.shape[1])).to_csv(output / f"topology_context_{stream}_seed{seed}.csv", index=False)
    record = {"seed": seed, "stream": stream,
              **{f"detector_{k}": v for k, v in core.metrics(model, parts[2], targets[2], cfg["window"], device).items()},
              **{f"operational_{k}": v for k, v in summary.items() if k != "refresh_reasons"}}
    target.write_text(json.dumps(record, indent=2, default=float))
    release_memory(model, records, cache)


def predict_window_scores(model, points: np.ndarray, window: int, device: torch.device, *, reconstruction: bool) -> np.ndarray:
    """Bounded inference.  Windows x[i-window:i] retain their documented target convention."""
    if len(points) <= window:
        raise ValueError("No valid windows available for scoring.")
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(window, len(points), 256):
            stop = min(start + 256, len(points))
            batch = np.stack([points[i-window:i] for i in range(start, stop)]).astype(np.float32)
            tensor = torch.as_tensor(batch, dtype=torch.float32, device=device)
            score = model.anomaly_score(tensor) if reconstruction else model.predict_proba(tensor)
            out.append(score.reshape(-1).detach().cpu().numpy())
            del tensor, score
    return np.concatenate(out).astype(np.float64)


def plot_detection_curves(labels: np.ndarray, scores: np.ndarray, prefix: Path, title: str) -> None:
    """Evaluation-only ROC/PR figures; labels are never used to choose a threshold."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
    if len(np.unique(labels)) < 2:
        return
    fpr, tpr, _ = roc_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(5.2, 3.5)); ax.plot(fpr, tpr, label=f"AUROC={roc_auc_score(labels, scores):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1); ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"{title}: ROC")
    ax.legend(); fig.tight_layout(); fig.savefig(prefix.with_name(prefix.name + "_roc.png"), dpi=250); plt.close(fig)
    precision, recall, _ = precision_recall_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(5.2, 3.5)); ax.plot(recall, precision, label=f"AP={average_precision_score(labels, scores):.3f}")
    ax.axhline(float(labels.mean()), linestyle="--", color="gray", linewidth=1, label="prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", title=f"{title}: precision-recall", xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(); fig.tight_layout(); fig.savefig(prefix.with_name(prefix.name + "_pr.png"), dpi=250); plt.close(fig)


def stage_synthetic_metrics(data_root: Path, output: Path, cfg: dict, device: torch.device, seed: int, stream: str):
    """Three-seed held-out detection and refresh-event metrics, derived from v1.9 checkpoints."""
    target = output / f"synthetic_{stream}_seed{seed}_extra_metrics.csv"
    coverage_target = output / f"synthetic_{stream}_seed{seed}_refresh_coverage.json"
    if target.exists() and coverage_target.exists(): print(f"[SKIP] {target.name}"); return
    streams, labels, raw, parts, targets, scale, model = load_model(data_root, output, device, seed, stream)
    w = cfg["window"]
    test_scores = predict_window_scores(model, parts[2], w, device, reconstruction=False)
    y_test = targets[2][w:w + len(test_scores)]
    fixed = detector_metrics(y_test, test_scores, threshold=0.5, probabilities=True, seconds_per_step=10.0)
    fixed.update({"dataset": "synthetic", "stream": stream, "seed": seed, "threshold_source": "fixed_probability_0.5"})
    val_scores = predict_window_scores(model, parts[1], w, device, reconstruction=False)
    y_val = targets[1][w:w + len(val_scores)]
    reports = [fixed]
    if len(np.unique(y_val)) == 2:
        threshold = validation_f1_threshold(y_val, val_scores)
        tuned = detector_metrics(y_test, test_scores, threshold=threshold, probabilities=True, seconds_per_step=10.0)
        tuned.update({"dataset": "synthetic", "stream": stream, "seed": seed, "threshold_source": "validation_f1"})
        reports.append(tuned)
    pd.DataFrame(reports).to_csv(target, index=False)
    (output / f"synthetic_{stream}_seed{seed}_extra_metrics.json").write_text(json.dumps(reports, indent=2, default=float))
    plot_detection_curves(y_test, test_scores, output / f"synthetic_{stream}_seed{seed}", f"Synthetic {stream}, seed {seed}")
    # Coverage needs two compact arrays only.  Do not deserialize the large
    # cached-attribution tensor from the controller archive in this stage.
    target_index, refreshed = core.load_controller_refresh_arrays(output / f"controller_{stream}_seed{seed}.npz")
    fresh = refreshed.astype(np.int8, copy=False)
    eligible = (target_index >= scale["test_start"]) & (target_index < len(labels))
    if not eligible.any():
        raise RuntimeError("No held-out controller records available for refresh coverage.")
    coverage = refresh_coverage_metrics(labels[target_index[eligible]], fresh[eligible])
    coverage.update({"stream": stream, "seed": seed, "scope": "held_out_test", "target_alignment": "synthetic_x_i_minus_w_to_i_minus_1_predicts_y_i"})
    coverage_target.write_text(json.dumps(coverage, indent=2, default=float))
    release_memory(model, target_index, fresh)


def stage_faithfulness(data_root: Path, output: Path, cfg: dict, device: torch.device):
    if (output / "faithfulness_summary.csv").exists(): print("[SKIP] faithfulness"); return
    _, _, _, parts, targets, scale, model = load_model(data_root, output, device, 42)
    core.faithfulness(model, parts[2], targets[2], np.zeros_like(scale["mean"]), cfg, device, output, 42)
    release_memory(model)


def stage_delta_faithfulness(data_root: Path, output: Path, cfg: dict, device: torch.device):
    """Primary check: perturb paired real windows and evaluate prediction difference."""
    import shap
    from lime.lime_tabular import LimeTabularExplainer
    target = output / "delta_faithfulness_per_window.csv"; summary = output / "delta_faithfulness_summary.csv"
    paired_path = output / 'delta_faithfulness_paired.csv'
    if target.exists() and summary.exists() and paired_path.exists():
        print('[SKIP] direct delta faithfulness'); return
    streams, labels, raw, parts, targets, scale, model = load_model(data_root, output, device, 42, "temperature")
    normalized=(raw-scale['mean'])/scale['std']; w=cfg['window']; lag=cfg['swing_lag']; required=w+1+lag
    start=max(int(scale['test_start']), required)
    candidates=np.asarray([i for i in range(start, len(labels)) if labels[i] == 1],dtype=int)
    rng=np.random.default_rng(42); chosen=rng.choice(candidates, min(cfg['faithfulness_windows'],len(candidates)), replace=False)
    rows=[]
    # The shared input is the union of adjacent windows. The background is
    # drawn exclusively from the training split, never from the test set.
    train = parts[0]
    union_width = w + 1
    train_unions = np.lib.stride_tricks.sliding_window_view(train, union_width, axis=0).transpose(0, 2, 1)
    bg_count = min(30, len(train_unions))
    bg_idx = rng.choice(len(train_unions), bg_count, replace=False)
    background = np.asarray(train_unions[bg_idx]).reshape(bg_count, -1)
    def delta_predict(flat):
        x = np.asarray(flat, dtype=np.float32).reshape(-1, union_width, train.shape[1])
        with torch.no_grad():
            tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
            result = core.score_fn(model)(tensor[:, 1:]) - core.score_fn(model)(tensor[:, :-1])
        return result.detach().cpu().numpy()
    kernel = shap.KernelExplainer(delta_predict, background)
    lime = LimeTabularExplainer(background, mode='regression', discretize_continuous=False, random_state=42)
    for n,target_index in enumerate(chosen,1):
        endpoint=int(target_index-1); history=normalized[endpoint-required+1:endpoint+1]
        if history.shape[0] != required: raise RuntimeError("Direct delta-faithfulness history contract failed.")
        full,_=core.fresh_swing(model,history,cfg,device)
        methods={"full_swing_delta":full[-(w+1):], **core.delta_baseline_attributions(model,history,cfg,device)}
        union = history[-union_width:].reshape(-1)
        sv = kernel.shap_values(union.reshape(1, -1), nsamples=cfg['kernel_shap_samples'])
        methods['kernel_shap_delta'] = np.asarray(sv).reshape(union_width, train.shape[1])
        explanation = lime.explain_instance(union, delta_predict, num_features=union.size)
        lm = np.zeros(union.size, dtype=np.float32)
        for feature, value in explanation.as_map()[1]: lm[feature] = value
        methods['lime_delta'] = lm.reshape(union_width, train.shape[1])
        for name,attr in methods.items():
            rows.append({"method":name,"endpoint_index":endpoint,"target_index":int(target_index),"aopc_prediction_difference":core.delta_aopc(model,history,attr,np.zeros_like(history),cfg,device)})
        if n%10==0: print(f"  direct delta faithfulness {n}/{len(chosen)}",flush=True)
    frame=pd.DataFrame(rows); frame.to_csv(target,index=False)
    out=[]
    for name,group in frame.groupby('method'):
        mean,lo,hi=core.bootstrap_ci(group.aopc_prediction_difference.values,cfg['bootstrap'],42)
        out.append({"method":name,"n_common_windows":len(group),"mean_prediction_difference_aopc":mean,"ci_low":lo,"ci_high":hi,"history_source":"real_observed_W_plus_delta_plus_lag"})
    pd.DataFrame(out).to_csv(summary,index=False)
    wide = frame.pivot(index='target_index', columns='method', values='aopc_prediction_difference')
    expected = {'full_swing_delta', 'aw_ig_delta', 'zero_ig_delta', 'kernel_shap_delta', 'lime_delta'}
    if set(wide.columns) != expected or wide.isna().any().any():
        raise RuntimeError('All five methods must share the same evaluated windows.')
    paired = []
    for baseline in sorted(expected - {'full_swing_delta'}):
        differences = (wide['full_swing_delta'] - wide[baseline]).to_numpy()
        mean, low, high = core.bootstrap_ci(differences, cfg['bootstrap'], 42)
        paired.append({'baseline': baseline, 'n_pairs': len(differences), 'mean_difference': mean,
                       'ci_low': low, 'ci_high': high, 'bootstrap_unit': 'window',
                       'interpretation': 'exploratory_window_bootstrap'})
    pd.DataFrame(paired).to_csv(output / 'delta_faithfulness_paired.csv', index=False)
    release_memory(model, frame)


def stage_policy(data_root: Path, output: Path, cfg: dict, device: torch.device, policy: str, seed: int):
    target = output / f"policy_temperature_{policy}_seed{seed}.json"
    if target.exists(): print(f"[SKIP] {target.name}"); return
    source = output / f"synthetic_temperature_seed{seed}.json"
    if not source.exists(): raise FileNotFoundError(f"Run synthetic temperature seed {seed} first.")
    saved = json.loads(source.read_text())
    policy_label = "offline_score_change_oracle" if policy == "score_change" else policy
    if policy == "adaptive":
        target.write_text(json.dumps({"policy": policy_label, "causal_online_policy": True, "result_source": source.name,
                                      **{k.removeprefix("operational_"): v for k, v in saved.items() if k.startswith("operational_")}}, indent=2)); return
    streams, _, raw, _, _, scale, model = load_model(data_root, output, device, seed)
    budget = int(saved["operational_n_fresh_calls"])
    stream = (streams["temperature"] - scale["mean"]) / scale["std"]
    records = core.run_adaptive(stream, model, {**cfg, **cfg["temperature"]}, device, policy, budget, seed)
    summary = core.summarize(records)
    if summary["n_fresh_calls"] != budget:
        raise RuntimeError("Synthetic matched-budget policy count mismatch.")
    target.write_text(json.dumps({"policy": policy_label, "seed": seed, "causal_online_policy": bool(policy != "score_change"), "matched_refresh_count": budget, **summary}, indent=2, default=float))
    release_memory(model, records)


def stage_ablation(data_root: Path, output: Path, cfg: dict, device: torch.device, seed: int, variant: str):
    target = output / f"ablation_{variant}_seed{seed}.json"
    if target.exists(): print(f"[SKIP] {target.name}"); return
    patches = {"Full": {}, "No_I1": {"always_refresh": True, "use_i1": False}, "No_I2": {"use_i2": False}, "No_I3": {"use_i3": False}, "No_I4": {"use_i4": False}, "No_I5": {"use_i5": False}}
    if variant not in patches: raise ValueError(variant)
    streams, _, _, _, _, scale, model = load_model(data_root, output, device, seed)
    start = scale["test_start"] - (cfg["window"] + cfg["swing_lag"] + 1)
    test_stream = ((streams["temperature"] - scale["mean"]) / scale["std"])[start:]
    records = core.run_adaptive(test_stream, model, {**cfg, **cfg["temperature"], **patches[variant]}, device, seed=seed)
    fresh = [r for r in records if r.refreshed][:cfg["faithfulness_windows"]]
    aopcs = [core.aopc(model, test_stream[r.t-cfg["window"]+1:r.t+1], r.cached_event_context[-cfg["window"]:], np.zeros(test_stream.shape[1], dtype=np.float32), device) for r in fresh]
    target.write_text(json.dumps({"seed": seed, "configuration": variant, **core.summarize(records), "aopc_fresh_only": float(np.mean(aopcs)), "aopc_n_fresh": len(aopcs)}, indent=2, default=float))
    release_memory(model, records)


def stage_scaling(data_root: Path, output: Path, cfg: dict, device: torch.device, seed: int, dimensions: int):
    target = output / f"scaling_D{dimensions}_seed{seed}.json"
    if target.exists(): print(f"[SKIP] {target.name}"); return
    streams, labels = core.load_synthetic(data_root); base = streams["temperature"]
    rng = np.random.default_rng(seed + dimensions)
    reps = int(np.ceil(dimensions / base.shape[1]))
    expanded = np.tile(base, (1, reps))[:, :dimensions] + rng.normal(0, 1e-3, (len(base), dimensions)).astype(np.float32)
    parts, targets, scale = split_with_audit(expanded, labels, cfg, output, f"scaling_D{dimensions}")
    core.set_seed(seed)
    model = core.train_binary(DataPlaneGRU(dimensions).to(device), parts[0], targets[0], parts[1], targets[1], cfg, device, seed)
    normalized = (expanded - scale["mean"]) / scale["std"]
    start = scale["test_start"] - (cfg["window"] + cfg["swing_lag"] + 1)
    records = core.run_adaptive(normalized[start:], model, {**cfg, **cfg["temperature"]}, device, seed=seed)
    target.write_text(json.dumps({"seed": seed, "sensor_count": dimensions, "stream_scope": "held_out_test_with_history", **core.summarize(records)}, indent=2, default=float))
    release_memory(model, records)


def stage_architecture(data_root: Path, output: Path, cfg: dict, device: torch.device, seed: int):
    target = output / f"architecture_seed{seed}.json"
    if target.exists(): print(f"[SKIP] {target.name}"); return
    streams, labels, raw, parts, targets, scale, gru = load_model(data_root, output, device, seed)
    core.set_seed(seed)
    trm = DataPlaneTransformer(raw.shape[1]).to(device)
    trm = core.train_binary(trm, parts[0], targets[0], parts[1], targets[1], cfg, device, seed)
    torch.save(trm.state_dict(), output / f"checkpoint_transformer_temperature_seed{seed}.pt")
    normalized = (raw - scale["mean"]) / scale["std"]
    gru_records = core.run_adaptive(normalized, gru, {**cfg, **cfg["temperature"]}, device, seed=seed)
    trm_records = core.run_adaptive(normalized, trm, {**cfg, **cfg["temperature"]}, device, seed=seed)
    target.write_text(json.dumps({"seed": seed, "same_split_seed_controller": True,
                                  "gru_detector": core.metrics(gru, parts[2], targets[2], cfg["window"], device),
                                  "transformer_detector": core.metrics(trm, parts[2], targets[2], cfg["window"], device),
                                  "gru_operational": core.summarize(gru_records),
                                  "transformer_operational": core.summarize(trm_records)}, indent=2, default=float))
    release_memory(gru, trm, gru_records, trm_records)


def public_paths(data_root: Path, dataset: str, unit: str):
    smd = data_root / "SMD" / "ServerMachineDataset"
    if dataset == "SMD":
        return smd / "train" / f"{unit}.txt", smd / "test" / f"{unit}.txt", smd / "test_label" / f"{unit}.txt"
    return data_root / "PSM" / "train.csv", data_root / "PSM" / "test.csv", data_root / "PSM" / "test_label.csv"


def read_public(dataset: str, paths):
    train_path, test_path, label_path = paths
    if dataset == "SMD":
        return np.loadtxt(train_path, delimiter=",").astype(np.float32), np.loadtxt(test_path, delimiter=",").astype(np.float32), np.loadtxt(label_path, delimiter=",").astype(int)
    train = pd.read_csv(train_path).drop(columns=["timestamp_(min)"], errors="ignore").ffill().fillna(0).values.astype(np.float32)
    test = pd.read_csv(test_path).drop(columns=["timestamp_(min)"], errors="ignore").ffill().fillna(0).values.astype(np.float32)
    labels = pd.read_csv(label_path).iloc[:, -1].values.astype(int)
    return train, test, labels


def adaptive_public_mask(scores: np.ndarray, cfg: dict) -> np.ndarray:
    mask = np.zeros(len(scores), dtype=bool); ema = float(scores[0]); dev = 1e-6; age = 0
    for i, score in enumerate(scores):
        signal = abs(float(score) - ema)
        fire = signal > cfg["temperature"]["score_k"] * max(dev, 1e-6) or age >= cfg["temperature"]["max_age"]
        mask[i] = fire
        age = 0 if fire else age + 1
        dev = .95 * dev + .05 * signal; ema = .95 * ema + .05 * float(score)
    mask[0] = True
    return mask


def matched_mask(policy: str, scores: np.ndarray, count: int, seed: int) -> np.ndarray:
    count = min(max(1, count), len(scores)); mask = np.zeros(len(scores), dtype=bool); mask[0] = True
    remaining = count - 1
    if policy == "periodic": mask[np.linspace(0, len(scores)-1, count, dtype=int)] = True
    elif policy == "random" and remaining: mask[np.random.default_rng(seed).choice(np.arange(1, len(scores)), remaining, replace=False)] = True
    elif policy == "score_change" and remaining:
        candidates = np.argsort(np.abs(np.diff(scores, prepend=scores[0]))[1:])[-remaining:] + 1; mask[candidates] = True
    if int(mask.sum()) != count: raise RuntimeError("Policy mask failed the exact matched-budget gate.")
    return mask


def normalized_aopc(model, window: np.ndarray, attribution: np.ndarray, reference: np.ndarray, device: torch.device, kmax: int = 8) -> float:
    """Score-relative AOPC; denominator floor prevents zero-score explosions."""
    fn = core.score_fn(model); x = np.array(window, copy=True)
    with torch.no_grad(): baseline = float(fn(torch.as_tensor(x, device=device).unsqueeze(0))[0])
    denom = max(abs(baseline), 0.1)
    ranks = np.argsort(np.abs(attribution).ravel())[::-1]; drops = []
    for k in range(1, kmax + 1):
        masked = x.copy(); idx = np.unravel_index(ranks[:k], x.shape); masked[idx] = np.broadcast_to(reference, x.shape)[idx]
        with torch.no_grad(): after = float(fn(torch.as_tensor(masked, device=device).unsqueeze(0))[0])
        drops.append((baseline - after) / denom)
    return float(np.mean(drops))


def stage_public_unit(data_root: Path, output: Path, cfg: dict, device: torch.device, dataset: str, unit: str, seed: int):
    require_precheck(output)
    stem = f"public_{dataset}_{unit}_seed{seed}".replace("/", "_")
    target = output / f"{stem}.csv"; summary_target = output / f"{stem}.json"
    if target.exists() and summary_target.exists(): print(f"[SKIP] {stem}"); return
    train, test, labels = read_public(dataset, public_paths(data_root, dataset, unit))
    mean = train.mean(0); std = np.where(train.std(0) < 1e-8, 1., train.std(0))
    train, test = (train - mean) / std, (test - mean) / std
    checkpoint = output / f"checkpoint_recon_{dataset}_{unit}_seed{seed}.pt"
    model = core.train_reconstruction(train, cfg, device, seed, checkpoint, checkpoint.exists())
    windows, scores = core.public_scores(model, test, cfg, device)
    # ReconGRU reconstructs the final element of x[i-window:i], i.e. x[i-1].
    # This one canonical helper is used by all public label-dependent analyses.
    y = aligned_public_labels(labels, cfg["window"], len(windows))
    detection = detector_metrics(y, scores, probabilities=False)
    detection.update({"dataset": dataset, "unit": unit, "seed": seed,
                      "score_definition": "log1p_reconstruction_mse",
                      "label_alignment": "window_end_t_minus_1",
                      "threshold_source": "none_threshold_free_only"})
    pd.DataFrame([detection]).to_csv(output / f"public_detection_{dataset}_{unit}_seed{seed}.csv", index=False)
    (output / f"public_detection_{dataset}_{unit}_seed{seed}.json").write_text(json.dumps(detection, indent=2, default=float))
    plot_detection_curves(y, scores, output / f"public_{dataset}_{unit}_seed{seed}", f"{dataset} {unit}, seed {seed}")
    adaptive = adaptive_public_mask(scores, cfg); budget = int(adaptive.sum())
    policies = {"adaptive": adaptive, "periodic": matched_mask("periodic", scores, budget, seed),
                "random": matched_mask("random", scores, budget, seed),
                "offline_score_change_oracle": matched_mask("score_change", scores, budget, seed)}
    policy_rows = []
    for name, mask in policies.items():
        coverage = refresh_coverage_metrics(y, mask.astype(int))
        policy_rows.append({"dataset": dataset, "unit": unit, "seed": seed, "mode": "score_only_scheduling", "policy": name,
                            "causal_online_policy": bool(name != "offline_score_change_oracle"),
                            "refresh_count": int(mask.sum()), "recompute_pct": float(100*mask.mean()), "matched_budget": budget,
                            "exact_budget_match": bool(int(mask.sum()) == budget), **coverage})
    positives = np.where(y == 1)[0]; rng = np.random.default_rng(seed)
    chosen = rng.choice(positives, min(cfg["public"]["subset_windows"], len(positives)), replace=False) if len(positives) else rng.choice(len(windows), cfg["public"]["subset_windows"], replace=False)
    pad = np.vstack([np.repeat(test[:1], cfg["window"]+cfg["swing_lag"]+1, axis=0), test])
    rows = []
    for number, idx in enumerate(chosen, 1):
        end = cfg["window"] + cfg["swing_lag"] + 1 + idx + cfg["window"]
        history = pad[end-(cfg["window"]+1+cfg["swing_lag"]):end]
        attribution, latency = core.fresh_swing(model, history, cfg, device)
        delta = float(scores[idx] - scores[max(0, idx-1)])
        residual = float(abs(attribution.sum() - delta))
        rows.append({"dataset": dataset, "unit": unit, "seed": seed, "mode": "full_attribution_subset", "window_index": int(idx), "label": int(y[idx]),
                     "raw_label_index": int(cfg["window"] - 1 + idx), "label_alignment": "window_end_t_minus_1",
                     "score_definition": "log1p_reconstruction_mse", "full_swing_latency_ms": latency,
                     "fresh_completeness_residual": residual, "aopc_score_relative": normalized_aopc(model, windows[idx], attribution[-cfg["window"]:], np.zeros(windows.shape[2], dtype=np.float32), device)})
        if number % 10 == 0: print(f"  {dataset}/{unit}/seed{seed}: {number}/{len(chosen)}", flush=True)
    frame = pd.DataFrame(rows); target.write_text(frame.to_csv(index=False))
    mean_resid = float(frame.fresh_completeness_residual.mean()); p95_resid = float(np.percentile(frame.fresh_completeness_residual, 95))
    summary_target.write_text(json.dumps({"dataset": dataset, "unit": unit, "seed": seed, "public_score": "log1p_reconstruction_mse", "label_alignment": "window_end_t_minus_1", "n_windows": len(frame),
                                           "mean_completeness_residual": mean_resid, "p95_completeness_residual": p95_resid,
                                           "completeness_gate_pass": bool(mean_resid <= cfg["public"]["max_mean_completeness_residual"] and p95_resid <= cfg["public"]["max_p95_completeness_residual"]),
                                           "policy_rows": policy_rows}, indent=2))
    (output / f"{stem}_policies.json").write_text(json.dumps(policy_rows, indent=2))
    release_memory(model, frame)


def stage_public_calibrated(data_root: Path, output: Path, cfg: dict, device: torch.device, dataset: str, unit: str, seed: int):
    """Supplementary public thresholded metrics using a chronological non-test calibration split."""
    require_precheck(output)
    target = output / f"public_calibrated_{dataset}_{unit}_seed{seed}.csv"
    if target.exists(): print(f"[SKIP] {target.name}"); return
    train_raw, test_raw, labels = read_public(dataset, public_paths(data_root, dataset, unit))
    cut = int((1.0 - float(cfg["public"]["calibration_fraction"])) * len(train_raw))
    if cut <= cfg["window"] or len(train_raw) - cut <= cfg["window"]:
        raise RuntimeError("Public training partition is too short for chronological training/calibration split.")
    fit_raw, calibration_raw = train_raw[:cut], train_raw[cut:]
    mu = fit_raw.mean(0); sd = np.where(fit_raw.std(0) < 1e-8, 1.0, fit_raw.std(0))
    fit = (fit_raw - mu) / sd; calibration = (calibration_raw - mu) / sd; test = (test_raw - mu) / sd
    checkpoint = output / f"checkpoint_recon_calibrated_{dataset}_{unit}_seed{seed}.pt"
    model = core.train_reconstruction(fit, cfg, device, seed, checkpoint, checkpoint.exists())
    calibration_scores = predict_window_scores(model, calibration, cfg["window"], device, reconstruction=True)
    threshold = normal_calibration_quantile(calibration_scores, cfg["public"]["calibration_quantile"])
    scores = predict_window_scores(model, test, cfg["window"], device, reconstruction=True)
    y = aligned_public_labels(labels, cfg["window"], len(scores))
    report = detector_metrics(y, scores, threshold=threshold, probabilities=False)
    report.update({"dataset": dataset, "unit": unit, "seed": seed,
                   "threshold_source": "chronological_training_calibration_quantile",
                   "calibration_fraction": cfg["public"]["calibration_fraction"],
                   "calibration_quantile": cfg["public"]["calibration_quantile"],
                   "label_alignment": "window_end_t_minus_1"})
    pd.DataFrame([report]).to_csv(target, index=False)
    (output / f"public_calibrated_{dataset}_{unit}_seed{seed}.json").write_text(json.dumps(report, indent=2, default=float))
    release_memory(model)


def attribution_for_method(method: str, model, current: np.ndarray, history: np.ndarray, cfg: dict, device: torch.device, kernel=None, lime=None):
    if method == "zero_ig": return core.zero_ig(model, current, cfg["ig_steps"], device)
    if method == "aw_ig": return core.aw_ig(model, np.vstack([current[0], current[:-1]]), current, cfg["ig_steps"], device)
    if method == "full_swing": return core.full_swing_current(model, history, cfg, device)
    flat = current.reshape(1, -1)
    if method == "kernel_shap": return np.asarray(kernel.shap_values(flat, nsamples=cfg["kernel_shap_samples"])).reshape(current.shape)
    explanation = lime.explain_instance(current.ravel(), lambda z: core.score_fn(model)(torch.as_tensor(z.reshape(-1, cfg["window"], current.shape[1]), dtype=torch.float32, device=device)).detach().cpu().numpy(), num_features=current.size)
    values = np.zeros(current.size); [values.__setitem__(i, value) for i, value in explanation.as_map()[1]]; return values.reshape(current.shape)
    raise ValueError(method)


def stage_method_latency(data_root: Path, output: Path, cfg: dict, device: torch.device, method: str):
    target = output / f"method_latency_{method}_seed42.csv"
    if target.exists(): print(f"[SKIP] {target.name}"); return
    if method not in METHODS: raise ValueError(method)
    _, _, _, parts, targets, scale, model = load_model(data_root, output, device, 42)
    test = parts[2]; labels = targets[2]; windows, y = core.windows(test, labels, cfg["window"])
    chosen = np.random.default_rng(42).choice(np.where(y == 1)[0], cfg["latency_windows"], replace=False)
    kernel = lime = None
    if method in {"kernel_shap", "lime"}:
        import shap
        from lime.lime_tabular import LimeTabularExplainer
        background = windows[:min(30, len(windows))].reshape(min(30, len(windows)), -1)
        if method == "kernel_shap":
            kernel = shap.KernelExplainer(lambda z: core.score_fn(model)(torch.as_tensor(z.reshape(-1, cfg["window"], test.shape[1]), dtype=torch.float32, device=device)).detach().cpu().numpy(), background)
        else: lime = LimeTabularExplainer(background, mode="regression", discretize_continuous=False, random_state=42)
    rows = []
    for idx in chosen:
        current = windows[idx]; history = np.concatenate([current[:1].repeat(cfg["window"]+cfg["swing_lag"]+1, 0), current], 0)[-(2*cfg["window"]+1+cfg["swing_lag"]):]
        attribution_for_method(method, model, current, history, cfg, device, kernel, lime)  # warm-up
        for trial in range(cfg["timing"]["repetitions"]):
            core.sync(); started = time.perf_counter(); attribution_for_method(method, model, current, history, cfg, device, kernel, lime); core.sync()
            rows.append({"method": method, "window_index": int(idx), "trial": trial, "latency_ms": (time.perf_counter()-started)*1000, "device": "GPU" if torch.cuda.is_available() else "CPU"})
    pd.DataFrame(rows).to_csv(target, index=False)
    release_memory(model)


def stage_timing(data_root: Path, output: Path, cfg: dict, device: torch.device, condition: str):
    target = output / f"timing_{condition}_trials.csv"
    if target.exists(): print(f"[SKIP] {target.name}"); return
    valid = {"unbatched_always_refresh": (True, True), "unbatched_adaptive_refresh": (True, False), "batched_always_refresh": (False, True), "batched_adaptive_refresh": (False, False)}
    if condition not in valid: raise ValueError(condition)
    serial, always = valid[condition]
    streams, _, _, _, _, scale, model = load_model(data_root, output, device, 42)
    stream = (streams["temperature"] - scale["mean"]) / scale["std"]
    probe = stream[-(cfg["timing"]["steps_per_trial"] + 2*cfg["window"] + cfg["swing_lag"] + 1):]
    for _ in range(cfg["timing"]["warmups"]): core.run_adaptive(probe[:min(len(probe), 80)], model, {**cfg, **cfg["temperature"], "always_refresh": always}, device, seed=42, serial_gradients=serial)
    rows = []
    for trial in range(cfg["timing"]["repetitions"]):
        records = core.run_adaptive(probe, model, {**cfg, **cfg["temperature"], "always_refresh": always}, device, seed=42+trial, serial_gradients=serial)
        rows.append({"condition": condition, "trial": trial, "attribution_evaluation": "serialized_fresh_calls" if serial else "batched_multipathig", "refresh_policy": "always" if always else "adaptive", "hardware": "GPU" if torch.cuda.is_available() else "CPU", **core.summarize(records)})
        release_memory(records)
    pd.DataFrame(rows).to_csv(target, index=False)
    release_memory(model)


def stage_posthoc(data_root: Path, output: Path, cfg: dict):
    """No-model analysis: paired faithfulness effects and tail latency distribution."""
    faith_path = output / "faithfulness_per_window.csv"
    if not faith_path.exists():
        raise FileNotFoundError("Run the common-window faithfulness stage first.")
    faith = pd.read_csv(faith_path)
    labels = np.load(data_root / "tier2_labels.npy").astype(int)
    test_labels = labels[int(.8 * len(labels)):]
    event_ids = positive_event_ids(test_labels[cfg["window"]:])
    indices = faith.window_index.unique().astype(int)
    event_map = {int(i): int(event_ids[i]) for i in indices if 0 <= i < len(event_ids)}
    clustered = len(event_map) == len(indices) and min(event_map.values(), default=-1) >= 0 and len(set(event_map.values())) >= 2
    paired_rows = []
    for baseline in ("aw_ig", "zero_ig", "kernel_shap", "lime"):
        result = paired_aopc_bootstrap(faith, candidate="full_swing", baseline=baseline,
                                       n_bootstrap=cfg["bootstrap"], seed=42,
                                       event_by_window=event_map if clustered else None)
        result["interpretation"] = "descriptive_paired_difference_not_universal_winner_claim"
        paired_rows.append(result)
    pd.DataFrame(paired_rows).to_csv(output / "paired_aopc_differences.csv", index=False)
    trials_path = output / "method_latency_trials.csv"
    if not trials_path.exists():
        raise FileNotFoundError("Run all five method-latency stages first.")
    latency_distribution(pd.read_csv(trials_path)).to_csv(output / "method_latency_tail_metrics.csv", index=False)
    print(f"[PASS] posthoc metrics saved; event_clustered_bootstrap={clustered}", flush=True)


def mean_sd(frame: pd.DataFrame, by: list[str], numeric: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(by, as_index=False)[numeric].agg(["mean", "std"])
    grouped.columns = ["_".join(x).strip("_") if isinstance(x, tuple) else x for x in grouped.columns]
    return grouped


def generate_paper_artifacts(output: Path) -> None:
    """Create every empirical table/figure from v1.9 result files only."""
    import matplotlib.pyplot as plt
    faith = pd.read_csv(output / "faithfulness_summary.csv")
    ablation = pd.read_csv(output / "ablation_summary.csv")
    scaling = pd.read_csv(output / "scaling_summary.csv")
    timing = pd.read_csv(output / "efficiency_2x2.csv")
    method_latency = pd.read_csv(output / "method_latency_summary.csv")
    public = pd.read_csv(output / "public_full_attribution_summary.csv")
    public_detection = pd.read_csv(output / "public_detection_summary.csv")
    synthetic_detection = pd.read_csv(output / "synthetic_extra_metrics_summary.csv")
    paired = pd.read_csv(output / "paired_aopc_differences.csv")
    delta_faith = pd.read_csv(output / "delta_faithfulness_summary.csv")
    tails = pd.read_csv(output / "method_latency_tail_metrics.csv")
    text = "# v1.9 Generated Paper Tables\n\n"
    for name, frame in [("Synthetic detector metrics", synthetic_detection), ("Current-score proxy faithfulness", faith), ("Direct prediction-difference faithfulness", delta_faith), ("Paired current-score AOPC differences", paired), ("Three-seed ablation", ablation), ("Repeated timing", timing), ("Method latency tails", tails), ("Three-seed scaling", scaling), ("Public threshold-free detection", public_detection), ("Public full-attribution subset", public)]:
        text += f"## {name}\n\n{frame.to_markdown(index=False)}\n\n"
    (output / "PAPER_TABLES_v1_9.md").write_text(text)
    plt.figure(figsize=(5.2, 3.2)); plt.errorbar(scaling["sensor_count"], scaling["amortized_step_latency_ms_mean_mean"], yerr=scaling["amortized_step_latency_ms_mean_std"], marker="o", capsize=3); plt.xlabel("Sensor count"); plt.ylabel("Amortized latency (ms)"); plt.tight_layout(); plt.savefig(output / "figure_scaling_latency_v1_9.png", dpi=200); plt.close()
    plt.figure(figsize=(5.2, 3.2)); plt.bar(faith.method, faith.aopc_mean, yerr=[faith.aopc_mean-faith.aopc_ci_low, faith.aopc_ci_high-faith.aopc_mean], capsize=3); plt.xticks(rotation=25, ha="right"); plt.ylabel("Mean-masking AOPC"); plt.tight_layout(); plt.savefig(output / "figure_common_window_faithfulness_v1_9.png", dpi=200); plt.close()
    plt.figure(figsize=(5.2, 3.2)); plt.bar(method_latency.method, method_latency.latency_ms_mean, yerr=method_latency.latency_ms_std, capsize=3); plt.xticks(rotation=25, ha="right"); plt.ylabel("Attribution latency (ms)"); plt.tight_layout(); plt.savefig(output / "figure_method_latency_v1_9.png", dpi=200); plt.close()
    tail = tails.set_index("method").reindex(method_latency.method)
    plt.figure(figsize=(5.2, 3.2)); plt.scatter(method_latency.latency_ms_mean, faith.aopc_mean, s=55)
    for method, x, y in zip(method_latency.method, method_latency.latency_ms_mean, faith.aopc_mean): plt.annotate(method, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=8)
    plt.xscale("log"); plt.xlabel("Mean T4 attribution latency (ms, log scale)"); plt.ylabel("Common-window AOPC"); plt.tight_layout(); plt.savefig(output / "figure_faithfulness_vs_latency_v1_9.png", dpi=200); plt.close()
    # Representative held-out timeline: fresh refreshes only are marked as explanations.
    records = core.load_controller_records(output / "controller_temperature_seed42.npz")
    labels = np.load(output.parent / "tier2_labels.npy") if (output.parent / "tier2_labels.npy").exists() else None
    target = np.asarray([r.target_index for r in records]); scores = np.asarray([r.current_score for r in records]); fresh = np.asarray([r.refreshed for r in records])
    if labels is not None:
        valid = (target >= 0) & (target < len(labels)); anomaly = np.zeros(len(target)); anomaly[valid] = labels[target[valid]]
        focus = np.flatnonzero(anomaly == 1); center = int(focus[0]) if len(focus) else min(100, len(target)-1); lo=max(0, center-60); hi=min(len(target), center+120)
        fig, ax = plt.subplots(figsize=(7.0, 3.2)); ax.plot(np.arange(lo, hi), scores[lo:hi], label="anomaly score")
        ax.fill_between(np.arange(lo, hi), 0, 1, where=anomaly[lo:hi] > 0, alpha=.16, transform=ax.get_xaxis_transform(), label="labelled anomaly")
        ax.scatter(np.flatnonzero(fresh[lo:hi]) + lo, scores[lo:hi][fresh[lo:hi]], marker="|", s=100, color="tab:red", label="fresh explanation")
        ax.set(xlabel="controller step", ylabel="score", title="Held-out detection and fresh-refresh timeline"); ax.legend(loc="best", fontsize=8); fig.tight_layout(); fig.savefig(output / "figure_detection_refresh_timeline_v1_9.png", dpi=200); plt.close(fig)
    # A fresh-only attribution heatmap, never a cached context.
    first_fresh = next(r for r in records if r.refreshed)
    fig, ax = plt.subplots(figsize=(6.0, 3.0)); im=ax.imshow(first_fresh.cached_event_context[-20:].T, aspect="auto", cmap="coolwarm")
    ax.set(xlabel="lag", ylabel="sensor", title="Representative fresh Full-SWING attribution map"); fig.colorbar(im, ax=ax, label="attribution"); fig.tight_layout(); fig.savefig(output / "figure_fresh_attribution_heatmap_v1_9.png", dpi=200); plt.close(fig)
    evidence = {
        "1_cached_context": "Cache hits are named cached_event_context; cache-fidelity is descriptive only and is never a current-change completeness claim.",
        "2_swing_identity": "Independent structural-reference and linear-completeness tests are mandatory. External-reference parity is reported as unavailable unless a pinned official fixture is supplied.",
        "3_completeness": "Only raw fresh SWING maps receive completeness residuals; I4 and cache reuse have no completeness claim.",
        "4_ablation_latency": "ablation_summary.csv separates fresh-call and amortized latency across three seeds.",
        "5_batching_vs_caching": "efficiency_2x2.csv contains repeated, warmed timing for all four batching x caching conditions.",
        "6_faithfulness": "faithfulness_summary.csv uses a common held-out window set, common masking protocol, and bootstrap intervals; it does not imply a winner absent supporting intervals.",
        "7_public_scope": "public_full_attribution_summary.csv exists only after each unit passes the completeness gate; public_policy_results.csv has exact matched budgets.",
        "8_numeric_consistency": "CONSISTENCY_REPORT_v1_9.json is generated from the same stage artifacts as all paper tables and figures.",
        "9_reproducibility": "run_manifest_v1_9.json records code-tree/data hashes, configuration, and hardware; checkpoints and per-stage output remain in the frozen bundle.",
        "10_presentation": "PAPER_TABLES_v1_9.md and the generated figures are the sole empirical sources for the next manuscript revision; the release checklist requires visual PDF review."
    }
    evidence.update({
        "11_public_alignment": "Public reconstruction labels use the final observation in each scored window, verified by regression tests and recorded as window_end_t_minus_1.",
        "12_imbalance_and_events": "Synthetic three-seed thresholded metrics include MCC, balanced accuracy, specificity, false-alarm and event-detection measures.",
        "13_refresh_coverage": "Refresh-event coverage uses explicit controller target_index values and separately reports normal-period refresh share.",
        "14_tail_latency": "method_latency_tail_metrics.csv records median/P95/P99/max from measured T4 trials.",
        "15_calibration_scope": "Primary public results are threshold-free; calibrated thresholded metrics are supplementary and use chronological non-test calibration."
    })
    (output / "PAPER_EVIDENCE_MAP_v1_9.json").write_text(json.dumps(evidence, indent=2))
    checklist = "\n".join(["# v1.1 release checklist from v1.9", "", "- [ ] Use only v1.9 generated numbers.", "- [ ] Label public reconstruction alignment as window_end_t_minus_1.", "- [ ] Label hardware from HARDWARE_PROFILE_v1_9.json.", "- [ ] Call cache hits cached_event_context, never current explanations.", "- [ ] Label score-change as offline oracle.", "- [ ] Do not claim Full-SWING wins faithfulness unless paired intervals support it.", "- [ ] State the documented public subset scope exactly.", "- [ ] Render and visually inspect the final PDF."])
    (output / "PAPER_RELEASE_CHECKLIST_v1_9.md").write_text(checklist)


def assemble(output: Path, cfg: dict, device: torch.device):
    required = []
    synthetic = []
    for seed in SEEDS:
        for stream in ("temperature", "power"):
            path = output / f"synthetic_{stream}_seed{seed}.json"; required.append(path)
            if path.exists(): synthetic.append(json.loads(path.read_text()))
    if len(synthetic) != 6: raise RuntimeError("Missing one or more three-seed synthetic stages.")
    pd.DataFrame(synthetic).to_csv(output / "synthetic_seed_results.csv", index=False)
    extra_paths = sorted(output.glob("synthetic_*_seed*_extra_metrics.csv"))
    if len(extra_paths) != 6: raise RuntimeError("All six synthetic extra-metric stages are required.")
    synthetic_extra = pd.concat([pd.read_csv(p) for p in extra_paths], ignore_index=True)
    if set(synthetic_extra.threshold_source) != {"fixed_probability_0.5", "validation_f1"}:
        raise RuntimeError("Synthetic fixed and validation threshold protocols are both required.")
    synthetic_extra.to_csv(output / "synthetic_extra_metrics_per_seed.csv", index=False)
    numeric_extra = [c for c in ["auroc", "auprc_average_precision", "mcc", "balanced_accuracy", "specificity", "false_positive_rate", "event_recall", "event_missed_count", "event_mean_delay_steps_detected_only", "false_alarm_episodes_per_1000_normal_steps"] if c in synthetic_extra]
    mean_sd(synthetic_extra, ["stream", "threshold_source"], numeric_extra).to_csv(output / "synthetic_extra_metrics_summary.csv", index=False)
    coverage_paths = sorted(output.glob("synthetic_*_seed*_refresh_coverage.json"))
    if len(coverage_paths) != 6: raise RuntimeError("All six synthetic refresh-coverage stages are required.")
    pd.DataFrame([json.loads(p.read_text()) for p in coverage_paths]).to_csv(output / "synthetic_refresh_coverage_per_seed.csv", index=False)
    ablations = [json.loads(p.read_text()) for p in output.glob("ablation_*_seed*.json")]
    if len(ablations) != 18: raise RuntimeError("All 18 seed × ablation stages are required.")
    ab = pd.DataFrame(ablations); ab.to_csv(output / "ablation_per_seed.csv", index=False)
    ablation_numeric = ["recompute_pct", "amortized_step_latency_ms_mean", "fresh_call_latency_ms_mean", "fresh_completeness_residual_mean", "aopc_fresh_only"]
    mean_sd(ab, ["configuration"], ablation_numeric).to_csv(output / "ablation_summary.csv", index=False)
    scales = [json.loads(p.read_text()) for p in output.glob("scaling_D*_seed*.json")]
    if len(scales) != 12: raise RuntimeError("All 12 seed × scaling stages are required.")
    sc = pd.DataFrame(scales); sc.to_csv(output / "scaling_per_seed.csv", index=False)
    scaling_numeric = ["recompute_pct", "amortized_step_latency_ms_mean", "fresh_call_latency_ms_mean", "fresh_completeness_residual_mean"]
    mean_sd(sc, ["sensor_count"], scaling_numeric).to_csv(output / "scaling_summary.csv", index=False)
    architectures = [json.loads(p.read_text()) for p in output.glob("architecture_seed*.json")]
    if len(architectures) != 3: raise RuntimeError("All three architecture stages are required.")
    (output / "architecture_validation.json").write_text(json.dumps(architectures, indent=2))
    policies = [json.loads(p.read_text()) for p in output.glob("policy_temperature_*_seed*.json")]
    if len(policies) != 12: raise RuntimeError("All four matched-budget synthetic policies across three seeds are required.")
    pd.DataFrame(policies).to_csv(output / "multi_policy_temperature.csv", index=False)
    timing = pd.concat([pd.read_csv(p) for p in output.glob("timing_*_trials.csv")], ignore_index=True)
    if timing.condition.nunique() != 4: raise RuntimeError("All four repeated timing conditions are required.")
    timing.to_csv(output / "efficiency_2x2_trials.csv", index=False)
    timing_numeric = ["recompute_pct", "amortized_step_latency_ms_mean", "fresh_call_latency_ms_mean", "fresh_completeness_residual_mean"]
    mean_sd(timing, ["condition", "attribution_evaluation", "refresh_policy", "hardware"], timing_numeric).to_csv(output / "efficiency_2x2.csv", index=False)
    method_latency = pd.concat([pd.read_csv(p) for p in output.glob("method_latency_*_seed42.csv")], ignore_index=True)
    if set(method_latency.method) != set(METHODS): raise RuntimeError("All five method-latency stages are required.")
    method_latency.to_csv(output / "method_latency_trials.csv", index=False)
    mean_sd(method_latency, ["method", "device"], ["latency_ms"]).to_csv(output / "method_latency_summary.csv", index=False)
    public_frames = [pd.read_csv(p) for p in output.glob("public_SMD_machine-*_seed*.csv")] + [pd.read_csv(p) for p in output.glob("public_PSM_PSM_seed*.csv")]
    public_summary_paths = [p for p in output.glob("public_SMD_machine-*_seed*.json") if not p.name.endswith("_policies.json")] + [p for p in output.glob("public_PSM_PSM_seed*.json") if not p.name.endswith("_policies.json")]
    public_summaries = [json.loads(p.read_text()) for p in public_summary_paths]
    if len(public_frames) != 4 or len(public_summaries) != 4: raise RuntimeError("Exactly the documented public SMD subset and PSM stage are required.")
    public = pd.concat(public_frames, ignore_index=True); public.to_csv(output / "public_full_attribution_subset.csv", index=False)
    policy_rows = [row for item in public_summaries for row in item["policy_rows"]]
    public_policy = pd.DataFrame(policy_rows)
    if not public_policy.exact_budget_match.all(): raise RuntimeError("Public matched-budget gate failed.")
    public_policy.to_csv(output / "public_policy_results.csv", index=False)
    public_summary = pd.DataFrame(public_summaries)
    if not public_summary.completeness_gate_pass.all(): raise RuntimeError("Public full-SWING completeness gate failed; do not freeze.")
    public_summary.to_csv(output / "public_full_attribution_summary.csv", index=False)
    public_detection_paths = sorted(output.glob("public_detection_*_seed*.csv"))
    if len(public_detection_paths) != 4: raise RuntimeError("Aligned threshold-free public detection metrics are required.")
    public_detection = pd.concat([pd.read_csv(p) for p in public_detection_paths], ignore_index=True)
    if not set(public_detection.label_alignment).issubset({"window_end_t_minus_1"}): raise RuntimeError("Public detection label alignment gate failed.")
    public_detection.to_csv(output / "public_detection_summary.csv", index=False)
    calibrated_paths = sorted(output.glob("public_calibrated_*_seed*.csv"))
    if len(calibrated_paths) != 4: raise RuntimeError("Supplementary calibrated public metric stages are required.")
    pd.concat([pd.read_csv(p) for p in calibrated_paths], ignore_index=True).to_csv(output / "public_calibrated_metrics_summary.csv", index=False)
    if not (output / "faithfulness_summary.csv").exists(): raise RuntimeError("Current-score proxy faithfulness stage missing.")
    if not (output / "delta_faithfulness_summary.csv").exists(): raise RuntimeError("Direct prediction-difference faithfulness stage missing.")
    if not (output / 'delta_faithfulness_paired.csv').exists(): raise RuntimeError('Direct paired prediction-difference comparisons missing.')
    if not (output / "paired_aopc_differences.csv").exists() or not (output / "method_latency_tail_metrics.csv").exists(): raise RuntimeError("Posthoc paired AOPC and latency-tail stages are required.")
    profile = hardware_profile()
    if not profile["cuda_available"]: raise RuntimeError("Cannot assemble official v1.9 timing results without CUDA.")
    report = {"status": "PASS", "three_seed_synthetic": True, "three_seed_synthetic_extra_metrics": True, "three_seed_refresh_coverage": True,
              "three_seed_ablation": True, "three_seed_scaling": True, "three_seed_architecture": True,
              "public_alignment_gate": True, "public_exact_budget_match": True, "public_completeness_gate": True, "public_threshold_free_metrics": True,
              "supplementary_public_calibration": True, "paired_aopc": True, "direct_prediction_difference_faithfulness": True, "latency_tails": True, "hardware": profile,
              "full_swing_linear_completeness": linear_completeness_check(str(device))}
    generate_paper_artifacts(output)
    (output / "CONSISTENCY_REPORT_v1_9.json").write_text(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True); parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage", required=True, choices=["preflight", "synthetic", "synthetic_metrics", "faithfulness", "delta_faithfulness", "policy", "ablation", "scaling", "architecture", "public", "public_calibrated", "timing", "method_latency", "posthoc", "assemble"])
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--stream", choices=["temperature", "power"], default="temperature")
    parser.add_argument("--variant"); parser.add_argument("--dimensions", type=int); parser.add_argument("--policy", choices=["adaptive", "periodic", "random", "score_change"])
    parser.add_argument("--condition"); parser.add_argument("--method", choices=METHODS); parser.add_argument("--dataset", choices=["SMD", "PSM"]); parser.add_argument("--unit")
    parser.add_argument("--allow-cpu-smoke", action="store_true", help="Permit only preflight development on CPU; official stages still reject CPU at assembly.")
    args = parser.parse_args(); cfg, device = init(args.data_root, args.output_root, require_cuda=not args.allow_cpu_smoke); started = time.perf_counter()
    print(f"[START] stage={args.stage} device={device} seed={args.seed}", flush=True)
    if args.stage == "preflight": stage_preflight(args.data_root, args.output_root, cfg, device)
    elif args.stage == "synthetic": stage_synthetic(args.data_root, args.output_root, cfg, device, args.seed, args.stream)
    elif args.stage == "synthetic_metrics": stage_synthetic_metrics(args.data_root, args.output_root, cfg, device, args.seed, args.stream)
    elif args.stage == "faithfulness": stage_faithfulness(args.data_root, args.output_root, cfg, device)
    elif args.stage == "delta_faithfulness": stage_delta_faithfulness(args.data_root, args.output_root, cfg, device)
    elif args.stage == "policy": stage_policy(args.data_root, args.output_root, cfg, device, args.policy, args.seed)
    elif args.stage == "ablation": stage_ablation(args.data_root, args.output_root, cfg, device, args.seed, args.variant)
    elif args.stage == "scaling": stage_scaling(args.data_root, args.output_root, cfg, device, args.seed, args.dimensions)
    elif args.stage == "architecture": stage_architecture(args.data_root, args.output_root, cfg, device, args.seed)
    elif args.stage == "public": stage_public_unit(args.data_root, args.output_root, cfg, device, args.dataset, args.unit, args.seed)
    elif args.stage == "public_calibrated": stage_public_calibrated(args.data_root, args.output_root, cfg, device, args.dataset, args.unit, args.seed)
    elif args.stage == "timing": stage_timing(args.data_root, args.output_root, cfg, device, args.condition)
    elif args.stage == "method_latency": stage_method_latency(args.data_root, args.output_root, cfg, device, args.method)
    elif args.stage == "posthoc": stage_posthoc(args.data_root, args.output_root, cfg)
    else: assemble(args.output_root, cfg, device)
    print(f"[DONE] stage={args.stage} elapsed_min={(time.perf_counter()-started)/60:.2f}", flush=True)


if __name__ == "__main__":
    main()
