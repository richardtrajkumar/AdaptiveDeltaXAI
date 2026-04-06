"""
evaluation.py
=============
Fidelity and faithfulness metrics for AdaptiveDeltaXAI.

Metrics implemented:
    aopc()                  — Area Over the Perturbation Curve
    per_sensor_spearman()   — Per-sensor rank correlation (fidelity)
    trigger_precision()     — Trigger alignment with anomaly windows
    latency_summary()       — Latency statistics
    print_comparison()      — Formatted comparison table
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from scipy.stats import spearmanr


def aopc(
    model_fn: Callable,
    W_curr: np.ndarray,
    E: np.ndarray,
    k_max: Optional[int] = None,
    mask_val: float = 0.0,
) -> float:
    """
    Area Over the Perturbation Curve (AOPC).

    The paper's primary faithfulness metric. Procedure:
      1. Rank (lag, feature) pairs by |E| in descending order
      2. Mask top-k features one by one (set to mask_val)
      3. Measure prediction degradation at each step
      4. Return mean degradation = AOPC score

    Higher AOPC = more faithful explanation (masking important
    features causes larger prediction drops).

    Args:
        model_fn:  Callable (1, window, features) → float
        W_curr:    (window, features) current window matrix
        E:         (window, features) attribution matrix
        k_max:     Number of features to mask. Default: 25% of total.
        mask_val:  Value used for masking. Default: 0 (mean-imputation
                   is better in practice — replace with feature mean).

    Returns:
        AOPC score (float). Positive = faithful, near 0 = uninformative.

    Note:
        Requires a trained model to produce meaningful scores.
        With random-weight models, AOPC ≈ 0 (model ignores input).
    """
    if k_max is None:
        k_max = max(1, W_curr.shape[0] * W_curr.shape[1] // 4)

    baseline_pred = model_fn(W_curr[np.newaxis])
    flat_E        = np.abs(E).flatten()
    ranked        = np.argsort(-flat_E)   # descending

    W_masked      = W_curr.copy()
    degradations  = []

    for k in range(1, k_max + 1):
        idx      = ranked[k - 1]
        row, col = np.unravel_index(idx, E.shape)
        W_masked[row, col] = mask_val
        degradations.append(baseline_pred - model_fn(W_masked[np.newaxis]))

    return float(np.mean(degradations))


def per_sensor_spearman(
    E_adaptive: np.ndarray,
    E_vanilla: np.ndarray,
) -> float:
    """
    Per-sensor Spearman rank correlation between adaptive and vanilla attributions.

    Correct fidelity metric when comparing cached (adaptive) vs fresh (vanilla)
    explanations. Works for any number of features >= 2.

    Procedure:
      1. Collapse window dimension: take mean absolute attribution per feature
      2. Rank both vectors
      3. Compute Spearman ρ between ranks

    Args:
        E_adaptive: (window, n_features) — cached/adaptive attribution
        E_vanilla:  (window, n_features) — ground-truth vanilla attribution

    Returns:
        Spearman ρ in [-1, 1]. Near 1.0 = high fidelity.
        Returns NaN if either vector is all-zeros (degenerate).

    Note:
        Do NOT use group-level correlation with only 2 groups — this
        gives degenerate ±1 results that average to ~0 across timesteps.
        Use this function instead (per feature, n >= 8).
    """
    a = np.abs(E_adaptive).mean(axis=0)   # (n_features,)
    b = np.abs(E_vanilla).mean(axis=0)    # (n_features,)

    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")   # degenerate — all-zero attribution

    rho, _ = spearmanr(a, b)
    return float(rho)


def trigger_precision(
    results: List,
    anomaly_labels: np.ndarray,
    tolerance_steps: int = 2,
) -> Dict:
    """
    Measure how well the adaptive trigger aligns with true anomaly windows.

    A triggered step is a TRUE POSITIVE if it falls within tolerance_steps
    of an anomaly-labelled step.

    Args:
        results:          List of ExplanationResult objects
        anomaly_labels:   (n_steps,) int array, 1 = anomaly
        tolerance_steps:  How many steps early/late a trigger is still counted

    Returns:
        dict with precision, recall, f1, tp, fp, fn counts
    """
    trigger_steps = set(r.t - 1 for r in results if r.triggered)
    anomaly_steps = set(int(i) for i, a in enumerate(anomaly_labels) if a == 1)

    # Expand anomaly steps by tolerance window
    anomaly_expanded = set()
    for s in anomaly_steps:
        for offset in range(-tolerance_steps, tolerance_steps + 1):
            if 0 <= s + offset < len(anomaly_labels):
                anomaly_expanded.add(s + offset)

    tp = len(trigger_steps & anomaly_expanded)
    fp = len(trigger_steps - anomaly_expanded)
    fn = len(anomaly_steps - trigger_steps)

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "precision":        precision,
        "recall":           recall,
        "f1":               f1,
        "true_positives":   tp,
        "false_positives":  fp,
        "false_negatives":  fn,
        "n_triggers":       len(trigger_steps),
        "n_anomaly_steps":  len(anomaly_steps),
    }


def latency_summary(latencies: List[float]) -> Dict:
    """Compute latency statistics from a list of per-step ms values."""
    a = np.array(latencies)
    return {
        "mean_ms":   float(np.mean(a)),
        "median_ms": float(np.median(a)),
        "p95_ms":    float(np.percentile(a, 95)),
        "p99_ms":    float(np.percentile(a, 99)),
        "max_ms":    float(np.max(a)),
        "total_s":   float(np.sum(a) / 1000),
    }


def print_comparison(
    label_a: str,
    label_b: str,
    metrics_a: Dict,
    metrics_b: Dict,
    speedup: Optional[float] = None,
):
    """Print a formatted side-by-side comparison table."""
    print(f"\n{'='*68}")
    print(f"{'Metric':<35} {label_a:>14} {label_b:>14}")
    print(f"{'-'*68}")

    shared_keys = [
        ("Recompute rate (%)",    "recompute_pct",   ".1f"),
        ("Mean latency (ms)",     "lat_mean_ms",     ".2f"),
        ("P95 latency (ms)",      "lat_p95_ms",      ".2f"),
        ("P99 latency (ms)",      "lat_p99_ms",      ".2f"),
        ("Mean cache age (steps)","cache_age_mean",  ".1f"),
        ("Max cache age (steps)", "cache_age_max",   "d"),
        ("AOPC (mean)",           "aopc_mean",       ".4f"),
        ("Spearman ρ (mean)",     "rho_mean",        ".3f"),
    ]

    for display, key, fmt in shared_keys:
        va = metrics_a.get(key, float("nan"))
        vb = metrics_b.get(key, float("nan"))
        try:
            sa = f"{va:{fmt}}"
            sb = f"{vb:{fmt}}"
        except (TypeError, ValueError):
            sa = str(va)
            sb = str(vb)
        print(f"  {display:<33} {sa:>14} {sb:>14}")

    if speedup is not None:
        print(f"  {'Speedup vs baseline':<33} {'1.0×':>14} {speedup:.0f}×{' ':>10}")

    print(f"{'='*68}\n")
