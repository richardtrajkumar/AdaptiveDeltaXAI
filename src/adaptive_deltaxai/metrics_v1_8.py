"""Additional evaluation metrics for AdaptiveDeltaXAI Master v1.9.

This module is the canonical v1.9 metrics implementation.  It never modifies
the frozen v1.7 results.
Evaluation conventions:
* Synthetic supervised detector: windows x[t-w:t] predict label y[t]
  (one-step-ahead classification), as in ``core_v1_9.windows``.
* Public reconstruction detector: windows x[t-w:t] reconstruct x[t-1].
  Score labels therefore align to y[t-1], NOT y[t].
* A score threshold must be fixed without consulting test labels.
* Cached event contexts are not fresh explanations of current predictions.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _binary_vector(y: Any, label: str = "labels") -> np.ndarray:
    arr = np.asarray(y).reshape(-1)
    if arr.size == 0 or not np.isin(arr, [0, 1]).all():
        raise ValueError(f"{label} must be a nonempty binary 0/1 vector")
    return arr.astype(np.int64)


def _numeric_vector(values: Any, name: str) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    if a.size == 0 or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a nonempty finite numeric vector")
    return a


def contiguous_events(mask: Any) -> list[tuple[int, int]]:
    """Return half-open [start, stop) intervals for runs of positive values."""
    y = _binary_vector(mask, 'mask')
    edges = np.diff(np.r_[0, y, 0]); starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def positive_event_ids(mask: Any) -> np.ndarray:
    """Return event number for positive timesteps, -1 for normal ones."""
    y = _binary_vector(mask)
    ids = np.full(len(y), -1, dtype=np.int64)
    for event_id, (start, end) in enumerate(contiguous_events(y)):
        ids[start:end] = event_id
    return ids


def event_detection_metrics(y_true: Any, predicted: Any, *, seconds_per_step: float | None = None) -> dict:
    """Unadjusted event overlap metrics, without point adjustment.

    A true event is detected if it contains >=1 positive prediction; delay is
    the first within-event positive minus the start of that event. Delays are
    summarized ONLY for detected events; missed count is always reported.
    False alarm episodes are predicted positive runs not overlapping any event.
    This is not a one-to-one event matching protocol for event-level precision.
    """
    y = _binary_vector(y_true); p = _binary_vector(predicted, 'predicted')
    if len(y) != len(p):
        raise ValueError('y_true and predicted must have equal length')
    actual_events = contiguous_events(y)
    alert_events = contiguous_events(p)
    delays = []
    for start, end in actual_events:
        within = np.flatnonzero(p[start:end])
        if within.size:
            delays.append(int(within[0]))
    false_episodes = sum(not np.any(y[start:end]) for start, end in alert_events)
    normal_count = int(np.sum(y == 0))
    result = {
        'event_count': len(actual_events),
        'event_detected_count': len(delays),
        'event_missed_count': len(actual_events) - len(delays),
        'event_recall': float(len(delays) / len(actual_events)) if actual_events else float('nan'),
        'event_mean_delay_steps_detected_only': float(np.mean(delays)) if delays else float('nan'),
        'event_median_delay_steps_detected_only': float(np.median(delays)) if delays else float('nan'),
        'alert_episode_count': len(alert_events),
        'false_alarm_episode_count': false_episodes,
        'false_alarm_episodes_per_1000_normal_steps':
            float(1000 * false_episodes / normal_count) if normal_count else float('nan'),
    }
    if seconds_per_step is not None:
        if seconds_per_step <= 0 or not np.isfinite(seconds_per_step):
            raise ValueError('seconds_per_step must be positive and finite')
        result['event_mean_delay_seconds_detected_only'] = (
            float(result['event_mean_delay_steps_detected_only'] * seconds_per_step)
            if delays else float('nan')
        )
    return result


def expected_calibration_error(y_true: Any, probabilities: Any, bins: int = 10) -> float:
    """Equal-width-bin ECE, a descriptive diagnostic, not a calibrated CI."""
    y = _binary_vector(y_true); p = _numeric_vector(probabilities, 'probabilities')
    if len(y) != len(p) or not np.all((p >= 0) & (p <= 1)):
        raise ValueError('Probabilities must align with labels and lie within [0,1]')
    if bins < 2:
        raise ValueError('bins must be >=2')
    b = np.minimum(np.floor(p * bins).astype(int), bins - 1)
    ece = 0.0
    for idx in range(bins):
        which = b == idx
        if np.any(which):
            ece += which.mean() * abs(float(y[which].mean()) - float(p[which].mean()))
    return float(ece)


def detector_metrics(
    y_true: Any,
    scores: Any,
    *,
    threshold: float | None = None,
    probabilities: bool = False,
    seconds_per_step: float | None = None,
) -> dict:
    """Threshold-free and, if provided, thresholded detection metrics.

    No threshold is chosen from test labels. Higher scores mean more anomalous.
    Public reconstruction scores are not probabilities; leave probabilities=False.
    """
    y = _binary_vector(y_true)
    s = _numeric_vector(scores, 'scores')
    if len(y) != len(s):
        raise ValueError(f'y_true ({len(y)}) and scores ({len(s)}) lengths differ')
    if probabilities and not np.all((0 <= s) & (s <= 1)):
        raise ValueError('probabilities=True requires scores in [0,1]')
    n_pos = int(y.sum()); n_neg = int(len(y) - n_pos)
    out = {
        'n_steps': int(len(y)), 'n_positive': n_pos, 'n_negative': n_neg,
        'positive_prevalence': float(n_pos / len(y)),
        'auroc': float(roc_auc_score(y, s)) if n_pos and n_neg else float('nan'),
        'auprc_average_precision': float(average_precision_score(y, s)) if n_pos else float('nan'),
    }
    out['auprc_lift_over_prevalence'] = (
        float(out['auprc_average_precision'] / out['positive_prevalence'])
        if n_pos else float('nan')
    )
    if probabilities:
        out['brier_score'] = float(brier_score_loss(y, s))
        out['ece_10_bins'] = expected_calibration_error(y, s, 10)
    if threshold is not None:
        if not np.isfinite(threshold):
            raise ValueError('threshold must be finite')
        h = (s >= float(threshold)).astype(np.int64)
        tn, fp, fn, tp = confusion_matrix(y, h, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if tn + fp else float('nan')
        sensitivity = tp / (tp + fn) if tp + fn else float('nan')
        out.update({
            'threshold': float(threshold),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'precision': float(precision_score(y, h, zero_division=0)),
            'recall': float(recall_score(y, h, zero_division=0)),
            'f1': float(f1_score(y, h, zero_division=0)),
            'specificity': float(specificity),
            'balanced_accuracy': float((specificity + sensitivity) / 2) if n_pos and n_neg else float('nan'),
            'mcc': float(matthews_corrcoef(y, h)),
            'false_positive_rate': float(fp / n_neg) if n_neg else float('nan'),
            'false_positives_per_1000_normal_steps': float(1000 * fp / n_neg) if n_neg else float('nan'),
        })
        out.update(event_detection_metrics(y, h, seconds_per_step=seconds_per_step))
    return out


def validation_f1_threshold(y_validation: Any, validation_scores: Any) -> float:
    """Select threshold using validation set ONLY (never test labels)."""
    y = _binary_vector(y_validation)
    s = _numeric_vector(validation_scores, 'validation_scores')
    if len(y) != len(s) or y.min() == y.max():
        raise ValueError('Validation labels/scores must align and include both classes')
    precision, recall, thresholds = precision_recall_curve(y, s)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def normal_calibration_quantile(normal_calibration_scores: Any, quantile: float = 0.995) -> float:
    """Unsupervised threshold from held-out normal calibration data only."""
    s = _numeric_vector(normal_calibration_scores, 'normal_calibration_scores')
    if not 0.0 < quantile < 1.0:
        raise ValueError('quantile must be between 0 and 1')
    return float(np.quantile(s, quantile))


def aligned_public_labels(labels: Any, window: int, n_scores: int) -> np.ndarray:
    """Align reconstruction of x[t-window:t]'s last point x[t-1] to y[t-1]."""
    y = _binary_vector(labels)
    if window < 1 or n_scores < 1 or (window - 1 + n_scores) > len(y):
        raise ValueError('Invalid public window or score count')
    return y[window - 1:window - 1 + n_scores]


def refresh_coverage_metrics(y_aligned: Any, refreshed: Any) -> dict:
    """Report when a refresh occurs during events, NOT detector accuracy.

    The two arrays must describe the same evaluation timeline and target index.
    """
    y = _binary_vector(y_aligned)
    m = _binary_vector(refreshed, 'refreshed')
    if len(y) != len(m):
        raise ValueError('Refresh mask and aligned labels must have equal length')
    events = contiguous_events(y)
    delays = []
    for start, end in events:
        offsets = np.flatnonzero(m[start:end])
        if offsets.size:
            delays.append(int(offsets[0]))
    return {
        'n_steps': len(y),
        'n_refreshes': int(m.sum()),
        'recompute_pct': float(100 * m.mean()),
        'n_true_anomaly_events': len(events),
        'events_with_refresh_count': len(delays),
        'event_refresh_coverage': float(len(delays) / len(events)) if events else float('nan'),
        'mean_steps_to_first_refresh_detected_events': float(np.mean(delays)) if delays else float('nan'),
        'refresh_on_anomaly_step_pct_of_all_refreshes': float(100 * (m & y).sum() / m.sum()) if m.sum() else float('nan'),
        'refresh_on_normal_step_pct_of_all_refreshes': float(100 * (m & (1 - y)).sum() / m.sum()) if m.sum() else float('nan'),
    }


def paired_aopc_bootstrap(
    faithfulness_per_window: pd.DataFrame,
    *,
    candidate: str = 'full_swing',
    baseline: str = 'aw_ig',
    n_bootstrap: int = 2000,
    seed: int = 42,
    event_by_window: dict[int, int] | None = None,
) -> dict:
    """Paired AOPC difference CI, optionally clustered by positive anomaly event.

    Each common window must appear exactly once per method. When event IDs are
    supplied, resample entire anomaly events to reduce overlap dependence.
    Without event IDs, inference is exploratory: nearby windows may be correlated.
    """
    table = faithfulness_per_window[['method', 'window_index', 'aopc']].copy()
    sub = table[table.method.isin([candidate, baseline])]
    if sub.duplicated(['method', 'window_index']).any():
        raise ValueError('Duplicate method-window pair')
    pivot = sub.pivot(index='window_index', columns='method', values='aopc')
    if candidate not in pivot or baseline not in pivot or pivot.isna().any().any():
        raise ValueError('Both methods must be evaluated on exactly the same windows')
    pivot = pivot.sort_index(); diff = (pivot[candidate] - pivot[baseline]).to_numpy(dtype=float)
    if not np.isfinite(diff).all() or n_bootstrap < 100:
        raise ValueError('Finite AOPC values and at least 100 bootstrap draws required')
    rng = np.random.default_rng(seed)
    if event_by_window is None:
        samples = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(n_bootstrap)])
        method = 'paired_window_bootstrap_exploratory'
        n_clusters = None
    else:
        event_ids = np.asarray([event_by_window[int(i)] for i in pivot.index])
        if (event_ids < 0).any():
            raise ValueError('All selected windows require nonnegative event IDs')
        unique_events = np.unique(event_ids)
        if len(unique_events) < 2:
            raise ValueError('At least two distinct anomaly events required for event bootstrap')
        event_groups = [diff[event_ids == e] for e in unique_events]
        samples = []
        for _ in range(n_bootstrap):
            selected = rng.integers(0, len(event_groups), len(event_groups))
            merged = np.concatenate([event_groups[i] for i in selected])
            samples.append(float(np.mean(merged)))
        samples = np.asarray(samples)
        method = 'paired_event_cluster_bootstrap'
        n_clusters = len(unique_events)
    return {
        'candidate': candidate, 'baseline': baseline,
        'n_paired_windows': int(len(diff)), 'n_event_clusters': n_clusters,
        'candidate_aopc_mean': float(pivot[candidate].mean()),
        'baseline_aopc_mean': float(pivot[baseline].mean()),
        'mean_aopc_delta_candidate_minus_baseline': float(diff.mean()),
        'delta_ci_95_low': float(np.quantile(samples, .025)),
        'delta_ci_95_high': float(np.quantile(samples, .975)),
        'bootstrap_method': method,
    }


def latency_distribution(method_latency_trials: pd.DataFrame) -> pd.DataFrame:
    """From actual per-method trial latencies; does NOT pool batch timing rows."""
    required = {'method', 'device', 'latency_ms'}
    if not required.issubset(method_latency_trials.columns):
        raise ValueError(f'Latency trials must contain {sorted(required)}')
    rows = []
    for (method, device), group in method_latency_trials.groupby(['method', 'device']):
        values = _numeric_vector(group.latency_ms, 'latency_ms')
        rows.append({'method': method, 'device': device, 'n_trials': len(values),
                     'latency_mean_ms': float(values.mean()),
                     'latency_median_ms': float(np.median(values)),
                     'latency_p95_ms': float(np.percentile(values, 95)),
                     'latency_p99_ms': float(np.percentile(values, 99)),
                     'latency_max_ms': float(values.max())})
    return pd.DataFrame(rows)
