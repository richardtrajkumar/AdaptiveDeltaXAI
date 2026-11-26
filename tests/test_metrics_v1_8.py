"""Lightweight synthetic tests; do not require real SMD/PSM data."""
import numpy as np
import pandas as pd
import pytest

from adaptive_deltaxai.metrics_v1_8 import (
    aligned_public_labels, contiguous_events, detector_metrics,
    event_detection_metrics, latency_distribution, paired_aopc_bootstrap,
    refresh_coverage_metrics, validation_f1_threshold, normal_calibration_quantile,
)


def test_classification_metrics_hand_computed():
    y = [0, 0, 1, 1, 0, 1]
    scores = [0.1, 0.8, 0.9, 0.7, 0.2, 0.2]
    out = detector_metrics(y, scores, threshold=0.5, probabilities=True)
    assert (out['tn'], out['fp'], out['fn'], out['tp']) == (2, 1, 1, 2)
    assert np.isclose(out['specificity'], 2/3)
    assert np.isclose(out['balanced_accuracy'], 2/3)
    assert np.isclose(out['mcc'], 1/3)
    assert np.isclose(out['false_positives_per_1000_normal_steps'], 1000/3)
    assert out['event_count'] == 2
    assert out['event_detected_count'] == 1
    assert out['event_missed_count'] == 1
    assert out['false_alarm_episode_count'] == 0  # FP inside an event-spanning alert run
    assert np.isfinite(out['brier_score'])


def test_public_alignment():
    labels = [0, 0, 1, 1, 0, 0]
    assert aligned_public_labels(labels, window=3, n_scores=3).tolist() == [1, 1, 0]
    with pytest.raises(ValueError):
        aligned_public_labels(labels, 7, 1)


def test_event_and_refresh_metrics():
    y = np.array([0,1,1,1,0,0,1,1,0], dtype=int)
    predicted = np.array([0,0,1,1,0,0,0,0,1], dtype=int)
    metrics = event_detection_metrics(y, predicted)
    assert metrics['event_count'] == 2
    assert metrics['event_detected_count'] == 1
    assert metrics['event_mean_delay_steps_detected_only'] == 1
    assert metrics['false_alarm_episode_count'] == 1
    refreshed = np.array([1,0,1,0,0,0,1,0,0])
    coverage = refresh_coverage_metrics(y, refreshed)
    assert coverage['event_refresh_coverage'] == 1.0
    assert coverage['n_refreshes'] == 3


def test_paired_ci_and_latency():
    rows = []
    for w in range(20):
        rows.extend([dict(method='full_swing', window_index=w, aopc=3+w/100),
                     dict(method='aw_ig', window_index=w, aopc=2+w/100)])
    frame = pd.DataFrame(rows)
    c = paired_aopc_bootstrap(frame, n_bootstrap=200, event_by_window={w:w//5 for w in range(20)})
    assert c['n_event_clusters'] == 4
    assert np.isclose(c['mean_aopc_delta_candidate_minus_baseline'], 1)
    assert np.isclose(c['delta_ci_95_low'],1)
    df = pd.DataFrame({'method':['full_swing']*5, 'device':['CPU']*5, 'latency_ms':[10,20,30,40,50]})
    results = latency_distribution(df).iloc[0]
    assert results.latency_median_ms == 30
    assert results.latency_max_ms == 50


def test_validation_threshold_only():
    y = [0,0,1,1]; s = [0.1,0.2,0.8,0.9]
    assert np.isclose(validation_f1_threshold(y,s), .8)
    assert np.isclose(normal_calibration_quantile([1,2,3,4],quantile=.5),2.5)
    with pytest.raises(ValueError):
        detector_metrics([1,0],[0.1,0.9],threshold=float('nan'))
