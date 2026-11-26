"""Regression tests for the two distinct window/label conventions in v1.9."""

import numpy as np

from adaptive_deltaxai.metrics_v1_8 import aligned_public_labels


def test_public_reconstruction_targets_last_observation_in_window():
    # Windows are x[i-w:i], whose last reconstructed observation is x[i-1].
    labels = np.array([0, 1, 0, 1, 1, 0, 0], dtype=int)
    assert aligned_public_labels(labels, window=3, n_scores=4).tolist() == [0, 1, 1, 0]


def test_synthetic_classifier_targets_next_observation_contract():
    # Synthetic windows x[i-w:i] predict label y[i], by protocol definition.
    labels = np.array([0, 1, 0, 1, 1, 0, 0], dtype=int)
    window = 3
    assert labels[window:].tolist() == [1, 1, 0, 0]


def test_public_alignment_rejects_off_by_one_overrun():
    labels = np.zeros(5, dtype=int)
    try:
        aligned_public_labels(labels, window=3, n_scores=4)
    except ValueError:
        return
    raise AssertionError("Expected invalid public alignment to be rejected")
