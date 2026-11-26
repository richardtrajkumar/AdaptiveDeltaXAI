"""Independent structural fixture for the published SWING path construction.

The fixture deliberately does not call ``swing_attribution``.  It reconstructs
the four official terms (current/previous × far/near) and their combination
from the published path procedure.  The production implementation must match
it on a deterministic linear model before a v1.6 result can be frozen.
"""

import torch

from adaptive_deltaxai.swing import multipath_integrated_gradients, swing_attribution


def official_reference(score, history, w=20, td=1, lag=5, samples=32):
    current = history[:, -w:, :]
    previous = history[:, -w-td:-td, :]
    far = history[:, -w-td-lag:-td-lag, :]
    near = history[:, -w-lag:-lag, :]
    current_path = [history[:, -w-offset:-offset, :] for offset in range(td, 0, -1)]
    previous_path = [history[:, -w-offset:-offset, :] for offset in range(2, td)]
    cur_far = multipath_integrated_gradients(score, current, far, current_path, samples)
    prev_far = multipath_integrated_gradients(score, previous, far, None, samples)
    cur_near = multipath_integrated_gradients(score, current, near, None, samples)
    prev_near = multipath_integrated_gradients(score, previous, near, previous_path, samples)
    out = torch.zeros_like(history)
    # Official composition is additive over the two overlapping windows.
    out[:, -w-td:-td, :] += -(prev_far + prev_near) / 2
    out[:, -w:, :] += (cur_far + cur_near) / 2
    return out


def test_production_swing_matches_official_path_reference():
    torch.manual_seed(123)
    history = torch.randn(2, 26, 3)
    weight = torch.tensor([0.3, -0.2, 0.7])
    def score(x): return (x[:, -1, :] * weight).sum(dim=1)
    expected = official_reference(score, history)
    actual = swing_attribution(score, history, window=20, prediction_delta=1,
                               retrospective_lag=5, n_samples=32)
    assert torch.allclose(actual, expected, atol=1e-6)
