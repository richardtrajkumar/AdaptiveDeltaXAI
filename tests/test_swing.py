"""Run with: PYTHONPATH=src pytest -q tests"""

import torch

from adaptive_deltaxai.swing import linear_completeness_check, swing_attribution


def test_linear_swing_completeness():
    assert linear_completeness_check() < 1e-5


def test_swing_returns_full_history_shape():
    history = torch.randn(2, 26, 3)
    def score(x):
        return x[:, -1, :].sum(dim=1)
    attr = swing_attribution(score, history, window=20, prediction_delta=1,
                             retrospective_lag=5, n_samples=16)
    assert attr.shape == history.shape


def test_official_path_composition_for_adjacent_windows():
    """A structural test of the four official SWING path terms.

    With prediction_delta=1 the official construction has an empty old-path
    list and one interpolating new-path window.  A linear score makes each
    path integral exact, so the combined near/far composition must recover
    the prediction difference.  This detects accidental replacement by
    zero-IG or adjacent-window IG.
    """
    history = torch.arange(26 * 2, dtype=torch.float32).reshape(1, 26, 2) / 10
    weight = torch.tensor([0.25, -0.15])
    def score(x):
        return (x[:, -1] * weight).sum(dim=1)
    attr = swing_attribution(score, history, window=20, prediction_delta=1,
                             retrospective_lag=5, n_samples=32, overall_path="combined")
    expected = score(history[:, -20:]) - score(history[:, -21:-1])
    assert torch.allclose(attr.sum(dim=(1, 2)), expected, atol=1e-5)
    # The baseline-only prefix is not an input to either explained window.
    assert torch.count_nonzero(attr[:, :5]) == 0


def test_exact_history_contract_is_26_observations():
    def score(x): return x[:, -1, :].sum(dim=1)
    with torch.no_grad():
        valid = torch.randn(1, 26, 2)
        short = torch.randn(1, 25, 2)
    assert swing_attribution(score, valid, 20, 1, 5, 8).shape == valid.shape
    try:
        swing_attribution(score, short, 20, 1, 5, 8)
    except ValueError as exc:
        assert "requires at least 26" in str(exc)
    else:
        raise AssertionError("25 observations must fail the exact history gate")
