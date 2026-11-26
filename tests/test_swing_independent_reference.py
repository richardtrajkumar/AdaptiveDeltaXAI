"""Independent reference-path checks for the v1.7 SWING implementation.

This fixture intentionally does not import the production multipath helper.
It reconstructs the published near/far shifted-window composition from raw
path nodes, preventing a shared helper from masking a path-composition error.
"""
import torch

from adaptive_deltaxai.swing import swing_attribution


def independent_line_integral(score, baseline, endpoint, nodes, samples=64):
    points = [baseline, *nodes, endpoint]
    segments = len(points) - 1
    locations = torch.linspace(0, float(segments), samples + 1, dtype=endpoint.dtype)
    path = []
    for location in locations:
        segment = min(int(location.floor().item()), segments - 1)
        alpha = location - segment
        path.append(points[segment] + alpha * (points[segment + 1] - points[segment]))
    total = torch.zeros_like(endpoint)
    for left, right in zip(path[:-1], path[1:]):
        midpoint = ((left + right) / 2).detach().requires_grad_(True)
        grad, = torch.autograd.grad(score(midpoint).sum(), midpoint)
        total = total + grad * (right - left)
    return total


def independent_swing(score, history, w=20, lag=5):
    current = history[:, -w:, :]
    previous = history[:, -w-1:-1, :]
    far = history[:, -w-1-lag:-1-lag, :]
    near = history[:, -w-lag:-lag, :]
    new_path = [history[:, -w-1:-1, :]]
    cur_far = independent_line_integral(score, far, current, new_path)
    prev_far = independent_line_integral(score, far, previous, [])
    cur_near = independent_line_integral(score, near, current, [])
    prev_near = independent_line_integral(score, near, previous, [])
    out = torch.zeros_like(history)
    out[:, -w-1:-1, :] += -(prev_far + prev_near) / 2
    out[:, -w:, :] += (cur_far + cur_near) / 2
    return out


def test_swing_matches_independent_reference_path():
    torch.manual_seed(19)
    history = torch.randn(1, 26, 3)
    weights = torch.tensor([0.3, -0.5, 0.2])
    def score(x): return (x[:, -1, :] * weights).sum(dim=1)
    expected = independent_swing(score, history)
    actual = swing_attribution(score, history, window=20, prediction_delta=1, retrospective_lag=5, n_samples=64)
    assert torch.allclose(actual, expected, atol=1e-5)
