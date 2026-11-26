"""Full shifted-window IG for prediction-change explanations.

This is a small, self-contained PyTorch port of the path construction in the
official Delta-XAI SWING implementation (AITRICS/Delta-XAI, `explainer/swing.py`).
It is intentionally separate from the adaptive scheduler so fresh-explanation
properties can be tested independently.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
import torch


TensorScore = Callable[[torch.Tensor], torch.Tensor]


def _score_and_grad(score_fn: TensorScore, inputs: torch.Tensor) -> torch.Tensor:
    """Return d sum(score_fn(inputs)) / d inputs with cuDNN disabled for GRUs."""
    x = inputs.detach().requires_grad_(True)
    with torch.backends.cudnn.flags(enabled=False):
        score = score_fn(x)
        if score.ndim != 1:
            score = score.reshape(score.shape[0], -1).sum(dim=1)
        grad, = torch.autograd.grad(score.sum(), x, create_graph=False)
    return grad


def multipath_integrated_gradients(
    score_fn: TensorScore,
    inputs: torch.Tensor,
    baseline: torch.Tensor,
    intermediate_points: Sequence[torch.Tensor] | None = None,
    n_samples: int = 64,
    serial_gradients: bool = False,
) -> torch.Tensor:
    """Trapezoidal integrated gradients over a piecewise-linear path.

    `n_samples` denotes the number of segments over the *entire* path, matching
    the official implementation rather than allocating this many points per
    segment.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least two")
    points = [baseline]
    points.extend([] if intermediate_points is None else intermediate_points)
    points.append(inputs)
    if len(points) < 2:
        return torch.zeros_like(inputs)

    segment_count = len(points) - 1
    positions = torch.linspace(0.0, float(segment_count), n_samples + 1,
                               device=inputs.device, dtype=inputs.dtype)
    nodes = []
    for position in positions:
        segment = min(int(torch.floor(position).item()), segment_count - 1)
        alpha = position - float(segment)
        nodes.append(points[segment] + alpha * (points[segment + 1] - points[segment]))
    nodes = torch.stack(nodes, dim=0)

    batch_size = inputs.shape[0]
    if serial_gradients:
        # Timing control only: mathematically identical nodes, one gradient
        # evaluation at a time.  This is the unbatched arm of the 2x2 study.
        gradients = torch.stack([_score_and_grad(score_fn, node) for node in nodes], dim=0)
    else:
        flat_nodes = nodes.reshape(-1, *inputs.shape[1:])
        gradients = _score_and_grad(score_fn, flat_nodes).reshape(nodes.shape[0], batch_size, *inputs.shape[1:])
    return (0.5 * (gradients[1:] + gradients[:-1]) * (nodes[1:] - nodes[:-1])).sum(dim=0)


def swing_attribution(
    score_fn: TensorScore,
    history: np.ndarray | torch.Tensor,
    window: int,
    prediction_delta: int = 1,
    retrospective_lag: int = 5,
    n_samples: int = 64,
    overall_path: str = "combined",
    serial_gradients: bool = False,
) -> torch.Tensor:
    """Explain f(current) - f(previous) using the original SWING composition.

    `history` must end at the current observation and contain at least
    `window + prediction_delta + retrospective_lag` observations. The
    returned matrix is the same shape as `history`; only the prior and current
    input-window locations receive attributions, exactly as in the official
    composition. The older baseline-only region remains zero.
    """
    if overall_path not in {"near", "far", "combined"}:
        raise ValueError("overall_path must be near, far, or combined")
    if not isinstance(history, torch.Tensor):
        history = torch.as_tensor(history, dtype=torch.float32)
    if history.ndim == 2:
        history = history.unsqueeze(0)
    if history.ndim != 3:
        raise ValueError("history must have shape [T,D] or [B,T,D]")

    # The four paths below reference only the trailing
    # W + delta + retrospective_lag observations.  Earlier releases used a
    # conservative 2W+delta+lag guard, although no slice consumed the first
    # W positions.  v1.9 makes this contract exact and unit-testable.
    required = window + prediction_delta + retrospective_lag
    if history.shape[1] < required:
        raise ValueError(f"history has {history.shape[1]} steps; requires at least {required}")

    # These slices exactly mirror create_cur_and_prev_input in the official
    # implementation, with the trailing observation being the current time.
    current = history[:, -window:, :]
    previous = history[:, -window - prediction_delta:-prediction_delta, :]
    far_baseline = history[:, -window - prediction_delta - retrospective_lag:-prediction_delta - retrospective_lag, :]
    near_baseline = history[:, -window - retrospective_lag:-retrospective_lag, :]

    # For the new/current path, visit each intervening historical window.
    new_path = [history[:, -window - offset:-offset, :] for offset in range(prediction_delta, 0, -1)]
    # For the old/previous path, use the official range(2, td).
    old_path = [history[:, -window - offset:-offset, :] for offset in range(2, prediction_delta)]

    current_far = multipath_integrated_gradients(score_fn, current, far_baseline, new_path, n_samples, serial_gradients)
    previous_far = multipath_integrated_gradients(score_fn, previous, far_baseline, None, n_samples, serial_gradients)
    current_near = multipath_integrated_gradients(score_fn, current, near_baseline, None, n_samples, serial_gradients)
    previous_near = multipath_integrated_gradients(score_fn, previous, near_baseline, old_path, n_samples, serial_gradients)

    if overall_path == "near":
        current_attr, previous_attr = current_near, previous_near
    elif overall_path == "far":
        current_attr, previous_attr = current_far, previous_far
    else:
        current_attr = (current_far + current_near) / 2.0
        previous_attr = (previous_far + previous_near) / 2.0

    out = torch.zeros_like(history)
    # The two explained windows overlap whenever prediction_delta < window.
    # Their attribution terms must therefore be accumulated, never assigned:
    # assignment erases the previous-path contribution over the overlap and
    # breaks the fresh-SWING completeness identity.
    out[:, -window - prediction_delta:-prediction_delta, :] += -previous_attr
    out[:, -window:, :] += current_attr
    return out


def linear_completeness_check(device: str = "cpu") -> float:
    """A deterministic unit test: linear scores make IG/SWING exact."""
    torch.manual_seed(7)
    weight = torch.tensor([0.2, -0.4, 0.6], device=device)
    history = torch.randn(1, 26, 3, device=device)
    def score_fn(x: torch.Tensor) -> torch.Tensor:
        return (x[:, -1, :] * weight).sum(dim=1)
    attrs = swing_attribution(score_fn, history, window=20, prediction_delta=1,
                              retrospective_lag=5, n_samples=32)
    current = history[:, -20:, :]
    previous = history[:, -21:-1, :]
    residual = (attrs.sum(dim=(1, 2)) - (score_fn(current) - score_fn(previous))).abs().max()
    return float(residual.detach().cpu())
