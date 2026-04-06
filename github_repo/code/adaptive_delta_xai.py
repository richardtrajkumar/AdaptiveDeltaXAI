"""
adaptive_delta_xai.py
=====================
AdaptiveDeltaXAI — Caching-Aware Online Explainability for Datacenter Telemetry

Extends Delta-XAI (Kim et al., 2026, arXiv:2511.23036) with:
  1. Adaptive per-sensor staleness threshold  θᵢ = k·σᵢ(t)
  2. Attribution drift detector              ||E_cached - EMA(E)||_F > ψ
  3. Online EMA detrending                   residual = x_t - EMA(x_t)
  4. Temporal discount                       E[lag] *= γ^lag
  5. Optional group attribution              ShaTS-style rack zone / PDU groups

Usage:
    from adaptive_delta_xai import AdaptiveDeltaXAI, VanillaDeltaXAI

    model_fn = lambda W: float(my_model(W))   # W: (1, window, features)

    adx = AdaptiveDeltaXAI(model=model_fn, window=20, k_sigma=2.5)
    for x_t in stream:
        result = adx.update(x_t)
        print(result.E, result.triggered, result.latency_ms)
"""

import numpy as np
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExplanationResult:
    """Output of one AdaptiveDeltaXAI.update() call."""
    t: int                  # current timestep index
    E: np.ndarray           # attribution matrix (window, n_features)
    triggered: bool         # True if SWING was recomputed this step
    reason: str             # 'init' | 'max_age' | 'pred_delta' | 'attr_drift' | 'cached'
    latency_ms: float       # wall-clock ms for this update
    cache_age: int          # steps since last recompute
    delta_pred: float       # |pred_t - pred_{t-1}| (prediction change)


# ─────────────────────────────────────────────────────────────────────────────
# VanillaDeltaXAI — paper-faithful SWING baseline
# ─────────────────────────────────────────────────────────────────────────────

class VanillaDeltaXAI:
    """
    Faithful reproduction of the Delta-XAI SWING mechanism.

    SWING (Shifted Window Integrated Gradients):
        E_t = (W_t - W_{t-1}) * (1/m) * Σ_{k=1}^{m} ∇f(W_{t-1} + k/m · ΔW)

    Key difference from standard IG:
        - Standard IG:  baseline = zero vector
        - SWING:        baseline = W_{t-1}  (previous window)

    This quantifies the CONTRIBUTION OF THE CHANGE ΔW, satisfying
    completeness, implementation invariance, and skew-symmetry in the
    online delta-explanation setting.

    Gradient approximation: numerical finite differences (model-agnostic).
    For differentiable PyTorch models, replace _gradient() with autograd.
    """

    def __init__(
        self,
        model: Callable,
        window: int = 30,
        n_steps: int = 20,
        eps: float = 1e-4,
    ):
        """
        Args:
            model:    Callable (1, window, features) → float
            window:   Sliding window length W
            n_steps:  IG integration steps m (higher = more accurate, slower)
            eps:      Finite-difference step size for gradient approximation
        """
        self.model  = model
        self.w      = window
        self.m      = n_steps
        self.eps    = eps
        self.buffer = deque(maxlen=window)
        self.W_prev: Optional[np.ndarray] = None
        self.t      = 0

    def _build_window(self, x_t: np.ndarray) -> np.ndarray:
        self.buffer.append(x_t.copy())
        W = np.array(self.buffer)
        if len(W) < self.w:
            W = np.vstack([np.zeros((self.w - len(W), x_t.shape[0])), W])
        return W  # (window, features)

    def _gradient(self, W: np.ndarray) -> np.ndarray:
        """Numerical gradient ∇f(W) via central finite differences."""
        base = self.model(W[np.newaxis])
        G = np.zeros_like(W)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W_plus = W.copy()
                W_plus[i, j] += self.eps
                G[i, j] = (self.model(W_plus[np.newaxis]) - base) / self.eps
        return G

    def _swing(self, W_prev: np.ndarray, W_curr: np.ndarray) -> np.ndarray:
        """SWING: piecewise IG path from W_prev → W_curr."""
        dW    = W_curr - W_prev
        grads = [
            self._gradient(W_prev + (k / self.m) * dW)
            for k in range(1, self.m + 1)
        ]
        return dW * np.mean(grads, axis=0)

    def update(self, x_t: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Process one new observation.

        Args:
            x_t: (n_features,) feature vector at current timestep

        Returns:
            E:          (window, n_features) SWING attribution matrix
            latency_ms: wall-clock time in milliseconds
        """
        t0     = time.perf_counter()
        W_curr = self._build_window(x_t)
        self.t += 1

        E = (np.zeros_like(W_curr)
             if self.W_prev is None
             else self._swing(self.W_prev, W_curr))

        self.W_prev = W_curr.copy()
        return E, (time.perf_counter() - t0) * 1000


# ─────────────────────────────────────────────────────────────────────────────
# AdaptiveDeltaXAI — enhanced wrapper with adaptive caching
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveDeltaXAI:
    """
    AdaptiveDeltaXAI extends VanillaDeltaXAI with four improvements:

    1. Adaptive per-sensor staleness threshold
       θᵢ(t) = k · σᵢ(t)   where σᵢ = EMA(|residualᵢ|, α)
       Each sensor gets its own threshold scaled to its volatility.

    2. Attribution drift detector (second-order staleness)
       Triggers recompute when ||E_cached - EMA(E_history)||_F > ψ
       Catches concept drift invisible to prediction-space monitoring.

    3. Online EMA detrending
       residual_t = x_t - EMA(x_t, α)
       Removes slow trend and periodic components (CRAC cycles,
       batch job cycles) so only anomaly-relevant signal triggers.

    4. Temporal discount + optional group attribution
       E[lag] *= γ^lag  (recent lags weighted more heavily)
       If feature_groups supplied: group-level SWING (ShaTS-style)

    Recompute conditions (any one fires SWING):
        (a) cache age >= T_max          (safety bound)
        (b) |residual_i| > k·σᵢ(t)     (per-sensor adaptive trigger)
        (c) ||E_cached - E_ema||_F > ψ  (attribution drift)
    """

    def __init__(
        self,
        model: Callable,
        window: int = 30,
        n_steps: int = 10,
        eps: float = 1e-4,
        k_sigma: float = 2.5,
        t_max: int = 100,
        gamma: float = 0.97,
        alpha_ema: float = 0.05,
        psi: float = 0.30,
        feature_groups: Optional[Dict[str, List[int]]] = None,
        name: str = "AdaptiveDeltaXAI",
    ):
        """
        Args:
            model:          Callable (1, window, features) → float
            window:         Sliding window length W
            n_steps:        IG integration steps (can be lower than vanilla)
            eps:            Finite-difference epsilon
            k_sigma:        θᵢ = k_sigma · σᵢ  (trigger sensitivity)
            t_max:          Maximum cache age before forced recompute
            gamma:          Temporal discount factor γ ∈ (0, 1]
            alpha_ema:      EMA smoothing coefficient for trend + sigma
            psi:            Attribution drift threshold ψ
            feature_groups: {group_name: [feature_indices]} for group attribution
            name:           Identifier for logging
        """
        self.model   = model
        self.w       = window
        self.m       = n_steps
        self.eps     = eps
        self.k       = k_sigma
        self.t_max   = t_max
        self.gamma   = gamma
        self.alpha   = alpha_ema
        self.psi     = psi
        self.groups  = feature_groups or {}
        self.name    = name

        # State
        self.buffer:     deque                    = deque(maxlen=window)
        self.W_prev:     Optional[np.ndarray]     = None
        self.E_cache:    Optional[np.ndarray]     = None
        self.E_ema:      Optional[np.ndarray]     = None
        self.sigma_ema:  Optional[np.ndarray]     = None
        self.trend_ema:  Optional[np.ndarray]     = None
        self.pred_prev:  Optional[float]          = None
        self.t:          int                      = 0
        self.t_last:     int                      = -t_max

        # Metrics history
        self.log: List[ExplanationResult] = []

    # ── Private helpers ───────────────────────────────────────────────────

    def _build_window(self, x_t: np.ndarray) -> np.ndarray:
        self.buffer.append(x_t.copy())
        W = np.array(self.buffer)
        if len(W) < self.w:
            W = np.vstack([np.zeros((self.w - len(W), x_t.shape[0])), W])
        return W

    def _detrend(self, x_t: np.ndarray) -> np.ndarray:
        """Online EMA detrending. Returns residual = x_t - trend."""
        if self.trend_ema is None:
            self.trend_ema = x_t.copy()
        self.trend_ema = (1 - self.alpha) * self.trend_ema + self.alpha * x_t
        return x_t - self.trend_ema

    def _check_trigger(self, residual: np.ndarray) -> Tuple[bool, str]:
        """Multi-condition staleness check. Returns (should_recompute, reason)."""
        age = self.t - self.t_last

        if age >= self.t_max:
            return True, "max_age"
        if self.E_cache is None:
            return True, "init"

        # Per-sensor adaptive threshold
        delta = np.abs(residual)
        if self.sigma_ema is None:
            self.sigma_ema = delta + 1e-6
        self.sigma_ema = (1 - self.alpha) * self.sigma_ema + self.alpha * delta
        if np.any(delta > self.k * self.sigma_ema):
            return True, "pred_delta"

        # Attribution drift
        if self.E_ema is not None:
            drift = float(np.linalg.norm(self.E_cache - self.E_ema, "fro"))
            if drift > self.psi:
                return True, "attr_drift"

        return False, "cached"

    def _gradient(self, W: np.ndarray) -> np.ndarray:
        base = self.model(W[np.newaxis])
        G = np.zeros_like(W)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W2 = W.copy()
                W2[i, j] += self.eps
                G[i, j] = (self.model(W2[np.newaxis]) - base) / self.eps
        return G

    def _swing(self, W_prev: np.ndarray, W_curr: np.ndarray) -> np.ndarray:
        """SWING with optional group attribution."""
        if self.groups:
            return self._group_swing(W_prev, W_curr)
        dW    = W_curr - W_prev
        grads = [
            self._gradient(W_prev + (k / self.m) * dW)
            for k in range(1, self.m + 1)
        ]
        E = dW * np.mean(grads, axis=0)
        return self._discount(E)

    def _group_swing(self, W_prev: np.ndarray, W_curr: np.ndarray) -> np.ndarray:
        """
        ShaTS-style group-level attribution.
        Each group's contribution = prediction drop when group reverts to W_prev.
        Broadcast back to individual features within each group.
        """
        E      = np.zeros_like(W_curr)
        p_full = self.model(W_curr[np.newaxis])
        for grp, idxs in self.groups.items():
            W_masked         = W_curr.copy()
            W_masked[:, idxs] = W_prev[:, idxs]
            contrib          = (p_full - self.model(W_masked[np.newaxis])) / max(len(idxs), 1)
            E[:, idxs]       = contrib
        return self._discount(E)

    def _discount(self, E: np.ndarray) -> np.ndarray:
        """Apply γ^lag temporal discount. lag=0 = most recent (bottom of window)."""
        w    = E.shape[0]
        lags = np.arange(w - 1, -1, -1)   # oldest lag = highest index
        return E * (self.gamma ** lags)[:, np.newaxis]

    def _update_ema(self, E: np.ndarray):
        if self.E_ema is None:
            self.E_ema = E.copy()
        self.E_ema = 0.9 * self.E_ema + 0.1 * E

    # ── Public API ────────────────────────────────────────────────────────

    def update(self, x_t: np.ndarray) -> ExplanationResult:
        """
        Process one new observation from the stream.

        Args:
            x_t: (n_features,) feature vector at current timestep

        Returns:
            ExplanationResult with attribution matrix E and metadata
        """
        t0      = time.perf_counter()
        self.t += 1

        W_curr   = self._build_window(x_t)
        residual = self._detrend(x_t)
        pred     = self.model(W_curr[np.newaxis])

        triggered, reason = self._check_trigger(residual)

        if triggered:
            if self.W_prev is None:
                self.E_cache = np.zeros_like(W_curr)
            else:
                self.E_cache = self._swing(self.W_prev, W_curr)
            self._update_ema(self.E_cache)
            self.t_last = self.t

        delta_pred     = abs(pred - self.pred_prev) if self.pred_prev is not None else 0.0
        self.W_prev    = W_curr.copy()
        self.pred_prev = pred

        result = ExplanationResult(
            t          = self.t,
            E          = self.E_cache.copy() if self.E_cache is not None
                         else np.zeros_like(W_curr),
            triggered  = triggered,
            reason     = reason,
            latency_ms = (time.perf_counter() - t0) * 1000,
            cache_age  = self.t - self.t_last,
            delta_pred = delta_pred,
        )
        self.log.append(result)
        return result

    def summary(self) -> Dict:
        """Aggregate metrics over all processed steps."""
        if not self.log:
            return {}
        n    = len(self.log)
        n_rc = sum(r.triggered for r in self.log)
        lats = [r.latency_ms for r in self.log]
        ages = [r.cache_age  for r in self.log]
        reasons: Dict[str, int] = {}
        for r in self.log:
            if r.triggered:
                reasons[r.reason] = reasons.get(r.reason, 0) + 1
        return {
            "n_steps":          n,
            "n_recompute":      n_rc,
            "recompute_pct":    100.0 * n_rc / n,
            "lat_mean_ms":      float(np.mean(lats)),
            "lat_p95_ms":       float(np.percentile(lats, 95)),
            "lat_p99_ms":       float(np.percentile(lats, 99)),
            "cache_age_mean":   float(np.mean(ages)),
            "cache_age_max":    int(np.max(ages)),
            "trigger_reasons":  reasons,
        }

    def reset(self):
        """Reset all state — use between independent evaluation runs."""
        self.buffer.clear()
        self.W_prev    = None
        self.E_cache   = None
        self.E_ema     = None
        self.sigma_ema = None
        self.trend_ema = None
        self.pred_prev = None
        self.t         = 0
        self.t_last    = -self.t_max
        self.log       = []


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: run comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_vanilla_vs_adaptive(
    stream: np.ndarray,
    model: Callable,
    window: int = 20,
    n_steps: int = 5,
    n_eval: int = 120,
    adaptive_kwargs: Optional[Dict] = None,
) -> Tuple[Dict, Dict]:
    """
    Run VanillaDeltaXAI and AdaptiveDeltaXAI on the same stream.

    Returns:
        vanilla_results:  dict with explanations, latencies, summary
        adaptive_results: dict with explanations, results log, summary
    """
    adaptive_kwargs = adaptive_kwargs or {}

    # Vanilla
    vanilla = VanillaDeltaXAI(model=model, window=window, n_steps=n_steps)
    van_Es, van_lats = [], []
    for x in stream[:n_eval]:
        E, lat = vanilla.update(x)
        van_Es.append(E)
        van_lats.append(lat)

    # Adaptive
    adaptive = AdaptiveDeltaXAI(model=model, window=window,
                                 n_steps=n_steps, **adaptive_kwargs)
    adp_results = [adaptive.update(x) for x in stream[:n_eval]]

    return (
        {"explanations": van_Es, "latencies": van_lats,
         "mean_latency": float(np.mean(van_lats)),
         "recompute_pct": 100.0},
        {"results": adp_results, **adaptive.summary()},
    )
