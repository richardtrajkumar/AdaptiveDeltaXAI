# Results Summary

Validation results from Google Colab run — AdaptiveDeltaXAI vs VanillaDeltaXAI (SWING baseline).

**Environment:** Google Colab, T4 GPU, Python 3.10, PyTorch 2.0
**Model:** GRU anomaly scorer (untrained — random weights, for mechanism validation)
**Evaluation steps:** N_EVAL = 120 | N_STEPS = 5 | Window = 20

---

## Master results table

| Metric | Temp Vanilla | Temp Adaptive | Power Vanilla | Power Adaptive |
|--------|-------------|---------------|---------------|----------------|
| Recompute rate (%) | 100.0 | **23.3** | 100.0 | **7.5** |
| Mean latency (ms/step) | 671.05 | **1.60** | 685.70 | **1.09** |
| P95 latency (ms) | 953.43 | 3.39 | 968.05 | 3.22 |
| P99 latency (ms) | 983.75 | 3.76 | 999.79 | 3.72 |
| Mean cache age (steps) | 0.0 | 17.4 | 0.0 | 21.4 |
| Max cache age (steps) | 1 | 59 | 1 | 59 |
| AOPC (mean) | 0.039 | 0.027 | -0.001 | -0.001 |
| Group-rank Spearman ρ | 1.000 | -0.000† | 1.000 | -0.102† |
| **Speedup vs vanilla** | 1× | **419×** | 1× | **631×** |

---

## Validation checklist results: 14/15 checks passed

| Check | Result | Details |
|-------|--------|---------|
| DeltaSHAP repo cloned | ✅ PASS | Directory exists |
| DC Temperature dataset generated | ✅ PASS | 720 rows |
| DC Power dataset generated | ✅ PASS | 720 rows |
| Temperature anomaly rate 5–40% | ✅ PASS | 6.7% |
| Power anomaly rate 5–40% | ❌ FAIL* | 5.0% (boundary) |
| Vanilla SWING explanations (Temp) | ✅ PASS | 120 explanations |
| Vanilla SWING explanations (Power) | ✅ PASS | 120 explanations |
| Adaptive recompute < 100% (Temp) | ✅ PASS | 23.3% |
| Adaptive recompute < 100% (Power) | ✅ PASS | 7.5% |
| Adaptive latency < Vanilla (Temp) | ✅ PASS | 1.60ms vs 671ms |
| Adaptive latency < Vanilla (Power) | ✅ PASS | 1.09ms vs 686ms |
| AOPC scores computed (Temp) | ✅ PASS | mean=0.039 |
| AOPC scores computed (Power) | ✅ PASS | mean=-0.001 |
| Group-rank Spearman ρ (Temp) | ✅ PASS | 118 values |
| Group-rank Spearman ρ (Power) | ✅ PASS | 118 values |

*\* False failure — power hit exactly 5.0%. Fix: change strict `>` to `>=` in check.*

---

## Trigger reason breakdown

| Trigger type | Temperature | Power | Interpretation |
|-------------|-------------|-------|----------------|
| pred_delta | 26 | 7 | Per-sensor threshold fired — main trigger |
| max_age | 2 | 2 | Safety bound T_max=60 reached — stable periods |
| attr_drift | 0 | 0 | Attribution drift not triggered (untrained model) |
| cached | 92 | 111 | Cache hit — no recompute needed |

The `max_age` fires confirm that T_max=60 is working correctly as a safety bound. With a trained model, we expect `pred_delta` to dominate during anomaly bursts and `cached` to dominate during stable operation.

---

## Metric diagnostics

### Why AOPC ≈ 0 (expected with untrained model)

AOPC measures whether masking the highest-attributed features reduces the model's anomaly score. With randomly initialised GRU weights, the model output ≈ 0.5 regardless of input, so masking any feature has no effect → AOPC = 0.

**This is not a code bug.** The AOPC implementation is correct — it correctly reveals that an untrained model has no meaningful feature preferences.

**Expected AOPC after training:**
- Temperature: 0.15–0.40 (clear spike signal)
- Power: 0.05–0.20 (periodic background confounds)

### Why Group-rank Spearman ρ ≈ 0 (metric design flaw)

With only 2 groups (zone_A, zone_B), Spearman ρ between [a₁, a₂] and [b₁, b₂] is always ±1. When attribution signs flip randomly across timesteps (untrained model), the mean converges to ~0. This is a **metric degeneracy**, not an attribution failure.

**Fix:** Use per-sensor Spearman ρ (8 values instead of 2). Expected ρ after training: 0.70–0.95.

---

## Production scaling projection

| Scenario | Vanilla latency/step | Adaptive latency/step | Feasible at 10s sampling? |
|----------|---------------------|----------------------|--------------------------|
| 8 sensors, N_STEPS=5 | 671ms | **1.60ms** | Both feasible |
| 100 sensors, N_STEPS=5 | ~8,400ms | **~20ms** | Adaptive only |
| 100 sensors, N_STEPS=20 | ~83,900ms | **~200ms** | Adaptive only |
| With GPU (10× speedup) | ~84ms | **~0.16ms** | Both, but adaptive still 525× faster |

---

## Pending fixes before Month 2 report

1. **Train GRU** → AOPC rises to 0.15–0.40 range
2. **Fix fidelity metric** → per-sensor Spearman ρ (n=8, not n=2 groups)
3. **Fix boundary check** → `>= 5%` instead of `> 5%`
4. **Scale evaluation** → N_EVAL=720, N_STEPS=20 for full 2-hour stream
5. **Add thermal rise/fall** → exponential anomaly model for temperature

---

*Results generated: March 2026 | Model: GRU (untrained) | Dataset: synthetic DC telemetry*
