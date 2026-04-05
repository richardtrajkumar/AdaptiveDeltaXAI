# AdaptiveDeltaXAI
Started from a paper under review with no public code, I conducted a dive deep on Delta-XAI, reconstructed the SWING mechanism faithfully from the mathematical description, extended it with four production-grade improvements, validated on synthetic datacenter telemetry in Google Colab, and achieved a 419–631× speedup over the vanilla baseline

# AdaptiveDeltaXAI — Explainable AI for Datacenter Telemetry Streams

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/arXiv-2511.23036-red)](https://arxiv.org/abs/2511.23036)
[![Colab](https://img.shields.io/badge/Colab-Open%20Notebook-yellow)](notebooks/Delta_XAI_Colab_Validation.ipynb)

> **Extends Delta-XAI (Kim et al., 2026) with adaptive caching, per-sensor thresholds, attribution drift detection, and online detrending — validated on synthetic datacenter temperature and power telemetry.**

---

## What this repository is

This repository documents a research deep-dive and working implementation based on:

**Delta-XAI: A Unified Framework for Explaining Prediction Changes in Online Time Series Monitoring**
Kim et al. (AITRICS), 2025/2026 · [arXiv:2511.23036](https://arxiv.org/abs/2511.23036) · [OpenReview:ZHW5pp5nE5](https://openreview.net/forum?id=ZHW5pp5nE5)

The original paper wraps 14 XAI methods and introduces **SWING** (Shifted Window Integrated Gradients) for online time series explanation. This repository:

1. **Reconstructs** the core SWING mechanism faithfully from the paper
2. **Extends** it with `AdaptiveDeltaXAI` — a caching layer designed for production datacenter monitoring
3. **Validates** both on synthetic DC temperature and power telemetry
4. **Benchmarks** recompute rate, explanation fidelity (AOPC), latency, and speedup

---

## Key Results

| Metric | Temp Vanilla | Temp Adaptive | Power Vanilla | Power Adaptive |
|--------|-------------|---------------|---------------|----------------|
| Recompute rate | 100% | **23.3%** | 100% | **7.5%** |
| Mean latency/step | 671 ms | **1.60 ms** | 686 ms | **1.09 ms** |
| P99 latency | 984 ms | **3.76 ms** | 1000 ms | **3.72 ms** |
| Mean cache age | 0 steps | **17.4 steps** | 0 steps | **21.4 steps** |
| Speedup vs vanilla | 1× | **419×** | 1× | **631×** |
| AOPC (untrained) | 0.039 | 0.027 | -0.001 | -0.001 |

> Note: AOPC values reflect untrained GRU (random weights). See [Future Work](#future-work) for trained-model targets.

---

## Repository structure

```
AdaptiveDeltaXAI/
│
├── README.md                          ← You are here
├── LICENSE
├── requirements.txt
│
├── code/
│   ├── adaptive_delta_xai.py          ← Core implementation (all classes)
│   ├── vanilla_delta_xai.py           ← SWING baseline (paper-faithful)
│   ├── generators.py                  ← DC temperature + power generators
│   └── evaluation.py                  ← AOPC, Spearman ρ, fidelity metrics
│
├── notebooks/
│   └── Delta_XAI_Colab_Validation.ipynb  ← Full Google Colab walkthrough
│
├── datasets/
│   ├── README_datasets.md             ← Dataset descriptions and schema
│   ├── sample_dc_temperature.csv      ← 100-row temperature sample
│   └── sample_dc_power.csv            ← 100-row power sample
│
├── results/
│   ├── dc_telemetry_datasets.png      ← Dataset validation plot
│   ├── validation_results.png         ← AdaptiveDeltaXAI vs Vanilla results
│   └── results_summary.md             ← Numerical results and interpretation
│
└── docs/
    ├── 01_paper_deep_dive.md          ← Sections 1-8 Delta-XAI analysis
    ├── 02_month12_baseline.md         ← Month 1-2 implementation notes
    ├── 03_colab_validation.md         ← Colab run results and diagnostics
    └── 04_future_work.md              ← Roadmap and next steps
```

---

## Quick start

### Option A — Google Colab (recommended, no setup)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/Delta_XAI_Colab_Validation.ipynb)

1. Open the notebook in Colab
2. Runtime → Change runtime type → T4 GPU
3. Run All cells (approximately 10–15 minutes)

### Option B — Local

```bash
git clone https://github.com/YOUR_USERNAME/AdaptiveDeltaXAI.git
cd AdaptiveDeltaXAI
pip install -r requirements.txt
python code/adaptive_delta_xai.py
```

---

## The AdaptiveDeltaXAI extensions

The original Delta-XAI runs SWING at every timestep — expensive at datacenter telemetry scale. `AdaptiveDeltaXAI` adds four improvements:

### 1. Adaptive per-sensor staleness threshold
```
θᵢ(t) = k · σᵢ(t)    where σᵢ = EMA(|residual_i|, α)
```
Each sensor gets its own trigger threshold proportional to its rolling volatility. A stable temperature sensor (σ ≈ 0.3°C) gets a tight threshold; a volatile power sensor (σ ≈ 10W) gets a loose one.

### 2. Attribution drift detector
```
Trigger if: ||E_cached − EMA(E_history)||_F > ψ
```
Catches concept drift where model predictions stay stable but the underlying feature importance has structurally changed — invisible to prediction-space monitoring alone.

### 3. Online EMA detrending
```
residual_t = x_t − EMA(x_t, α)
```
Removes slow trend and periodic components (CRAC cooling cycles, batch job workload cycles) from the trigger signal. Only anomaly-relevant residuals fire the recompute.

### 4. Temporal discount + group attribution
```
E[lag] *= γ^lag    (γ = 0.95–0.97)
```
Recent timesteps weighted more heavily. Optional ShaTS-style group attribution maps individual sensor attributions to rack zones or PDU groups — the level of abstraction SRE teams can actually act on.

---

## Dataset descriptions

### DC Temperature telemetry

```
Physical model:
  temp_i(t) = 35°C + U(-2,+2)          baseline
            + 2·sin(2πt/86400 + φᵢ)    diurnal cycle
            + 1.5·sin(2πt/600 + φᵢ)    10-min CRAC cycle
            + 0.15·temp_{i-1}(t)        thermal coupling
            + N(0, 0.3²)               noise
            + U(8,20)°C × anomaly_mask  cooling failure burst

Duration: 2h | Sampling: 10s | Sensors: 8 | Anomaly rate: ~6.7%
```

### DC Power telemetry

```
Physical model:
  power_i(t) = 200W + U(-20,+20)       server baseline
             + 50·sin(2πt/1800 + φᵢ)   30-min batch cycle
             + PDU_coupling(t)          shared PDU noise
             + N(0, 4²)                noise
             × U(1.5,2.5) × anomaly    power surge

Duration: 2h | Sampling: 10s | Servers: 8 | Anomaly rate: ~5%
Clipped to [50W, 900W] (physical hardware limits)
```

---

## Relation to existing work

| Paper | Year | Relation to this work |
|-------|------|-----------------------|
| **Delta-XAI** (Kim et al.) | 2026 | Foundation — SWING mechanism reproduced here |
| **DeltaSHAP** (AITRICS) | 2025 | Workshop predecessor, [public repo](https://github.com/AITRICS/DeltaSHAP) |
| **ShaTS** | 2025 | Group attribution approach integrated |
| **CausalRCA** | 2022 | Planned causal post-processing layer |
| **OmniAnomaly** | 2019 | Alternative backbone model (planned) |
| **MSCRED** | 2019 | Multi-sensor spatial model (planned) |

---

## Future work

### Month 2–3 (immediate)
- [ ] Train GRU on synthetic DC data → unlock meaningful AOPC scores
- [ ] Fix per-sensor Spearman ρ metric (currently degenerate with 2 groups)
- [ ] Add exponential thermal rise/fall time constant to temperature generator
- [ ] Scale N_EVAL=720, N_STEPS=20 for full-stream evaluation

### Month 3–4 (Extension 1)
- [ ] Rack topology graph grouping (ShaTS integration)
- [ ] Build DC adjacency matrix from rack floor plan
- [ ] Compare AOPC: graph-grouped vs flat-grouped vs per-sensor

### Month 4–5 (Extension 2)
- [ ] CausalRCA post-processing — online Granger causality
- [ ] Root Cause Score per sensor during anomaly bursts
- [ ] Validate that injected anomaly sensor ranks #1 in RCS

### Month 6+ (DC-XAI Framework)
- [ ] Unified framework: Delta-XAI + adaptive caching + topology groups + causal attribution
- [ ] Benchmark on real DC telemetry (if available)
- [ ] Target venue: IEEE TNSM or MLSys workshop

---

## Citation

If you build on this work, please also cite the original Delta-XAI paper:

```bibtex
@article{kim2025deltaxai,
  title={Delta-XAI: A Unified Framework for Explaining Prediction Changes
         in Online Time Series Monitoring},
  author={Kim, Changhun and others},
  journal={arXiv preprint arXiv:2511.23036},
  year={2025}
}
```

```bibtex
@misc{adaptivedelta2026,
  title={AdaptiveDeltaXAI: Caching-Aware Online Explainability
         for Datacenter Telemetry},
  author={Richard T Rajkumar},
  year={2026},
  url={https://github.com/richardtrajkumar/AdaptiveDeltaXAI}
}
```

---

## Acknowledgements

Built on top of concepts from [DeltaSHAP](https://github.com/AITRICS/DeltaSHAP) (AITRICS) and [WinIT](https://github.com/layer6ai-labs/WinIT) (Layer6 AI). Synthetic datasets modelled after ASHRAE datacenter thermal guidelines and standard server power profiles.

---

*Research in progress — contributions and issues welcome.*
