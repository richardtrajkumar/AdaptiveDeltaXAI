# AdaptiveDeltaXAI Master v1.9

Master v1.9 is a new result-generating release. It does not alter or consume
frozen v1.8 results. Each experiment stage starts a fresh Python process and
writes its durable artifact before releasing model memory.

## Scientific contracts introduced in v1.9

1. **Synthetic temporal contract.** The classifier input is
   `x[i-W:i]`; its endpoint is `i-1`; its target is `y[i]`. The final online
   endpoint has no label and is excluded from every label-dependent metric.
2. **Full-SWING history.** The near/far paths require exactly
   `W + delta + lag = 26` observations at `W=20`, `delta=1`, and `lag=5`.
   The repository reports internal path-structure and linear-completeness
   checks. It does not claim numerical output parity with an external official
   implementation unless a separately pinned reference fixture is added.
3. **Cache age.** `age_before=t-last_refresh` drives the max-age decision.
   On a cache hit, the display uses `age_after=age_before+1` and
   `gamma**age_after`; the first cache hit therefore has age one.
4. **Faithfulness.** `faithfulness_*` is a supplementary current-score proxy.
   `delta_faithfulness_*` is the primary paired-window prediction-difference
   perturbation test and uses real observed history.
5. **No-I1 interpretation.** The No-I1 ablation is named the
   always-refresh reference. Adaptive placement is tested with per-seed,
   exactly matched periodic, random, and offline-score-change-oracle controls.

## Required data layout

```
ADXAI_Dataset_48hr/
  tier2_temperature_stream.npy
  tier2_power_stream.npy
  tier2_labels.npy
  SMD/ServerMachineDataset/{train,test,test_label}/
  PSM/{train.csv,test.csv,test_label.csv}
```

## Run order

Use `AdaptiveDeltaXAI_Master_v1_9.ipynb`. Do not use **Run all**. The notebook
has a self-contained setup and runner definition in every executable cell, so
reconnects do not cause `NameError` failures. Official timing/assembly requires
CUDA; CPU is allowed only for the smoke/preflight route.

## Canonical code path

Only `scripts/core_v1_9.py`, `scripts/run_v1_9_stage.py`, and
`src/adaptive_deltaxai/swing.py` are executable experiment code. Earlier
alternative controller/experiment modules are retained under `legacy/` for
provenance and are excluded from the code-tree hash and release instructions.

## Public release wording

An internal checksum verifies bundle integrity. It does **not** demonstrate
public availability. Add a public repository URL, immutable commit/tag, and
archive DOI/checksum to the manuscript only after they actually exist.
