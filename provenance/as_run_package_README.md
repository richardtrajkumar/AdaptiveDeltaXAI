# AdaptiveDeltaXAI — Master v1.9

Master v1.9 is the canonical correction-and-evaluation release following the
immutable Master v1.7 T4 bundle. It corrects public reconstruction-label
alignment and adds justified Part-4 detection, refresh-coverage, paired
faithfulness, latency-tail, visualization, and reproducibility evidence.

## What v1.9 adds or corrects

- Public reconstruction windows `x[t-W:t]` are evaluated against `y[t-1]`,
  the label of the last reconstructed observation; regression tests enforce it.
- Threshold-independent public AUROC/AUPRC are primary evidence. Calibrated
  public threshold metrics use chronological non-test calibration and are
  supplementary.
- Synthetic three-seed MCC, balanced accuracy, specificity, false-alarm, and
  event-detection metrics are added without altering the frozen v1.7 baseline.
- Refresh-event coverage uses an explicit stored controller target index.
- Full-SWING paired AOPC differences and measured median/P95/P99 latency are
  generated from canonical artifacts.
- Offline score-change scheduling controls are named `offline_score_change_oracle`.
- Every stage is a short-lived process and saves its output before exit.

## Canonical execution path

Use `notebooks/AdaptiveDeltaXAI_Master_v1_9.ipynb`. It creates or reuses
`results_master_v1_9` beneath the selected data directory. Do **not** use
“Run all.” Execute numbered cells in order. The official pipeline requires a
CUDA runtime and blocks freezing if GPU provenance is absent.

## Release rules

- Do not alter or overwrite the frozen v1.7 directory.
- Do not run public stages until the public-alignment regression tests pass.
- Do not use test labels to select a public detection threshold.
- Do not call cached event context a current prediction-change explanation.
- Do not freeze until `CONSISTENCY_REPORT_v1_9.json` is `PASS`.
- Generate all empirical manuscript tables and figures from the frozen v1.9
  directory only.

## Version lineage

| Version | Principal retained or corrected contribution |
|---|---|
| v1.1–v1.2 | synthetic telemetry and detector foundations |
| v1.3 | chronological split, train-only normalization, public-data loaders |
| v1.4–v1.6 | cache semantics, timing separation, full-SWING path, staged execution |
| v1.7 | frozen T4 evidence, three-seed synthetic studies, public subset and release gates |
| v1.9 | corrected public label alignment; detection/event/refresh metrics; paired effects; latency tails; strengthened freeze gates |

See `manifests/FEATURE_MANIFEST_v1_9.json` and `ICLR_METHOD_CONTRACT.md`.
