# Master v1.9 Scientific Release Contract

## Method semantics

1. **Fresh explanation.** Every fresh explanation uses the near/far multipath shifted-window integrated-gradients construction in `src/adaptive_deltaxai/swing.py`. Raw fresh maps alone receive completeness residuals.
2. **Adaptive scheduler.** I1/I2 select refreshes from an EMA residual; I3 forces refresh on input drift or maximum cache age. I4 decays only displayed cached event context. I5 groups output by topology.
3. **Cached output.** A non-refresh output is `cached_event_context`: operational history of a prior event, not an explanation of the current prediction change and not a completeness-preserving attribution.
4. **Window/label contracts.** Synthetic windows `x[i-W:i]` predict `y[i]`. Public reconstruction windows `x[i-W:i]` reconstruct `x[i-1]` and must use `y[i-1]`. Regression tests block a release if either convention changes.
5. **SWING identity.** Independent structural and linear-completeness tests are mandatory. Direct numerical parity with an external official implementation is claimed only if a pinned reference fixture is available.

## Required evidence before freezing

- Chronological 70/10/20 split, training-only normalization, and saved synthetic split/window audits.
- Three seeds for synthetic streams, ablations, scaling, and GRU versus Transformer validation.
- Synthetic three-seed imbalance and event-level metrics, with thresholds defined before held-out evaluation.
- Common-window AW-IG, Full-SWING, zero-IG, KernelSHAP, and LIME comparison; paired AOPC differences and confidence intervals.
- Repeated GPU timing after warm-up, including median/P95/P99/max method latency and all four batching × caching conditions.
- Public full-test threshold-free AUROC/AUPRC with `window_end_t_minus_1` alignment.
- Calibrated thresholded public metrics derived from a chronological non-test training partition only.
- Per-unit public policy rows with exact matched refresh counts and offline-oracle labelling.
- Public fresh-attribution subsets that pass completeness gates after corrected alignment.
- Hardware, configuration, source-tree/data hashes, generated tables/figures, and a PASS consistency report.

## Reporting constraints

- Use timing provenance only from `HARDWARE_PROFILE_v1_9.json`.
- Treat public threshold-free metrics as primary. Label calibrated thresholded public metrics supplementary.
- Never infer Full-SWING superiority from AOPC unless the paired confidence interval supports it.
- State public scope exactly: three selected SMD units and PSM, with evaluated windows.
- Do not conflate an offline score-change oracle with causal online scheduling.
- All empirical manuscript values must originate in the final frozen v1.9 directory.
