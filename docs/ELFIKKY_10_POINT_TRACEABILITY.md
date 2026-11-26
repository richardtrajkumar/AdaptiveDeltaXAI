# Master v1.6 traceability to the ten review points

| Review point | v1.6 mechanism | Required saved evidence | Permitted paper claim |
|---|---|---|---|
| 1. Cached vs. current explanation | `cached_event_context` is a separate record field; fresh maps are retained separately. | Cache-fidelity CSV, cache age, fresh comparison maps. | Cached maps provide historical operational context, not a guaranteed explanation of the current change. |
| 2. Original SWING vs. implementation | `swing.py` ports the official near/far dual-baseline and multipath construction. | Linear-model completeness test; source link/citation; implementation version in manifest. | Full-SWING construction is used only after preflight passes. |
| 3. Completeness | Completeness is calculated only on `raw_fresh_attribution`. | Fresh residual mean and p95; no cached residual claim. | Fresh SWING is approximately complete at the reported integration resolution. |
| 4. Ablation latency | Every configuration exposes fresh-call and amortized-step timing separately. | One ablation CSV generated from a common timer. | Scheduler changes amortized operational latency; it does not change intrinsic fresh-call cost. |
| 5. GPU vs. caching | A 2×2 design is retained in the final benchmark protocol. | Vectorized/unvectorized × always/adaptive table. | Batching and adaptive scheduling have separately reported effects. |
| 6. Faithfulness | A fixed common held-out prediction-change set, shared masking functional, and bootstrap CI are required. | Per-window curves and bootstrap results. | Faithfulness is reported for the specified evaluation functional only. |
| 7. Public data | `score_only_scheduling` and `full_attribution_subset` are separate modes. | Explicit mode column and subset fresh-map results. | Public score-only results establish scheduling behavior only. |
| 8. Numerical consistency | Assertions run before freezing; tables are derived from a single result directory. | `run_manifest.json`, all-seed JSON, consistency report, SHA-256 receipt. | All reported final-table values derive from the frozen v1.6 bundle. |
| 9. Reproducibility | Repository config, deterministic seeds, manifest, checkpoints, and frozen receipt. | Repository release tag and result archive. | The released package contains the reported configuration and generated artifacts. |
| 10. Presentation | Tables and captions are generated only after the frozen bundle is accepted. | Paper-source table CSVs and a visual PDF review checklist. | No claim; this is a release-quality gate. |

No earlier notebook result is a final v1.6 paper result. Earlier outputs may be
used for debugging and sanity comparison only.
