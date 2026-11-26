# Master v1.9 Implementation Manifest

This manifest is the implementation contract for the v1.7 → v1.9 upgrade.
Master v1.7 remains immutable.

1. Preserve v1.7 and create an independent v1.9 frozen bundle.
2. Correct public reconstruction labels to the final window observation.
3. Use the correction in every public label-dependent analysis.
4. Add synthetic/public alignment regression tests.
5. Regenerate the documented public subset after correction.
6. Report aligned threshold-free public AUROC, AUPRC, prevalence, and lift.
7. Use chronological non-test public calibration before thresholded public metrics.
8. Add synthetic MCC, balanced accuracy, specificity, and false alarms.
9. Add synthetic event recall, misses, delay, and false alarm episodes.
10. Keep fixed and validation-selected synthetic thresholds separate.
11. Add anomaly-event refresh coverage and first-refresh delay.
12. Persist explicit controller target indices.
13. Recompute public policy coverage with corrected labels and exact budgets.
14. Label score-change ranking as an offline oracle.
15. Add paired Full-SWING-minus-baseline AOPC confidence intervals.
16. Use event-cluster bootstrapping when the selected windows permit it.
17. Do not claim faithfulness superiority absent supporting intervals.
18. Add mean, median, P95, P99, maximum, and count latency evidence.
19. Keep method latency, scheduling, batching, and 2×2 timing separate.
20. Generate only justified detection, refresh, attribution, and trade-off figures.
21. Preserve cached-event-context semantics.
22. Preserve fresh-only completeness semantics.
23. Retain Parts 1–3 integrity safeguards and three-seed synthetic evidence.
24. State public scope accurately; do not claim all 28 SMD machines.
25. Record checkpoints, per-stage outputs, code/data hashes, and manifests.
26. Block official timing output without CUDA/GPU provenance.
27. Use one canonical v1.9 code path rather than sidecar scripts.
28. Use external SWING parity only when an authoritative fixture is supplied.
29. Run stages as separate processes and save each durable artifact immediately.
30. Update the paper only after the frozen v1.9 bundle passes.
31. Require alignment correction/tests before any affected public rerun.
32. Make threshold-independent public metrics primary; calibrated metrics supplementary.
33. Do not expand datasets, architectures, or public scope beyond this declared v1.9 protocol.
