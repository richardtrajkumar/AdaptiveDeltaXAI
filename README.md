# AdaptiveDeltaXAI

Adaptive scheduling of **fresh Full-SWING prediction-change explanations** for continuous datacenter telemetry
monitoring. The controller decides when a new Full-SWING attribution is computed; between refreshes it shows
*cached event context*, which is **not** claimed to explain the current prediction change.

**Relationship to Delta-XAI.** This work builds on Delta-XAI (Kim et al., arXiv:2511.23036) and its Full-SWING
attribution. It does not propose a new attribution method; the contribution is the refresh controller, the cached-context
labelling, and the evaluation of both.

- Paper: [arXiv link, added after posting]
- Release: `v1.9.0` (experimental evidence: Master v1.9) · Code archive: [Zenodo code DOI] · Frozen results archive: [Zenodo results DOI]

## What this repository contains
| Path | Content |
|---|---|
| `src/`, `scripts/`, `tests/`, `configs/` | v1.9 implementation, stage runner, regression tests, frozen configuration |
| `notebooks/` | the Colab notebook exactly as run (self-contained; includes runtime hotfix cells; its title cell still reads "DRAFT", see `provenance/CODE_TREE_PROVENANCE.md`) |
| `data/synthetic/` | the 48-hour synthetic dataset, its metadata/event log, checksums, and a generator that reproduces it bit-for-bit |
| `data/public/` | downloader and checksums for the SMD/PSM files used (the data themselves are not stored here) |
| `results/` | tables, figures and manifests generated from the frozen bundle (the full bundle is archived on Zenodo) |
| `provenance/` | the three hotfix patches and the code-tree provenance note |
| `tools/verify_release.py` | checks code, data and results against the run manifest |

## Verify in about a minute (CPU, no GPU needed)
    python tools/verify_release.py --deep
    python tools/verify_release.py --results-zip results_master_v1_9_frozen.zip     # after downloading the archive
    python data/synthetic/generator/generate_tier2.py --out /tmp/tier2                # regenerates the dataset

## Reproduce
See `REPRODUCIBILITY.md`. The official pipeline needs a CUDA GPU (a Tesla T4 was used) and is run stage by stage
in the notebook; do not use "Run all".

## Scope and limits
- Synthetic stream: 48 h, 8 temperature + 3 power channels, 72 injected fault events, three seeds (42, 52, 62).
  Chronological 70/10/20 split with train-only normalisation.
- Public data: three SMD units (machine-1-1, -1-2, -1-3) and PSM, single seed, not all 28 SMD machines. Public scheduling
  results are score-only; real Full-SWING maps are computed for a 50-window subset per unit. Public labels use the
  `window_end_t_minus_1` alignment.
- Faithfulness: a direct prediction-difference perturbation test on 50 common temperature windows does not support a
  Full-SWING superiority claim; see `results/paper_tables/PAPER_TABLES_v1_9.md`.
- Refresh rates over the full synthetic stream (three seeds): 8.09 +/- 0.02% (temperature) and 4.63 +/- 0.25% (power).
- Repeated T4 batched timing probe (100-step workload, 16.83% refresh): 62.59 +/- 6.49 ms (always refresh) vs
  9.75 +/- 0.24 ms (adaptive) per decision, a 6.42-fold ratio of means; detector scoring and grouping are excluded.

## Citation
See `CITATION.cff`. Cite the version DOI shown on the Zenodo record.

## Legacy code
Earlier notebooks and prototype code are preserved under the git tag `legacy-pre-v1.9` and the branch `legacy`.
They are superseded and some of their reported speedups are **not** supported by the final evaluation.
