# Code provenance

The notebook `notebooks/AdaptiveDeltaXAI_Master_v1_9_as_run.ipynb` restores a self-contained source tree from an embedded
archive and then applies **runtime hotfix cells**. This repository contains the *fully patched* scripts; the fixes are kept as
unified diffs in `provenance/hotfix_patches/` (apply in numeric order to the embedded source).

| Patch | Notebook cell(s) | Changes | When applied |
|---|---|---|---|
| 01 | "Fix 1" / "hotfix #2" | removes `from .controller import ...` from `src/adaptive_deltaxai/__init__.py` (the module is not in the archive) | before stage 4 (and re-applied after reconnects) |
| 02 | "prerun before #5" / "Fix 2" | cache-fidelity fresh map uses the controller's 46-observation history (`2W+1+lag`) instead of `W+1+lag` | **before the cache-fidelity stage (cell 5)**; not re-applied before assembly |
| 03 | "hot fix #2 before 18" / "hotfix #3" | `load_controller_records` reads the compressed `.npz` once and uses dictionary membership | before assembly (cell 18) |

Two other notebook cells change data, not code: "hot fix precell 17" concatenates the five per-method latency CSVs into
`method_latency_trials.csv`, and "pre-cell hot fix" runs `pip install lime`.

## What the recorded hash covers
`results/manifests/run_manifest_v1_9.json` records `code_tree_sha256 = 7d57cd04...55c21b`. Every stage process rewrites that
manifest, so the frozen manifest reflects the last stage run before freezing (assembly). The hash is reproduced **exactly**
by the embedded source with patches **01 and 03 applied and patch 02 not applied** (all 16 combinations were tested;
this is the only match). That is consistent with the runtime having been restarted before assembly and cell 9 ("Fix 2")
not having been re-run, while cells 3/25/26 were.

Consequence: the cache-fidelity CSVs were produced after patch 02 was applied (cell 5 ran after the "prerun before #5"
cell), but the tree hashed at assembly does not contain it. `tools/verify_release.py` therefore reverses patch 02 before
hashing the released tree. This ordering is inferred from the notebook cell order and the hash match; it was not
independently re-executed (no GPU here).

## Notebook title
The notebook's first cell is titled "Master v1.9 - DRAFT; not release-validated; do not use Run all". The frozen
`CONSISTENCY_REPORT_v1_9.json` reports `PASS` and the freeze cell ran afterwards, but the title cell was not updated.
The notebook is shipped unmodified, so its history is preserved.

Patches 01-03 do not touch the scientific scripts other than the one-line cache-fidelity history length and the loader.
