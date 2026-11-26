# Reproducibility

## Environment of record (from the freeze receipt)
Python 3.13.16 · PyTorch 2.14.1+cu130 · CUDA 13.0 · Tesla T4 · Intel Xeon @ 2.00 GHz · ~13.3 GB RAM · Linux 6.6.159.
Seeds 42, 52, 62. Split: chronological 70/10/20 with train-only normalization.
`requirements.txt` is the unpinned list shipped with the run; versions of the non-PyTorch packages were not recorded.
[Add `requirements-lock.txt` from `pip freeze` in the environment you reproduce in.]
The notebook also runs `pip install lime` as a setup cell, so `lime` must be installed.

## What is verified without a GPU
1. `python tools/verify_release.py --deep` checks the synthetic data hashes, that the released code reproduces the
   manifest's `code_tree_sha256` (see `provenance/CODE_TREE_PROVENANCE.md`), and that the CSV files match the arrays.
2. `python data/synthetic/generator/generate_tier2.py --out DIR` regenerates all seven synthetic files; their bytes match
   `data/synthetic/`.
3. Eight metric tests, including the public t-1 label-alignment gate, do not use torch. They were run against
   `src/adaptive_deltaxai/metrics_v1_8.py` in isolation while preparing this release and passed (the as-run notebook
   also reports `8 passed` for its test cell). Running `pytest -q tests` normally requires torch.
4. `python data/public/download_public_data.py --out DATA_ROOT` downloads SMD/PSM at pinned upstream commits and verifies
   every file's SHA-256.

## What needs PyTorch
`pytest -q tests` also runs the SWING tests (linear completeness, independent reference, structural parity) and the
v1.9 temporal-contract tests (target/index, 26-observation history, cache age). They need `torch`; run them in your
environment before relying on the code.

## Full re-run (GPU)
1. Put the three synthetic `.npy` files and the downloaded public data under one `DATA_ROOT`
   (`DATA_ROOT/tier2_*.npy`, `DATA_ROOT/SMD/ServerMachineDataset/...`, `DATA_ROOT/PSM/*.csv`).
2. Open `notebooks/AdaptiveDeltaXAI_Master_v1_9_as_run.ipynb` on a T4 runtime and execute the numbered cells in order
   (not "Run all"). Each stage is a short-lived process that saves its output before exiting.
3. Compare the freeze receipt and the generated tables with `results/`.
The notebook restores its own embedded copy of the source and applies runtime hotfix cells; the scripts in this repository
already contain the fixes (see `provenance/`). Several notebook cells are one-off recovery cells from Colab reconnects
(directory listings, file checks, re-applying fixes); they are not part of the method.

## Known gaps
- The full frozen bundle (55.8 MB) is not in git; obtain it from the Zenodo results record and check its SHA-256.
- Exact timings depend on hardware and drivers; the receipt records the hardware used.
- `src/adaptive_deltaxai/__init__.py` lists three names in `__all__` (`AdaptiveSwingController`, `StepRecord`, `summarize`)
  that are not defined in this package; the controller lives in `scripts/core_v1_9.py`. This is kept as run; `from
  adaptive_deltaxai import *` would fail, explicit imports work.
