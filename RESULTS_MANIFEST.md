# Results manifest

| Artifact | Where | Size | SHA-256 |
|---|---|---|---|
| Frozen result bundle (208 files) | Zenodo results record [DOI] | 55,777,070 bytes | `119ffebeca5d49cb863e31679a124ee8a916c3476439685ebcee23d97574c23a` |
| Freeze receipt | `results/manifests/results_master_v1_9_frozen.freeze_receipt.json` | small | `7abc48fc61adf9726d518ccbb8985113ab7339ca8055ada22ca2de3b15ea59c3` |
| Code-tree hash recorded at assembly | `results/manifests/run_manifest_v1_9.json` | - | `7d57cd0418187fe008ea3e9498abc9ef22848ae7752116db44374f04fa55c21b` |
| Synthetic data | `data/synthetic/` | ~3.4 MB | `data/synthetic/SHA256SUMS` |
| Public data (downloaded) | upstream repos at pinned commits | ~159 MB | `data/public/SHA256SUMS` |

The bundle contains per-seed checkpoints (`.pt`), controller records (`.npz`), per-stage CSV/JSON, six paper figures
plus ROC/PR plots, and the consistency report. This repository's `results/` folder holds the small files (tables,
figures, manifests) copied from the frozen bundle; the checkpoints, controller records and the six
`topology_context_*.csv` files are only in the archive.

The bundle was uploaded in four 15 MiB parts during preparation; the reassembled file's SHA-256 above matches the receipt.

## Verify
    python tools/verify_release.py --results-zip /path/to/results_master_v1_9_frozen.zip
