# Public data (not stored in this repository)

The paper uses three SMD units (machine-1-1, machine-1-2, machine-1-3) and PSM. The files are large
(about 159 MB in total; `PSM/train.csv` alone is 64 MB), so they are downloaded from their original
sources rather than committed here.

    python data/public/download_public_data.py --out /path/to/DATA_ROOT

The script fetches each file from the upstream repository at a pinned commit and verifies it against
`SHA256SUMS`. Use `--verify-only` to check files you already have (for example, a copy on Google Drive).

| Dataset | Source (pinned commit in `public_sources.json`) | Licence | Cite |
|---|---|---|---|
| SMD | github.com/NetManAIOps/OmniAnomaly, `ServerMachineDataset/` | MIT | Su et al., KDD 2019 (OmniAnomaly) |
| PSM | github.com/eBay/RANSynCoders, `data/` | CC BY 4.0 | Abdulaal, Liu, Lancewicki, KDD 2021 (RANSynCoders) |

Both licences permit redistribution with attribution, but this repository keeps only the checksums and
the downloader so that the upstream sources remain the single point of truth.
