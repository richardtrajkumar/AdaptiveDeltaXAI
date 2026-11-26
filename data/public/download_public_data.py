#!/usr/bin/env python3
"""Download the exact public SMD and PSM files used in the paper and verify their SHA-256.

SMD (3 units: machine-1-1, machine-1-2, machine-1-3) comes from NetManAIOps/OmniAnomaly (MIT licence).
PSM comes from eBay/RANSynCoders (CC BY 4.0). Files are fetched from the upstream repositories at the
pinned commits listed in public_sources.json; nothing is redistributed from this repository.

Usage:  python data/public/download_public_data.py --out /path/to/DATA_ROOT
The layout produced is the one the pipeline expects:
    DATA_ROOT/SMD/ServerMachineDataset/{train,test,test_label}/machine-1-{1,2,3}.txt
    DATA_ROOT/PSM/{train,test,test_label}.csv
"""
import argparse, hashlib, json, sys, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = json.loads((HERE / "public_sources.json").read_text())
RAW = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"
REPOS = {"SMD": "NetManAIOps/OmniAnomaly", "PSM": "eBay/RANSynCoders"}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="DATA_ROOT directory to populate")
    ap.add_argument("--verify-only", action="store_true", help="only check existing files")
    args = ap.parse_args()
    out = Path(args.out); bad = 0
    for item in SRC["files"]:
        dest = out / item["dest"]
        if not args.verify_only and not (dest.exists() and sha256(dest) == item["sha256"]):
            dest.parent.mkdir(parents=True, exist_ok=True)
            url = RAW.format(repo=REPOS[item["dataset"]], commit=SRC[item["dataset"]]["commit"], path=item["upstream_path"])
            print(f"downloading {item['dest']} ({item['bytes']/1e6:.1f} MB)", flush=True)
            urllib.request.urlretrieve(url, dest)
        ok = dest.exists() and sha256(dest) == item["sha256"]
        print(("OK       " if ok else "MISMATCH ") + item["dest"])
        bad += (not ok)
    print("\nAll public files verified." if not bad else f"\n{bad} file(s) failed verification.")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
