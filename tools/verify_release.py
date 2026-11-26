#!/usr/bin/env python3
"""Verify this release against the frozen run manifest. CPU only; stdlib only unless --deep is used.

Checks
  1. Synthetic data files match the SHA-256 values recorded in results/manifests/run_manifest_v1_9.json.
  2. The released code reproduces the manifest's code_tree_sha256 (hash computed exactly as in
     scripts/run_v1_9_stage.py, after reversing hotfix 2, which was not present in the tree when the manifest was written).
  3. (--deep) the CSV files agree with the .npy arrays.
  4. (--results-zip PATH) the frozen result bundle matches the SHA-256 in the freeze receipt.
  5. (--public-dir DIR) the downloaded SMD/PSM files match data/public/SHA256SUMS.

Usage:  python tools/verify_release.py [--deep] [--results-zip PATH] [--public-dir DIR]
"""
import argparse, hashlib, json, shutil, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "results" / "manifests" / "run_manifest_v1_9.json"
RECEIPT = REPO / "results" / "manifests" / "results_master_v1_9_frozen.freeze_receipt.json"

# path inside the as-run package -> path in this repository (the 22 files the manifest hash covers)
AS_RUN = {  # path inside the as-run source tree -> path in this repository (the 22 files the manifest hash covers)
    "ICLR_METHOD_CONTRACT.md": "docs/ICLR_METHOD_CONTRACT.md",
    "README.md": "provenance/as_run_package_README.md",
    "README_V1_9.md": "provenance/as_run_package_README_V1_9.md",
    "configs/master_v1_9.json": "configs/master_v1_9.json",
    "docs/ELFIKKY_10_POINT_TRACEABILITY.md": "docs/ELFIKKY_10_POINT_TRACEABILITY.md",
    "docs/V1_8_IMPLEMENTATION_MANIFEST.md": "docs/V1_8_IMPLEMENTATION_MANIFEST.md",
    "manifests/FEATURE_MANIFEST_v1_9.json": "manifests/FEATURE_MANIFEST_v1_9.json",
    "requirements.txt": "requirements.txt",
    "scripts/build_notebook_v1_8.py": "scripts/build_notebook_v1_8.py",
    "scripts/core_v1_9.py": "scripts/core_v1_9.py",
    "scripts/freeze_results_v1_9.py": "scripts/freeze_results_v1_9.py",
    "scripts/run_v1_9_stage.py": "scripts/run_v1_9_stage.py",
    "scripts/split_bundle_v1_9.py": "scripts/split_bundle_v1_9.py",
    "src/adaptive_deltaxai/__init__.py": "src/adaptive_deltaxai/__init__.py",
    "src/adaptive_deltaxai/metrics_v1_8.py": "src/adaptive_deltaxai/metrics_v1_8.py",
    "src/adaptive_deltaxai/models.py": "src/adaptive_deltaxai/models.py",
    "src/adaptive_deltaxai/swing.py": "src/adaptive_deltaxai/swing.py",
    "tests/test_metrics_v1_8.py": "tests/test_metrics_v1_8.py",
    "tests/test_public_alignment_v1_8.py": "tests/test_public_alignment_v1_8.py",
    "tests/test_swing.py": "tests/test_swing.py",
    "tests/test_swing_independent_reference.py": "tests/test_swing_independent_reference.py",
    "tests/test_swing_official_parity.py": "tests/test_swing_official_parity.py",
    "tests/test_v1_9_temporal_contract.py": "tests/test_v1_9_temporal_contract.py",
}

# Hotfix 2 (cache-fidelity history length) was applied in the notebook before the cache-fidelity stage but is absent from
# the tree that was hashed at assembly, so it is reversed here before hashing. See provenance/CODE_TREE_PROVENANCE.md.
HOTFIX2_NEW = "history_length=2*w+1+lag; hist=stream[r.t-history_length+1:r.t+1]"
HOTFIX2_OLD = "hist=stream[r.t-(w+1+lag)+1:r.t+1]"

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def tree_hash(root):  # identical to scripts/run_v1_9_stage.py::tree_hash
    digest = hashlib.sha256()
    for path in sorted(p for p in Path(root).rglob("*") if p.is_file() and p.suffix in {".py", ".json", ".md", ".txt"}):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()

def report(ok, label, detail=""):
    print(("PASS  " if ok else "FAIL  ") + label + (f"   {detail}" if detail else ""))
    return ok

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--deep", action="store_true"); ap.add_argument("--results-zip"); ap.add_argument("--public-dir")
    a = ap.parse_args(); manifest = json.loads(MANIFEST.read_text()); ok = True

    for name, want in manifest["synthetic_data_sha256"].items():
        got = sha256_file(REPO / "data" / "synthetic" / name)
        ok &= report(got == want, f"synthetic data  {name}", got[:16] + "...")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for rel, repo_rel in AS_RUN.items():
            dest = tmp / rel; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(REPO / repo_rel, dest)
        core = tmp / "scripts" / "core_v1_9.py"; text = core.read_text()
        if HOTFIX2_NEW not in text: ok &= report(False, "hotfix 2 marker present in core script")
        core.write_text(text.replace(HOTFIX2_NEW, HOTFIX2_OLD))
        got = tree_hash(tmp)
    ok &= report(got == manifest["code_tree_sha256"], "code tree hash (as run at assembly)", got[:16] + "...")

    if a.deep:
        import numpy as np, pandas as pd
        S = REPO / "data" / "synthetic"
        for stream, cols in (("temperature", 8), ("power", 3)):
            arr = np.load(S / f"tier2_{stream}_stream.npy"); df = pd.read_csv(S / f"tier2_{stream}_48h.csv")
            same = np.allclose(df.iloc[:, 1:1 + cols].values.astype("float32"), arr, rtol=0, atol=1e-4)
            same &= bool((df["label"].values == np.load(S / "tier2_labels.npy")).all())
            ok &= report(same, f"CSV matches npy ({stream}, labels)")

    if a.results_zip:
        want = json.loads(RECEIPT.read_text())["sha256"]; got = sha256_file(a.results_zip)
        ok &= report(got == want, "frozen result bundle SHA-256", got[:16] + "...")

    if a.public_dir:
        for line in (REPO / "data" / "public" / "SHA256SUMS").read_text().splitlines():
            want, rel = line.split(None, 1); p = Path(a.public_dir) / rel.strip()
            ok &= report(p.exists() and sha256_file(p) == want, f"public data  {rel.strip()}")

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED"); return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
