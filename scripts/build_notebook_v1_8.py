#!/usr/bin/env python3
"""Build the self-contained, memory-bounded Master v1.9 Colab notebook."""
from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUT = ROOT / "notebooks" / "AdaptiveDeltaXAI_Master_v1_9.ipynb"
EXCLUDE_PARTS = {"results", "results_template", "notebooks", "__pycache__", ".git"}
LEGACY = {
    "configs/master_v1_6.json", "configs/master_v1_7.json", "manifests/FEATURE_MANIFEST_v1_6.json", "manifests/FEATURE_MANIFEST_v1_7.json", "manifests/experiment_config_v1_6.json",
    "scripts/build_notebook.py", "scripts/build_notebook_v1_7.py", "scripts/freeze_results.py", "scripts/freeze_results_v1_7.py",
    "scripts/run_master_v1_6.py", "scripts/run_v1_6_conference_protocol.py", "scripts/run_v1_6_stage.py", "scripts/run_v1_7_stage.py",
    "scripts/core_v1_7.py", "scripts/run_v1_6_conference_protocol.py", "src/adaptive_deltaxai/swing_v1_7.py",
    "docs/PAPER_EVIDENCE_LANGUAGE_v1_6.json", "docs/PAPER_RELEASE_CHECKLIST_v1_6.md", "docs/RUN_ORDER.md",
}


def archive_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in ROOT.rglob("*"):
            relative = path.relative_to(ROOT)
            if path.is_file() and str(relative) not in LEGACY and not any(part in EXCLUDE_PARTS for part in relative.parts):
                zf.write(path, Path("AdaptiveDeltaXAI_v1_9") / relative)
    return buffer.getvalue()


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [text + "\n"]}


def code(lines: list[str]) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in lines]}


def main() -> None:
    payload = base64.b64encode(archive_bytes()).decode()
    cells = [
        md("# AdaptiveDeltaXAI — Master v1.9\n\nCorrected, memory-bounded Colab protocol. v1.7 remains immutable. **Do not use Run all.** Run executable cells in numerical order; every experiment stage saves to Drive and exits so Colab releases RAM/GPU memory. Official result stages require CUDA.\n\nPrimary public metrics are threshold-independent. Calibrated thresholded public metrics are saved separately as supplementary evidence."),
        code(["# 0. Restore the self-contained v1.9 source and install non-Torch dependencies.", "import base64, os, shutil, subprocess, sys", "from pathlib import Path", f"ARCHIVE_B64 = {payload!r}", "REPO_ROOT = Path('/content/AdaptiveDeltaXAI_v1_9')", "if not REPO_ROOT.exists():", "    archive = Path('/content/AdaptiveDeltaXAI_v1_9_source.zip')", "    archive.write_bytes(base64.b64decode(ARCHIVE_B64))", "    shutil.unpack_archive(str(archive), '/content')", "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-cache-dir', '-r', str(REPO_ROOT/'requirements.txt')], check=True)", "print('[PASS] v1.9 source restored:', REPO_ROOT)"]),
        code(["# 1. Mount Drive. v1.7 remains untouched; v1.9 has its own output directory.", "from google.colab import drive", "drive.mount('/content/drive', force_remount=False)", "DATA_ROOT = Path('/content/drive/MyDrive/Colab Notebooks/ADXAI_Dataset_48hr')", "RESULTS_ROOT = DATA_ROOT/'results_master_v1_9'", "assert DATA_ROOT.exists(), f'Missing data root: {DATA_ROOT}'", "print('DATA_ROOT:', DATA_ROOT)", "print('RESULTS_ROOT:', RESULTS_ROOT)" ]),
        code(["# 2. Mandatory preflight: dataset inputs, T4/CUDA, and all unit/regression tests.", "import json", "required = ['tier2_temperature_stream.npy', 'tier2_power_stream.npy', 'tier2_labels.npy']", "missing = [x for x in required if not (DATA_ROOT/x).exists()]", "assert not missing, f'Missing synthetic inputs: {missing}'", "probe = subprocess.run([sys.executable, '-c', 'import torch, json; print(json.dumps({\"torch\":torch.__version__,\"cuda\":torch.cuda.is_available(),\"gpu\":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))'], capture_output=True, text=True, check=True)", "hardware = json.loads(probe.stdout)", "print('Hardware:', hardware)", "assert hardware['cuda'], 'Official v1.9 runs require CUDA/T4. Reconnect to T4 before continuing.'", "env = dict(os.environ); env['PYTHONPATH'] = str(REPO_ROOT/'src')", "subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests'], cwd=REPO_ROOT, env=env, check=True)", "print('[PASS] tests include the public t-1 reconstruction-label alignment regression gate.')" ]),
        code(["# 3. Diagnostic stage runner: live output; each stage exits and releases memory.", "# Output is streamed instead of buffered, avoiding another source of RAM growth during long stages.", "def run_stage(stage, *extra):", "    runner = REPO_ROOT/'scripts'/'run_v1_9_stage.py'", "    env = dict(os.environ); env['PYTHONPATH'] = str(REPO_ROOT/'src'); env['PYTHONUNBUFFERED'] = '1'", "    cmd = [sys.executable, str(runner), '--data-root', str(DATA_ROOT), '--output-root', str(RESULTS_ROOT), '--stage', stage, *map(str, extra)]", "    print('RUN:', ' '.join(cmd), flush=True)", "    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)", "    assert proc.stdout is not None", "    for line in proc.stdout: print(line, end='', flush=True)", "    rc = proc.wait()", "    if rc: raise RuntimeError(f'Stage {stage} failed; complete diagnostic printed above.')", "    return rc", "print('[PASS] canonical v1.9 stages run one process at a time and stream progress live.')" ]),
        code(["# 4. Durable preflight receipt. This must pass before any public data stage can start.", "run_stage('preflight')"]),
        code(["# 5. Synthetic GRU/controller: seed 42.", "for stream in ('temperature','power'): run_stage('synthetic', '--seed', 42, '--stream', stream)"]),
        code(["# 6. Synthetic GRU/controller: seed 52.", "for stream in ('temperature','power'): run_stage('synthetic', '--seed', 52, '--stream', stream)"]),
        code(["# 7. Synthetic GRU/controller: seed 62.", "for stream in ('temperature','power'): run_stage('synthetic', '--seed', 62, '--stream', stream)"]),
        code(["# 8. Three-seed synthetic detector, imbalance, event, and refresh-coverage metrics.", "for seed in (42,52,62):", "    for stream in ('temperature','power'): run_stage('synthetic_metrics', '--seed', seed, '--stream', stream)"]),
        code(["# 9. Common-window fresh-attribution faithfulness: AW-IG, Full-SWING, zero-IG, KernelSHAP, and LIME.", "run_stage('faithfulness')"]),
        code(["# 10. Exact matched-budget synthetic scheduling controls. score_change is recorded as an offline oracle.", "for policy in ('adaptive','periodic','random','score_change'): run_stage('policy', '--policy', policy)"]),
        code(["# 11. Three-seed six-configuration ablation.", "for seed in (42,52,62):", "    for variant in ('Full','No_I1','No_I2','No_I3','No_I4','No_I5'): run_stage('ablation','--seed',seed,'--variant',variant)"]),
        code(["# 12. Repeated T4 batching × caching 2×2 timing.", "for condition in ('unbatched_always_refresh','unbatched_adaptive_refresh','batched_always_refresh','batched_adaptive_refresh'): run_stage('timing','--condition',condition)"]),
        code(["# 13. Comparable T4 latency trials for all five attribution methods.", "for method in ('zero_ig','aw_ig','full_swing','kernel_shap','lime'): run_stage('method_latency','--method',method)"]),
        code(["# 14. Three-seed controlled 8/16/32/64-feature scaling.", "for seed in (42,52,62):", "    for dimension in (8,16,32,64): run_stage('scaling','--seed',seed,'--dimensions',dimension)"]),
        code(["# 15. Three-seed GRU-versus-Transformer validation.", "for seed in (42,52,62): run_stage('architecture','--seed',seed)"]),
        code(["# 16. Corrected public threshold-free evaluation and fresh Full-SWING subset, one unit per process.", "for dataset, unit in [('SMD','machine-1-1'),('SMD','machine-1-2'),('SMD','machine-1-3'),('PSM','PSM')]: run_stage('public','--dataset',dataset,'--unit',unit,'--seed',42)"]),
        code(["# 17. Supplementary public calibrated threshold metrics; chronological non-test calibration only.", "for dataset, unit in [('SMD','machine-1-1'),('SMD','machine-1-2'),('SMD','machine-1-3'),('PSM','PSM')]: run_stage('public_calibrated','--dataset',dataset,'--unit',unit,'--seed',42)"]),
        code(["# 18. Post-hoc paired AOPC confidence intervals and latency median/P95/P99/max.", "run_stage('posthoc')"]),
        code(["# 19. Status. Do not assemble until every item is DONE.", "expected = ['PRECHECK_v1_9.json','synthetic_seed_results.csv','synthetic_extra_metrics_summary.csv','synthetic_refresh_coverage_per_seed.csv','faithfulness_summary.csv','paired_aopc_differences.csv','ablation_summary.csv','efficiency_2x2.csv','method_latency_tail_metrics.csv','scaling_summary.csv','architecture_validation.json','public_detection_summary.csv','public_calibrated_metrics_summary.csv','public_full_attribution_subset.csv']", "for name in expected: print(('DONE    ' if (RESULTS_ROOT/name).exists() else 'PENDING '), name)"]),
        code(["# 20. Assemble tables, figures, evidence map, and strict consistency report.", "run_stage('assemble')", "print((RESULTS_ROOT/'CONSISTENCY_REPORT_v1_9.json').read_text())"]),
        code(["# 21. Freeze only after Cell 20 prints status PASS.", "bundle = DATA_ROOT/'results_master_v1_9_frozen.zip'", "subprocess.run([sys.executable, str(REPO_ROOT/'scripts'/'freeze_results_v1_9.py'), '--results', str(RESULTS_ROOT), '--output', str(bundle)], cwd=REPO_ROOT, env=dict(os.environ, PYTHONPATH=str(REPO_ROOT/'src')), check=True)", "print('Frozen v1.9 bundle:', bundle)"]),
    ]
    notebook = {"cells": cells, "metadata": {"colab": {"name": OUT.name}, "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    OUT.write_text(json.dumps(notebook, indent=1))
    release_zip = ROOT.parent / "AdaptiveDeltaXAI_Master_v1_9_source.zip"
    release_zip.write_bytes(base64.b64decode(payload))
    print(OUT); print(release_zip)


if __name__ == "__main__":
    main()
