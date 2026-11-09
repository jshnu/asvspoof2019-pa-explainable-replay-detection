"""Reproduce ASVspoof 2019 PA min-tDCF using the official v1 eval_metrics.py."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent / "tdcf"))
import eval_metrics as em  # official ASVspoof 2019 tDCF v1 implementation

EXPERIMENTS = {
    "E1_CQCC": "E1_CQCC",
    "E2_TKEO": "E2_TKEO",
    "E3_ACOUSTIC": "E3_ACOUSTIC",
    "E4_CQCC_TKEO": "E4_CQCC_TKEO",
    "E5_CQCC_ACOUSTIC": "E5_CQCC_ACOUSTIC",
    "E6_TKEO_ACOUSTIC": "E6_TKEO_ACOUSTIC",
    "E7_ALL": "E7_ALL",
}

COST_MODEL = {
    "Pspoof": 0.05,
    "Ptar": (1 - 0.05) * 0.99,
    "Pnon": (1 - 0.05) * 0.01,
    "Cmiss_asv": 1,
    "Cfa_asv": 10,
    "Cmiss_cm": 1,
    "Cfa_cm": 10,
}


def locate_asv_file(asv_dir: Path, split: str) -> Path:
    p = asv_dir / f"ASVspoof2019.PA.asv.{split}.gi.trl.scores.txt"
    if not p.exists():
        raise FileNotFoundError(f"ASV score file not found: {p}")
    return p


def read_asv_scores(path: Path):
    # Official file: source/class, trial_type, ASV score; no trial IDs.
    a = np.genfromtxt(path, dtype=str)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"Expected 3 columns in {path}; got shape {a.shape}")
    trial_type = a[:, 1]
    scores = a[:, 2].astype(float)
    tar = scores[trial_type == "target"]
    non = scores[trial_type == "nontarget"]
    spoof = scores[trial_type == "spoof"]
    if min(tar.size, non.size, spoof.size) == 0:
        raise ValueError(f"Missing target/nontarget/spoof group in {path}")
    return tar, non, spoof, {"target": int(tar.size), "nontarget": int(non.size), "spoof": int(spoof.size)}


def evaluate_split(pred_path: Path, asv_path: Path):
    df = pd.read_csv(pred_path)
    required = {"utterance_id", "key", "score_bonafide"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{pred_path}: missing columns {sorted(missing)}")
    if df["utterance_id"].duplicated().any():
        raise ValueError(f"Duplicate utterance IDs in {pred_path}")
    if df["score_bonafide"].isna().any() or not np.isfinite(df["score_bonafide"].to_numpy(float)).all():
        raise ValueError(f"Non-finite CM scores in {pred_path}")
    if not set(df["key"].unique()).issubset({"bonafide", "spoof"}):
        raise ValueError(f"Unexpected CM labels in {pred_path}: {df['key'].unique()}")
    bona = df.loc[df.key == "bonafide", "score_bonafide"].to_numpy(float)
    spoof_cm = df.loc[df.key == "spoof", "score_bonafide"].to_numpy(float)
    tar, non, spoof_asv, asv_counts = read_asv_scores(asv_path)

    # Match the official evaluate_tDCF_asvspoof19.py: determine ASV EER
    # threshold from the supplied ASV score file for this split.
    asv_eer, asv_threshold = em.compute_eer(tar, non)
    pfa, pmiss, pmiss_spoof = em.obtain_asv_error_rates(tar, non, spoof_asv, asv_threshold)
    curve, thresholds = em.compute_tDCF(bona, spoof_cm, pfa, pmiss, pmiss_spoof, COST_MODEL, False)
    idx = int(np.argmin(curve))
    cm_eer, cm_eer_threshold = em.compute_eer(bona, spoof_cm)
    return {
        "cm_trials": int(len(df)),
        "cm_bonafide": int(len(bona)),
        "cm_spoof": int(len(spoof_cm)),
        "asv_trials": int(sum(asv_counts.values())),
        "asv_counts": asv_counts,
        "asv_eer_percent": float(asv_eer * 100),
        "asv_threshold": float(asv_threshold),
        "asv_pfa_nontarget": float(pfa),
        "asv_pmiss_target": float(pmiss),
        "asv_pmiss_spoof": float(pmiss_spoof),
        "cm_eer_percent": float(cm_eer * 100),
        "cm_eer_threshold": float(cm_eer_threshold),
        "min_tdcf": float(curve[idx]),
        "cm_threshold_at_min_tdcf": float(thresholds[idx]),
        "normalization": "official eval_metrics.compute_tDCF: divide by min(C1, C2)",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=ROOT)
    ap.add_argument("--asv-dir", type=Path, default=ROOT.parent / "archive" / "PA" / "ASVspoof2019_PA_asv_scores")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "results" / "tdcf")
    args = ap.parse_args()
    project = args.project_root.resolve()
    asv_dir = args.asv_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"protocol": "ASVspoof 2019 PA official tDCF_python_v1", "cost_model": COST_MODEL, "splits": {}}
    for split in ("dev", "eval"):
        asv_path = locate_asv_file(asv_dir, split)
        results["splits"][split] = {}
        for exp_dir in EXPERIMENTS.values():
            pred = project / "results" / "xgb_ablation" / exp_dir / f"{split}_predictions.csv"
            if not pred.exists():
                raise FileNotFoundError(pred)
            results["splits"][split][exp_dir] = evaluate_split(pred, asv_path)
            print(f"{split:4s} {exp_dir:20s} min-tDCF={results['splits'][split][exp_dir]['min_tdcf']:.8f} "
                  f"ASV-EER={results['splits'][split][exp_dir]['asv_eer_percent']:.4f}%")

    json_path = out_dir / "pa_min_tdcf_results.json"
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    rows = []
    for split, exps in results["splits"].items():
        for exp, m in exps.items():
            rows.append({"split": split, "experiment": exp, **m})
    csv_path = out_dir / "pa_min_tdcf_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")

if __name__ == "__main__":
    main()
