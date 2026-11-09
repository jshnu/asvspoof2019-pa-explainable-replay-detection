#!/usr/bin/env python3
"""Reproducible environment x replay-attack analysis for ASVspoof 2019 PA.

Run from project root:
  .venv\\Scripts\\python.exe tools\\analyze_pa_conditions.py

Inputs expected:
  data/processed/manifests/PA_eval.csv
  results/xgb_ablation/<EXPERIMENT>/eval_predictions.csv
  results/xgb_ablation/<EXPERIMENT>/metrics.json

Outputs are written to results/condition_analysis/.
FAR is computed from score_bonafide >= threshold_selected_on_dev.
Only spoof trials are included in FAR denominators.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EXPERIMENTS = [
    "E1_CQCC", "E2_TKEO", "E3_ACOUSTIC", "E4_CQCC_TKEO",
    "E5_CQCC_ACOUSTIC", "E6_TKEO_ACOUSTIC", "E7_ALL",
]

# ASVspoof 2019 PA protocol definitions (README.PA)
ROOM = {"a": "2-5 m²", "b": "5-10 m²", "c": "10-20 m²"}
T60 = {"a": "50-200 ms", "b": "200-600 ms", "c": "600-1000 ms"}
TALKER_DIST = {"a": "10-50 cm", "b": "50-100 cm", "c": "100-150 cm"}
ATTACK_DIST = {"A": "10-50 cm", "B": "50-100 cm", "C": ">100 cm"}
DEVICE = {"A": "Perfect", "B": "High", "C": "Low"}


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z*z/n
    centre = (p + z*z/(2*n)) / den
    half = z * np.sqrt((p*(1-p) + z*z/(4*n))/n) / den
    return max(0.0, centre-half), min(1.0, centre+half)


def add_environment_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    env = df["environment"].astype(str).str.lower()
    atk = df["attack_id"].astype(str).str.upper()
    if not env.str.fullmatch(r"[abc]{3}").all():
        bad = sorted(env[~env.str.fullmatch(r"[abc]{3}")].unique())
        raise ValueError(f"Unexpected environment codes: {bad}")
    valid_attack = atk.str.fullmatch(r"[ABC]{2}") | atk.eq("-")
    if not valid_attack.all():
        bad = sorted(atk[~valid_attack].unique())
        raise ValueError(f"Unexpected attack codes: {bad}")
    df = df.copy()
    df["environment"] = env
    df["attack_id"] = atk
    df["room_size"] = env.str[0].map(ROOM)
    df["t60"] = env.str[1].map(T60)
    df["talker_asv_distance"] = env.str[2].map(TALKER_DIST)
    df["attacker_talker_distance"] = atk.str[0].map(ATTACK_DIST)
    df["replay_device_quality"] = atk.str[1].map(DEVICE)
    return df


def summarize(group: pd.DataFrame, keys: list[str], experiment: str) -> pd.DataFrame:
    rows = []
    for values, g in group.groupby(keys, dropna=False, sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        n = len(g)
        # score thresholding is used rather than trusting a stored predicted label.
        fa = int((g["score_bonafide"] >= g["threshold"]).sum())
        lo, hi = wilson_interval(fa, n)
        row = dict(zip(keys, values))
        row.update({
            "experiment": experiment,
            "spoof_trials": n,
            "false_accepts": fa,
            "far": fa / n if n else np.nan,
            "far_percent": 100 * fa / n if n else np.nan,
            "far_ci95_low_percent": 100 * lo,
            "far_ci95_high_percent": 100 * hi,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def save_heatmap(pivot: pd.DataFrame, title: str, path: Path, value_label: str = "FAR (%)") -> None:
    # Use matplotlib defaults; no custom palette is imposed.
    fig_w = max(9, 0.72 * len(pivot.columns) + 4)
    fig_h = max(6, 0.32 * len(pivot.index) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    arr = pivot.to_numpy(dtype=float)
    im = ax.imshow(arr, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    ax.set_xlabel("Replay attack ID (attacker distance, device quality)")
    ax.set_ylabel("Environment ID (room size, T60, talker-ASV distance)")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(value_label)
    # Annotate only finite cells; each cell also has its sample count in CSV.
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i, j]):
                ax.text(j, i, f"{arr[i,j]:.1f}", ha="center", va="center", fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.cwd(), help="Project root (default: current directory)")
    ap.add_argument("--experiments", nargs="*", default=EXPERIMENTS, help="Experiment folder names")
    args = ap.parse_args()
    root = args.root.resolve()
    manifest_path = root / "data/processed/manifests/PA_eval.csv"
    results_root = root / "results/xgb_ablation"
    out = root / "results/condition_analysis"
    out.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    required_m = {"utterance_id", "environment", "attack_id", "key"}
    if not required_m.issubset(manifest.columns):
        raise ValueError(f"Manifest missing columns: {sorted(required_m - set(manifest.columns))}")
    manifest = manifest[["utterance_id", "environment", "attack_id", "key"]].copy()

    all_env, all_attack, all_joint = [], [], []
    e7_joint = None
    for exp in args.experiments:
        exp_dir = results_root / exp
        pred_path = exp_dir / "eval_predictions.csv"
        metrics_path = exp_dir / "metrics.json"
        if not pred_path.exists() or not metrics_path.exists():
            raise FileNotFoundError(f"Missing prediction or metrics for {exp}: {exp_dir}")
        pred = pd.read_csv(pred_path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        threshold = float(metrics["threshold_selected_on_dev"])
        required_p = {"utterance_id", "key", "score_bonafide"}
        if not required_p.issubset(pred.columns):
            raise ValueError(f"{exp} predictions missing columns: {sorted(required_p - set(pred.columns))}")
        # Use manifest as source of truth for environment/attack labels.
        merged = pred.merge(manifest, on="utterance_id", how="left", suffixes=("_pred", "_manifest"), validate="one_to_one")
        if merged["environment"].isna().any():
            raise ValueError(f"{exp}: {merged['environment'].isna().sum()} predictions did not join to manifest")
        if not (merged["key_pred"].astype(str) == merged["key_manifest"].astype(str)).all():
            raise ValueError(f"{exp}: prediction labels disagree with evaluation manifest")
        # Attack label from manifest; only spoof trials are included below.
        merged["key"] = merged["key_manifest"]
        merged["attack_id"] = merged["attack_id_manifest"]
        merged = add_environment_descriptions(merged)
        merged["threshold"] = threshold
        spoof = merged[merged["key"].astype(str).str.lower().eq("spoof")].copy()
        if spoof.empty:
            raise ValueError(f"{exp}: no spoof rows after merge")
        env_s = summarize(spoof, ["environment", "room_size", "t60", "talker_asv_distance"], exp)
        atk_s = summarize(spoof, ["attack_id", "attacker_talker_distance", "replay_device_quality"], exp)
        joint_s = summarize(spoof, ["environment", "attack_id"], exp)
        all_env.append(env_s); all_attack.append(atk_s); all_joint.append(joint_s)
        if exp == "E7_ALL":
            e7_joint = joint_s
        # Save merged per-utterance analysis table for auditability.
        merged[["utterance_id", "key", "environment", "attack_id", "score_bonafide", "threshold"]].to_csv(
            out / f"{exp}_joined_eval_predictions.csv", index=False)
        print(f"{exp}: threshold={threshold:.12g}; spoof trials={len(spoof):,}; FAR={spoof['score_bonafide'].ge(threshold).mean()*100:.4f}%")

    env_df = pd.concat(all_env, ignore_index=True)
    atk_df = pd.concat(all_attack, ignore_index=True)
    joint_df = pd.concat(all_joint, ignore_index=True)
    env_df.to_csv(out / "far_by_environment_all_experiments.csv", index=False)
    atk_df.to_csv(out / "far_by_attack_all_experiments.csv", index=False)
    joint_df.to_csv(out / "far_by_environment_attack_all_experiments.csv", index=False)

    # Marginal parameter analyses, computed from individual spoof trials.
    # Reload joined tables so all rates use identical, auditable trial-level data.
    param_rows = []
    for exp in args.experiments:
        d = pd.read_csv(out / f"{exp}_joined_eval_predictions.csv")
        d = d[d["key"].astype(str).str.lower().eq("spoof")].copy()
        threshold = float(d["threshold"].iloc[0])
        d["false_accept"] = d["score_bonafide"] >= threshold
        env = d["environment"].astype(str)
        atk = d["attack_id"].astype(str)
        d["room_size"] = env.str[0].map(ROOM)
        d["t60"] = env.str[1].map(T60)
        d["talker_asv_distance"] = env.str[2].map(TALKER_DIST)
        d["attacker_talker_distance"] = atk.str[0].map(ATTACK_DIST)
        d["replay_device_quality"] = atk.str[1].map(DEVICE)
        for col in ["room_size", "t60", "talker_asv_distance", "attacker_talker_distance", "replay_device_quality"]:
            for value, g in d.groupby(col, sort=True):
                n = len(g); k = int(g["false_accept"].sum()); lo, hi = wilson_interval(k, n)
                param_rows.append({"experiment": exp, "parameter": col, "level": value,
                                   "spoof_trials": n, "false_accepts": k, "far_percent": 100*k/n,
                                   "far_ci95_low_percent": 100*lo, "far_ci95_high_percent": 100*hi})
    pd.DataFrame(param_rows).to_csv(out / "far_by_physical_parameter_all_experiments.csv", index=False)

    # One heatmap per experiment; row order is observed sorted environment codes.
    for exp in args.experiments:
        sub = joint_df[joint_df["experiment"] == exp]
        p = sub.pivot(index="environment", columns="attack_id", values="far_percent").sort_index()
        p.to_csv(out / f"{exp}_environment_attack_far_matrix.csv")
        save_heatmap(p, f"{exp}: evaluation FAR by environment × attack (%)",
                     out / f"{exp}_environment_attack_far_heatmap.png")

    # Compact machine-readable run metadata.
    metadata = {
        "analysis": "ASVspoof 2019 PA evaluation environment x attack FAR",
        "threshold_rule": "score_bonafide >= threshold_selected_on_dev",
        "threshold_source": "each experiment metrics.json",
        "denominator": "spoof trials only within each group",
        "confidence_interval": "95% Wilson score interval",
        "experiments": args.experiments,
        "environment_code": {"1st": "room size", "2nd": "T60 reverberation", "3rd": "talker-to-ASV distance"},
        "attack_code": {"1st": "attacker-to-talker distance", "2nd": "replay device quality"},
    }
    (out / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nWrote reproducible analysis to: {out}")

if __name__ == "__main__":
    main()
