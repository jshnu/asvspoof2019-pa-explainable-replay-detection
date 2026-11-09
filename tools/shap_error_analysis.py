#!/usr/bin/env python3
"""
Evaluation-set SHAP error analysis for ASVspoof 2019 PA E7 (all features).

Run from the project root:
  python tools/shap_error_analysis.py --max-per-class 2000

Inputs:
  results/fusion_features/X_eval.npy
  results/fusion_features/metadata_eval.csv
  results/xgb_ablation/E7_ALL/xgb_fusion.json
  results/xgb_ablation/E7_ALL/metrics.json
  results/xgb_ablation/E7_ALL/eval_predictions.csv

The script computes SHAP values for false-accepted spoof trials and a
condition-matched sample of correctly rejected spoof trials. The decision
threshold is read from threshold_selected_on_dev in metrics.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default=".")
    p.add_argument("--max-per-class", type=int, default=2000,
                   help="Maximum number of false accepts and true rejects to explain.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--model-dir", default="results/xgb_ablation/E7_ALL")
    p.add_argument("--feature-dir", default="results/fusion_features")
    p.add_argument("--output-dir", default="results/shap_error_analysis")
    return p.parse_args()


def sample_condition_matched(df: pd.DataFrame, max_n: int, seed: int) -> pd.DataFrame:
    """Sample up to max_n rows, approximately preserving environment x attack mix."""
    if len(df) <= max_n:
        return df.copy()
    rng = np.random.default_rng(seed)
    group_cols = ["environment", "attack_id"]
    groups = list(df.groupby(group_cols, dropna=False, sort=False))
    # Proportional allocation with at least one sample per nonempty stratum
    sizes = np.array([len(g) for _, g in groups], dtype=int)
    raw = sizes / sizes.sum() * max_n
    alloc = np.floor(raw).astype(int)
    alloc = np.minimum(alloc, sizes)
    remaining = max_n - int(alloc.sum())
    fractions = raw - np.floor(raw)
    for idx in np.argsort(-fractions):
        if remaining <= 0:
            break
        if alloc[idx] < sizes[idx]:
            alloc[idx] += 1
            remaining -= 1
    chosen = []
    for ((_, _), group), n in zip(groups, alloc):
        if n > 0:
            chosen.append(group.sample(n=int(n), random_state=int(rng.integers(0, 2**31 - 1))))
    return pd.concat(chosen, ignore_index=True) if chosen else df.iloc[0:0].copy()


def normalize_shap_values(values, n_rows: int, n_features: int) -> np.ndarray:
    # SHAP API differs slightly between versions.
    if hasattr(values, "values"):
        values = values.values
    if isinstance(values, list):
        values = values[-1]
    arr = np.asarray(values)
    if arr.ndim == 3:
        # Binary classifiers may return (rows, features, classes).
        arr = arr[:, :, -1]
    if arr.shape != (n_rows, n_features):
        raise ValueError(f"Unexpected SHAP shape {arr.shape}; expected {(n_rows, n_features)}")
    return arr


def main():
    args = parse_args()
    root = Path(args.project_root).resolve()
    model_dir = root / args.model_dir
    feature_dir = root / args.feature_dir
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    required = {
        "X_eval": feature_dir / "X_eval.npy",
        "metadata_eval": feature_dir / "metadata_eval.csv",
        "model": model_dir / "xgb_fusion.json",
        "metrics": model_dir / "metrics.json",
        "predictions": model_dir / "eval_predictions.csv",
        "feature_names": model_dir / "feature_names.json",
    }
    missing = [f"{k}: {v}" for k, v in required.items() if not v.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    metrics = json.loads(required["metrics"].read_text(encoding="utf-8"))
    threshold = float(metrics["threshold_selected_on_dev"])
    feature_names = json.loads(required["feature_names"].read_text(encoding="utf-8"))
    if isinstance(feature_names, dict):
        feature_names = feature_names.get("feature_names", [])
    if not feature_names:
        raise ValueError("Could not read feature names from feature_names.json")

    metadata = pd.read_csv(required["metadata_eval"], dtype={"utterance_id": str})
    preds = pd.read_csv(required["predictions"], dtype={"utterance_id": str})
    needed_meta = {"utterance_id", "key", "attack_id", "environment"}
    if not needed_meta.issubset(metadata.columns):
        raise ValueError(f"metadata_eval.csv needs columns {sorted(needed_meta)}; got {list(metadata.columns)}")
    needed_pred = {"utterance_id", "key", "score_bonafide"}
    if not needed_pred.issubset(preds.columns):
        raise ValueError(f"eval_predictions.csv needs columns {sorted(needed_pred)}; got {list(preds.columns)}")

    if metadata["utterance_id"].duplicated().any() or preds["utterance_id"].duplicated().any():
        raise ValueError("Duplicate utterance IDs detected.")
    merged = metadata.merge(
        preds[["utterance_id", "key", "score_bonafide"]],
        on="utterance_id", how="inner", suffixes=("_meta", "_pred"), validate="one_to_one"
    )
    if len(merged) != len(metadata) or len(merged) != len(preds):
        raise ValueError(
            f"Prediction/metadata row mismatch: metadata={len(metadata)}, predictions={len(preds)}, joined={len(merged)}"
        )
    if not (merged["key_meta"].astype(str).values == merged["key_pred"].astype(str).values).all():
        raise ValueError("Labels disagree between metadata and prediction files.")
    merged["key"] = merged["key_meta"].astype(str)
    merged["score_bonafide"] = pd.to_numeric(merged["score_bonafide"], errors="raise")
    merged["decision"] = np.where(merged["score_bonafide"] >= threshold, "bonafide", "spoof")
    merged["error_type"] = np.where(
        (merged["key"] == "spoof") & (merged["decision"] == "bonafide"), "false_accept",
        np.where((merged["key"] == "spoof") & (merged["decision"] == "spoof"),
                 "correct_reject", "not_spoof")
    )

    # Matrices were built in manifest order. Verify identity against metadata_eval.
    X = np.load(required["X_eval"], mmap_mode="r")
    if X.shape != (len(metadata), len(feature_names)):
        raise ValueError(
            f"X_eval shape {X.shape} does not match metadata/features "
            f"({len(metadata)}, {len(feature_names)})."
        )
    # metadata_eval.csv is written in exactly the same row order as X_eval.npy.
    row_map = pd.Series(np.arange(len(metadata)), index=metadata["utterance_id"])
    merged["matrix_row"] = merged["utterance_id"].map(row_map)
    if merged["matrix_row"].isna().any():
        raise ValueError("Some prediction IDs could not be mapped to X_eval rows.")
    merged["matrix_row"] = merged["matrix_row"].astype(int)

    fa = merged[merged["error_type"] == "false_accept"].copy()
    tr = merged[merged["error_type"] == "correct_reject"].copy()
    if fa.empty or tr.empty:
        raise ValueError(f"Need both false accepts and correct rejects; found FA={len(fa)}, TR={len(tr)}")

    fa_sample = sample_condition_matched(fa, args.max_per_class, args.seed)
    tr_sample = sample_condition_matched(tr, args.max_per_class, args.seed + 1)
    selected = pd.concat([fa_sample, tr_sample], ignore_index=True)
    selected.to_csv(out / "selected_trials_for_shap.csv", index=False)

    # Load the exact E7 model and explain only selected rows.
    try:
        import xgboost as xgb
        import shap
    except ImportError as exc:
        raise SystemExit("Install dependencies: python -m pip install xgboost shap") from exc

    model = xgb.XGBClassifier()
    model.load_model(str(required["model"]))
    row_indices = selected["matrix_row"].to_numpy(dtype=int)
    X_selected = np.asarray(X[row_indices, :], dtype=np.float32)

    explainer = shap.TreeExplainer(model)
    shap_raw = explainer.shap_values(X_selected, check_additivity=False)
    shap_values = normalize_shap_values(shap_raw, len(selected), len(feature_names))

    # Per-trial signed SHAP values permit forensic review of individual errors.
    shap_long = pd.DataFrame(shap_values, columns=feature_names)
    shap_long.insert(0, "utterance_id", selected["utterance_id"].values)
    shap_long.insert(1, "error_type", selected["error_type"].values)
    shap_long.insert(2, "environment", selected["environment"].values)
    shap_long.insert(3, "attack_id", selected["attack_id"].values)
    shap_long.insert(4, "score_bonafide", selected["score_bonafide"].values)
    shap_long.to_csv(out / "selected_trial_shap_values.csv", index=False)

    # Global comparison: mean signed and mean absolute SHAP by error group.
    rows = []
    for label, idx in selected.groupby("error_type").groups.items():
        pos = selected.index.get_indexer(idx)
        vals = shap_values[pos]
        mean_signed = vals.mean(axis=0)
        mean_abs = np.abs(vals).mean(axis=0)
        for j, name in enumerate(feature_names):
            rows.append({
                "error_type": label,
                "feature": name,
                "mean_signed_shap": float(mean_signed[j]),
                "mean_abs_shap": float(mean_abs[j]),
                "n_trials": int(len(pos)),
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "shap_feature_summary_by_error.csv", index=False)

    # Top features by absolute SHAP for each error class.
    top = (summary.sort_values(["error_type", "mean_abs_shap"], ascending=[True, False])
           .groupby("error_type", sort=False).head(args.top_k))
    top.to_csv(out / "top_shap_features_by_error.csv", index=False)

    # Condition-specific comparison for the worst observed environment/attack strata.
    cond_rows = []
    for (env, attack, error_type), group in selected.groupby(
        ["environment", "attack_id", "error_type"], dropna=False
    ):
        idx = selected.index.get_indexer(group.index)
        vals = shap_values[idx]
        abs_mean = np.abs(vals).mean(axis=0)
        signed_mean = vals.mean(axis=0)
        top_idx = np.argsort(abs_mean)[::-1][:args.top_k]
        for j in top_idx:
            cond_rows.append({
                "environment": env,
                "attack_id": attack,
                "error_type": error_type,
                "n_trials_explained": int(len(group)),
                "feature": feature_names[j],
                "mean_signed_shap": float(signed_mean[j]),
                "mean_abs_shap": float(abs_mean[j]),
            })
    pd.DataFrame(cond_rows).to_csv(out / "top_shap_features_by_condition.csv", index=False)

    # Beeswarm plots for the two spoof decision outcomes.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for error_type in ("false_accept", "correct_reject"):
        mask = selected["error_type"].to_numpy() == error_type
        if int(mask.sum()) == 0:
            continue
        shap.summary_plot(
            shap_values[mask],
            X_selected[mask],
            feature_names=feature_names,
            max_display=args.top_k,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(out / f"shap_beeswarm_{error_type}.png", dpi=180, bbox_inches="tight")
        plt.close()

    # Group-level summaries for CQCC / TKEO / acoustic.
    group_rows = []
    prefixes = ("cqcc__", "tkeo__", "acoustic__")
    for error_type, group in selected.groupby("error_type"):
        idx = selected.index.get_indexer(group.index)
        vals = shap_values[idx]
        for prefix in prefixes:
            cols = [i for i, name in enumerate(feature_names) if name.startswith(prefix)]
            group_rows.append({
                "error_type": error_type,
                "feature_group": prefix[:-2],
                "n_features": len(cols),
                "mean_abs_shap": float(np.abs(vals[:, cols]).mean()) if cols else np.nan,
                "sum_mean_abs_shap": float(np.abs(vals[:, cols]).mean(axis=0).sum()) if cols else np.nan,
            })
    pd.DataFrame(group_rows).to_csv(out / "shap_group_summary_by_error.csv", index=False)

    metadata_out = {
        "model": "E7_ALL",
        "threshold_selected_on_dev": threshold,
        "evaluation_trials": int(len(merged)),
        "evaluation_spoof_trials": int((merged["key"] == "spoof").sum()),
        "false_accepts_total": int(len(fa)),
        "correct_rejects_total": int(len(tr)),
        "false_accepts_explained": int(len(fa_sample)),
        "correct_rejects_explained": int(len(tr_sample)),
        "feature_count": int(len(feature_names)),
        "sampling": "condition-stratified random sample; seed-controlled",
        "seed": args.seed,
        "max_per_class": args.max_per_class,
        "shap_class": "positive / bona fide class; positive SHAP increases bona-fide score",
        "note": "SHAP associations explain model output, not causal acoustic mechanisms."
    }
    (out / "analysis_metadata.json").write_text(json.dumps(metadata_out, indent=2), encoding="utf-8")
    print(json.dumps(metadata_out, indent=2))
    print(f"\nOutputs written to: {out}")


if __name__ == "__main__":
    main()
