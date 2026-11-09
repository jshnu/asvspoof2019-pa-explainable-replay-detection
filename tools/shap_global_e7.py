#!/usr/bin/env python3
"""Global SHAP analysis for the final ASVspoof PA E7 XGBoost model.

Uses a reproducible random sample of the held-out evaluation feature matrix.
This is post-hoc interpretation only; it must not be used to change the model,
select hyperparameters, or tune a decision threshold.

Example (from project root):
  python tools/shap_global_e7.py \
      --model-dir results/xgb_tuning/E7_FINAL/best_model \
      --output-dir results/explainability/E7_FINAL/global \
      --sample-size 2000 --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _normalize_shap(raw_values, n_rows: int, n_features: int) -> np.ndarray:
    """Normalize SHAP outputs from supported SHAP/XGBoost API shapes."""
    if isinstance(raw_values, list):
        raw_values = raw_values[-1]
    values = np.asarray(raw_values)
    if values.ndim == 3:
        # Binary classifiers may expose a trailing output-class axis.
        if values.shape[0] == n_rows and values.shape[1] == n_features:
            values = values[:, :, -1]
        elif values.shape[1] == n_rows and values.shape[2] == n_features:
            values = values[-1, :, :]
    if values.ndim != 2 or values.shape != (n_rows, n_features):
        raise ValueError(
            f"Unexpected SHAP array shape {values.shape}; expected {(n_rows, n_features)}."
        )
    return values.astype(np.float32, copy=False)


def _feature_group(name: str) -> str:
    for prefix, group in (
        ("cqcc__", "CQCC"),
        ("tkeo__", "TECC/TKEO"),
        ("acoustic__", "RLFCC/acoustic"),
    ):
        if name.startswith(prefix):
            return group
    return "Other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--feature-dir", default="results/fusion_features")
    parser.add_argument("--model-dir", default="results/xgb_tuning/E7_FINAL/best_model")
    parser.add_argument("--output-dir", default="results/explainability/E7_FINAL/global")
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=25)
    args = parser.parse_args()

    if args.sample_size < 1 or args.batch_size < 1 or args.top_k < 1:
        parser.error("sample-size, batch-size and top-k must all be positive integers")

    root = args.project_root.resolve()
    feature_dir = (root / args.feature_dir).resolve()
    model_dir = (root / args.model_dir).resolve()
    out_dir = (root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "X_eval": feature_dir / "X_eval.npy",
        "metadata_eval": feature_dir / "metadata_eval.csv",
        "model": model_dir / "xgb_fusion.json",
        "feature_names": model_dir / "feature_names.json",
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    try:
        import xgboost as xgb
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install required packages with: pip install xgboost shap matplotlib") from exc

    metadata = pd.read_csv(paths["metadata_eval"], dtype={"utterance_id": str})
    if "utterance_id" not in metadata.columns:
        raise ValueError("metadata_eval.csv must contain utterance_id")

    X = np.load(paths["X_eval"], mmap_mode="r")
    feature_names = json.loads(paths["feature_names"].read_text(encoding="utf-8"))
    if isinstance(feature_names, dict):
        feature_names = feature_names.get("feature_names", [])
    if not feature_names:
        raise ValueError("feature_names.json does not contain a feature-name list")
    if X.ndim != 2 or X.shape != (len(metadata), len(feature_names)):
        raise ValueError(
            f"Evaluation matrix shape {X.shape} does not match metadata/features "
            f"({len(metadata)}, {len(feature_names)})."
        )
    if metadata["utterance_id"].duplicated().any():
        raise ValueError("Duplicate utterance IDs in metadata_eval.csv")

    n_total = len(metadata)
    n_sample = min(args.sample_size, n_total)
    rng = np.random.default_rng(args.seed)
    row_indices = np.sort(rng.choice(n_total, size=n_sample, replace=False))
    X_sample = np.asarray(X[row_indices, :], dtype=np.float32)

    model = xgb.XGBClassifier()
    model.load_model(str(paths["model"]))
    explainer = shap.TreeExplainer(model, model_output="raw")

    shap_batches: list[np.ndarray] = []
    for start in range(0, n_sample, args.batch_size):
        end = min(start + args.batch_size, n_sample)
        raw = explainer.shap_values(X_sample[start:end], check_additivity=False)
        shap_batches.append(_normalize_shap(raw, end - start, len(feature_names)))
        print(f"SHAP: {end:,}/{n_sample:,} evaluation samples", flush=True)
    shap_values = np.concatenate(shap_batches, axis=0)

    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)
    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "feature_group": [_feature_group(str(name)) for name in feature_names],
        "mean_abs_shap_log_odds": mean_abs,
        "mean_signed_shap_log_odds": mean_signed,
    }).sort_values("mean_abs_shap_log_odds", ascending=False)
    feature_importance.to_csv(out_dir / "shap_global_feature_importance.csv", index=False)

    # Group-level totals summarize the magnitude distributed over each feature group.
    group_rows = []
    for group in ("CQCC", "TECC/TKEO", "RLFCC/acoustic", "Other"):
        cols = [i for i, name in enumerate(feature_names) if _feature_group(str(name)) == group]
        if not cols:
            continue
        group_rows.append({
            "feature_group": group,
            "feature_count": len(cols),
            "sum_mean_abs_shap_log_odds": float(mean_abs[cols].sum()),
            "mean_abs_shap_per_feature": float(mean_abs[cols].mean()),
            "mean_net_signed_shap_log_odds": float(shap_values[:, cols].sum(axis=1).mean()),
        })
    group_df = pd.DataFrame(group_rows)
    group_total = group_df["sum_mean_abs_shap_log_odds"].sum()
    group_df["share_of_total_abs_shap_percent"] = (
        100.0 * group_df["sum_mean_abs_shap_log_odds"] / group_total if group_total else 0.0
    )
    group_df.to_csv(out_dir / "shap_global_group_importance.csv", index=False)

    # Save the exact sample so figures and conclusions are reproducible.
    sample_columns = ["utterance_id"]
    for optional in ("key", "environment", "attack_id"):
        if optional in metadata.columns:
            sample_columns.append(optional)
    sample_info = metadata.iloc[row_indices][sample_columns].copy()
    sample_info.insert(0, "matrix_row", row_indices)
    sample_info.to_csv(out_dir / "shap_global_sample.csv", index=False)

    # Beeswarm and global bar plots.
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        max_display=args.top_k,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "shap_global_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close()

    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        plot_type="bar",
        max_display=args.top_k,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "shap_global_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

    summary = {
        "model_path": str(paths["model"]),
        "feature_count": int(len(feature_names)),
        "evaluation_population_size": int(n_total),
        "shap_sample_size": int(n_sample),
        "sampling": "uniform random sample without replacement from the complete evaluation set",
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "shap_output_space": "raw margin / log-odds for the bona-fide class",
        "important_caveat": "This is post-hoc explanation only. SHAP values describe model associations, not causal acoustic mechanisms, and were not used for model selection or threshold tuning.",
        "outputs": [
            "shap_global_feature_importance.csv",
            "shap_global_group_importance.csv",
            "shap_global_sample.csv",
            "shap_global_beeswarm.png",
            "shap_global_bar.png",
        ],
    }
    (out_dir / "shap_global_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nGlobal SHAP summary:")
    print(group_df.to_string(index=False))
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
