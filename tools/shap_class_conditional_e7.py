#!/usr/bin/env python3
"""Balanced, class-conditional SHAP analysis for the final ASVspoof PA E7 model.

The script draws the same number of bona-fide and spoof examples from the
held-out evaluation set (without replacement), explains them with TreeSHAP in
raw-margin/log-odds space, and reports feature- and group-level summaries.

This is post-hoc interpretation only. It must not be used to modify the model,
select hyperparameters, or tune a classification threshold.

Example (from project root):
  python tools/shap_class_conditional_e7.py \
      --model-dir results/xgb_tuning/E7_FINAL/best_model \
      --output-dir results/explainability/E7_FINAL/class_conditional \
      --samples-per-class 1000 --batch-size 256 --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def normalize_shap(raw_values: Any, n_rows: int, n_features: int) -> np.ndarray:
    """Normalize common SHAP binary-classifier return shapes."""
    if isinstance(raw_values, list):
        raw_values = raw_values[-1]
    values = np.asarray(raw_values)
    if values.ndim == 3:
        if values.shape[0] == n_rows and values.shape[1] == n_features:
            values = values[:, :, -1]
        elif values.shape[1] == n_rows and values.shape[2] == n_features:
            values = values[-1, :, :]
    if values.ndim != 2 or values.shape != (n_rows, n_features):
        raise ValueError(
            f"Unexpected SHAP shape {values.shape}; expected {(n_rows, n_features)}."
        )
    return values.astype(np.float32, copy=False)


def feature_group(name: str) -> str:
    if name.startswith("cqcc__"):
        return "CQCC"
    if name.startswith("tkeo__"):
        return "TECC/TKEO"
    if name.startswith("acoustic__"):
        return "RLFCC/acoustic"
    return "Other"


def load_feature_names(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("feature_names") or payload.get("features")
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{path} must contain a non-empty feature-name list")
    return [str(x) for x in payload]


def explain_class(
    explainer: Any,
    X: np.ndarray,
    feature_names: list[str],
    batch_size: int,
    label: str,
) -> np.ndarray:
    batches: list[np.ndarray] = []
    n_rows = X.shape[0]
    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        raw = explainer.shap_values(X[start:end], check_additivity=False)
        batches.append(normalize_shap(raw, end - start, len(feature_names)))
        print(f"SHAP {label}: {end:,}/{n_rows:,} samples", flush=True)
    return np.concatenate(batches, axis=0)


def build_summaries(
    shap_values: np.ndarray,
    feature_names: list[str],
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = np.asarray([feature_group(name) for name in feature_names], dtype=object)
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)  # positive => bona fide; negative => spoof
    direction = 1.0 if label == "bonafide" else -1.0
    aligned = shap_values * direction  # positive => supports the true class
    support_fraction = (aligned > 0).mean(axis=0)

    feature_df = pd.DataFrame({
        "class": label,
        "feature": feature_names,
        "feature_group": groups,
        "mean_abs_shap_log_odds": mean_abs,
        "mean_signed_shap_toward_bonafide_log_odds": mean_signed,
        "mean_signed_shap_toward_true_class_log_odds": aligned.mean(axis=0),
        "fraction_samples_feature_supports_true_class": support_fraction,
    }).sort_values("mean_abs_shap_log_odds", ascending=False)

    group_rows: list[dict[str, Any]] = []
    for group in ("CQCC", "TECC/TKEO", "RLFCC/acoustic", "Other"):
        columns = np.flatnonzero(groups == group)
        if columns.size == 0:
            continue
        per_sample_group = shap_values[:, columns].sum(axis=1)
        aligned_group = per_sample_group * direction
        group_rows.append({
            "class": label,
            "feature_group": group,
            "feature_count": int(columns.size),
            "sum_mean_abs_shap_log_odds": float(mean_abs[columns].sum()),
            "mean_abs_shap_per_feature": float(mean_abs[columns].mean()),
            "mean_net_shap_toward_bonafide_log_odds": float(per_sample_group.mean()),
            "mean_net_shap_toward_true_class_log_odds": float(aligned_group.mean()),
            "fraction_samples_group_supports_true_class": float((aligned_group > 0).mean()),
        })
    group_df = pd.DataFrame(group_rows)
    denom = float(group_df["sum_mean_abs_shap_log_odds"].sum())
    group_df["share_of_class_total_abs_shap_percent"] = (
        100.0 * group_df["sum_mean_abs_shap_log_odds"] / denom if denom else 0.0
    )
    return feature_df, group_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--feature-dir", default="results/fusion_features")
    parser.add_argument("--model-dir", default="results/xgb_tuning/E7_FINAL/best_model")
    parser.add_argument("--output-dir", default="results/explainability/E7_FINAL/class_conditional")
    parser.add_argument("--samples-per-class", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=25)
    args = parser.parse_args()

    if min(args.samples_per_class, args.batch_size, args.top_k) < 1:
        parser.error("samples-per-class, batch-size and top-k must be positive")

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
        raise SystemExit("Install requirements with: pip install xgboost shap matplotlib") from exc

    metadata = pd.read_csv(paths["metadata_eval"], dtype={"utterance_id": str})
    required = {"utterance_id", "key"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"metadata_eval.csv must contain {sorted(required)}")
    if metadata["utterance_id"].duplicated().any():
        raise ValueError("Duplicate utterance IDs in metadata_eval.csv")

    keys = metadata["key"].astype(str).to_numpy()
    unexpected = sorted(set(keys) - {"bonafide", "spoof"})
    if unexpected:
        raise ValueError(f"Unexpected labels in metadata_eval.csv: {unexpected}")

    feature_names = load_feature_names(paths["feature_names"])
    X_all = np.load(paths["X_eval"], mmap_mode="r")
    if X_all.ndim != 2 or X_all.shape != (len(metadata), len(feature_names)):
        raise ValueError(
            f"X_eval shape {X_all.shape} does not match metadata/features "
            f"({len(metadata)}, {len(feature_names)})."
        )

    class_rows = {
        "bonafide": np.flatnonzero(keys == "bonafide"),
        "spoof": np.flatnonzero(keys == "spoof"),
    }
    available = {label: int(rows.size) for label, rows in class_rows.items()}
    n_each = min(args.samples_per_class, *available.values())
    if n_each < args.samples_per_class:
        print(
            f"Requested {args.samples_per_class:,} per class, but a class has fewer samples; "
            f"using {n_each:,} per class.", flush=True
        )

    rng = np.random.default_rng(args.seed)
    selected = {
        label: np.sort(rng.choice(rows, size=n_each, replace=False))
        for label, rows in class_rows.items()
    }

    model = xgb.XGBClassifier()
    model.load_model(str(paths["model"]))
    expected = int(model.get_booster().num_features())
    if expected != len(feature_names):
        raise ValueError(f"Model expects {expected} features but feature_names.json has {len(feature_names)}")
    explainer = shap.TreeExplainer(model, model_output="raw")

    all_feature_frames: list[pd.DataFrame] = []
    all_group_frames: list[pd.DataFrame] = []
    sample_frames: list[pd.DataFrame] = []
    class_shap: dict[str, np.ndarray] = {}
    class_X: dict[str, np.ndarray] = {}

    for label in ("bonafide", "spoof"):
        rows = selected[label]
        X_sample = np.asarray(X_all[rows, :], dtype=np.float32)
        if not np.isfinite(X_sample).all():
            raise ValueError(f"Non-finite values found in sampled {label} features")
        shap_values = explain_class(explainer, X_sample, feature_names, args.batch_size, label)
        class_shap[label] = shap_values
        class_X[label] = X_sample

        feature_df, group_df = build_summaries(shap_values, feature_names, label)
        feature_df.to_csv(out_dir / f"shap_{label}_feature_importance.csv", index=False)
        group_df.to_csv(out_dir / f"shap_{label}_group_importance.csv", index=False)
        all_feature_frames.append(feature_df)
        all_group_frames.append(group_df)

        sample_cols = ["utterance_id", "key"]
        for optional in ("environment", "attack_id"):
            if optional in metadata.columns:
                sample_cols.append(optional)
        sample_info = metadata.iloc[rows][sample_cols].copy()
        sample_info.insert(0, "matrix_row", rows)
        sample_frames.append(sample_info)

        shap.summary_plot(
            shap_values, X_sample, feature_names=feature_names,
            max_display=args.top_k, show=False,
        )
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_{label}_beeswarm.png", dpi=300, bbox_inches="tight")
        plt.close()

        shap.summary_plot(
            shap_values, X_sample, feature_names=feature_names,
            plot_type="bar", max_display=args.top_k, show=False,
        )
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_{label}_bar.png", dpi=300, bbox_inches="tight")
        plt.close()

    feature_all = pd.concat(all_feature_frames, ignore_index=True)
    group_all = pd.concat(all_group_frames, ignore_index=True)
    samples_all = pd.concat(sample_frames, ignore_index=True)
    feature_all.to_csv(out_dir / "shap_class_conditional_feature_importance.csv", index=False)
    group_all.to_csv(out_dir / "shap_class_conditional_group_importance.csv", index=False)
    samples_all.to_csv(out_dir / "shap_class_conditional_sample.csv", index=False)

    # Paper-oriented comparison: average |SHAP| per feature by feature group and class.
    group_order = ["CQCC", "TECC/TKEO", "RLFCC/acoustic"]
    pivot = group_all.pivot(index="feature_group", columns="class", values="mean_abs_shap_per_feature")
    pivot = pivot.reindex(group_order).fillna(0.0)
    ax = pivot.plot(kind="bar", figsize=(9, 5), rot=0)
    ax.set_ylabel("Mean absolute SHAP per feature (log-odds)")
    ax.set_xlabel("Feature group")
    ax.set_title("Class-conditional feature-group attribution")
    ax.legend(title="Actual class")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_class_group_mean_abs_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    share_pivot = group_all.pivot(index="feature_group", columns="class", values="share_of_class_total_abs_shap_percent")
    share_pivot = share_pivot.reindex(group_order).fillna(0.0)
    ax = share_pivot.plot(kind="bar", figsize=(9, 5), rot=0)
    ax.set_ylabel("Share of class total |SHAP| (%)")
    ax.set_xlabel("Feature group")
    ax.set_title("Relative feature-group attribution by actual class")
    ax.legend(title="Actual class")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_class_group_share_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    metadata_out = {
        "model_path": str(paths["model"]),
        "feature_count": len(feature_names),
        "evaluation_population_size": len(metadata),
        "evaluation_class_counts": {
            "bonafide": available["bonafide"],
            "spoof": available["spoof"],
        },
        "samples_per_class_explained": int(n_each),
        "sampling": "independent uniform random samples without replacement within each true class",
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "shap_output_space": "raw margin / log-odds for the bona-fide class",
        "direction_convention": "positive SHAP pushes towards bona fide; negative SHAP pushes towards spoof; class-aligned support reverses the sign for spoof samples",
        "important_caveat": "Post-hoc model explanation only. SHAP associations are not causal acoustic mechanisms and were not used for model selection or threshold tuning.",
        "outputs": [
            "shap_bonafide_feature_importance.csv",
            "shap_spoof_feature_importance.csv",
            "shap_bonafide_group_importance.csv",
            "shap_spoof_group_importance.csv",
            "shap_class_conditional_feature_importance.csv",
            "shap_class_conditional_group_importance.csv",
            "shap_class_conditional_sample.csv",
            "shap_bonafide_beeswarm.png",
            "shap_spoof_beeswarm.png",
            "shap_bonafide_bar.png",
            "shap_spoof_bar.png",
            "shap_class_group_mean_abs_comparison.png",
            "shap_class_group_share_comparison.png",
        ],
    }
    (out_dir / "shap_class_conditional_metadata.json").write_text(
        json.dumps(metadata_out, indent=2), encoding="utf-8"
    )

    print("\nClass-conditional group SHAP summary:")
    print(group_all.to_string(index=False))
    print(f"\nSamples explained: {n_each:,} per class ({2*n_each:,} total)")
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
