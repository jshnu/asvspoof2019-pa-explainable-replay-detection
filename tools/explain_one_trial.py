#!/usr/bin/env python3
"""Explain one ASVspoof PA E7 prediction with local SHAP values.

Run from project root:
  python tools/explain_one_trial.py --error-type false_accept
  python tools/explain_one_trial.py --utterance-id PA_E_XXXXXXXX

Uses the existing fusion feature matrix and metadata in results/fusion_features,
the trained E7 model, and the dev-selected threshold. SHAP is computed in raw
margin (log-odds) space and checked against XGBoost's predicted probability.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700, 700)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature-dir", default="results/fusion_features")
    p.add_argument("--model-dir", default="results/xgb_tuning/E7_FINAL/best_model")
    p.add_argument("--prediction-file", default=None,
                   help="Optional prediction CSV. By default, search model-dir then model-dir/final_eval.")
    p.add_argument("--output-dir", default="results/explainability/E7_FINAL/individual")
    p.add_argument("--utterance-id", default=None)
    p.add_argument("--error-type", choices=["false_accept", "false_reject", "correct_accept", "correct_reject"],
                   default="false_accept")
    p.add_argument("--top-k", type=int, default=20)
    args = p.parse_args()

    feature_dir, model_dir, out_dir = map(Path, (args.feature_dir, args.model_dir, args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = Path(args.prediction_file) if args.prediction_file else None
    if prediction_path is None:
        prediction_candidates = [
            model_dir / "eval_predictions.csv",
            model_dir / "final_eval" / "eval_predictions.csv",
        ]
        prediction_path = next((candidate for candidate in prediction_candidates if candidate.is_file()),
                               prediction_candidates[-1])
    paths = {
        "X": feature_dir / "X_eval.npy",
        "schema": feature_dir / "feature_schema.json",
        "metadata": feature_dir / "metadata_eval.csv",
        "model": model_dir / "xgb_fusion.json",
        "names": model_dir / "feature_names.json",
        "metrics": model_dir / "metrics.json",
        "predictions": prediction_path,
    }
    missing = [str(v) for v in paths.values() if not v.is_file()]
    if missing:
        raise FileNotFoundError("Missing required project files:\n" + "\n".join(missing))

    import xgboost as xgb
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))

    # Current E7_FINAL metrics.json stores the dev-selected operating
    # threshold under refit_dev_metrics.threshold_at_min_tdcf. Older
    # experiment scripts used threshold_selected_on_dev at the top level.
    # The evaluation report also stores the same threshold under
    # fixed_threshold.value, so accept that schema as a final fallback.
    threshold_candidates = [
        ("refit_dev_metrics.threshold_at_min_tdcf",
         metrics.get("refit_dev_metrics", {}).get("threshold_at_min_tdcf")
         if isinstance(metrics.get("refit_dev_metrics"), dict) else None),
        ("threshold_selected_on_dev", metrics.get("threshold_selected_on_dev")),
        ("fixed_threshold.value",
         metrics.get("fixed_threshold", {}).get("value")
         if isinstance(metrics.get("fixed_threshold"), dict) else None),
    ]
    threshold_source, threshold_value = next(
        ((name, value) for name, value in threshold_candidates if value is not None),
        (None, None),
    )
    if threshold_value is None:
        available = ", ".join(metrics.keys())
        raise KeyError(
            "Could not find the development-selected threshold in metrics.json. "
            "Expected refit_dev_metrics.threshold_at_min_tdcf, "
            "threshold_selected_on_dev, or fixed_threshold.value. "
            f"Available top-level keys: {available}"
        )
    threshold = float(threshold_value)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"Invalid bona-fide probability threshold {threshold!r} "
            f"loaded from {threshold_source}."
        )
    print(f"Dev-selected bona-fide threshold: {threshold:.12g} ({threshold_source})")
    feature_names = json.loads(paths["names"].read_text(encoding="utf-8"))
    schema = json.loads(paths["schema"].read_text(encoding="utf-8"))
    all_names = schema["feature_names"]
    lookup = {name: i for i, name in enumerate(all_names)}
    missing_names = [name for name in feature_names if name not in lookup]
    if missing_names:
        raise ValueError(f"{len(missing_names)} model features absent from feature schema.")
    col_idx = np.asarray([lookup[name] for name in feature_names], dtype=int)

    metadata = pd.read_csv(paths["metadata"], dtype={"utterance_id": str})
    predictions = pd.read_csv(paths["predictions"], dtype={"utterance_id": str})
    required_meta = {"utterance_id", "key"}
    required_pred = {"utterance_id", "score_bonafide"}
    if not required_meta.issubset(metadata.columns) or not required_pred.issubset(predictions.columns):
        raise ValueError("Metadata/prediction CSVs lack required utterance_id, key or score_bonafide columns.")
    df = metadata.merge(predictions[["utterance_id", "score_bonafide"]],
                        on="utterance_id", how="inner", validate="one_to_one")
    if len(df) != len(metadata) or len(df) != len(predictions):
        raise ValueError("Prediction and metadata IDs do not align one-to-one.")
    df["score_bonafide"] = pd.to_numeric(df["score_bonafide"], errors="raise")
    df["predicted_key"] = np.where(df["score_bonafide"] >= threshold, "bonafide", "spoof")
    df["error_type"] = np.select(
        [
            (df["key"] == "spoof") & (df["predicted_key"] == "bonafide"),
            (df["key"] == "bonafide") & (df["predicted_key"] == "spoof"),
            (df["key"] == "bonafide") & (df["predicted_key"] == "bonafide"),
            (df["key"] == "spoof") & (df["predicted_key"] == "spoof"),
        ],
        ["false_accept", "false_reject", "correct_accept", "correct_reject"],
        default="unknown"
    )
    if args.utterance_id:
        selected = df[df["utterance_id"] == args.utterance_id]
        if selected.empty:
            raise ValueError(f"Utterance ID not found: {args.utterance_id}")
        row = selected.iloc[0]
    else:
        candidates = df[df["error_type"] == args.error_type].copy()
        if candidates.empty:
            raise ValueError(f"No trials found for {args.error_type}.")
        # Select the most confident example of the requested category.
        if args.error_type == "false_accept":
            row = candidates.sort_values("score_bonafide", ascending=False).iloc[0]
        elif args.error_type == "false_reject":
            row = candidates.sort_values("score_bonafide", ascending=True).iloc[0]
        elif args.error_type == "correct_accept":
            row = candidates.sort_values("score_bonafide", ascending=False).iloc[0]
        else:
            row = candidates.sort_values("score_bonafide", ascending=True).iloc[0]

    utterance_id = str(row["utterance_id"])
    # Critical alignment: metadata_eval.csv and X_eval.npy share row order.
    matches = np.flatnonzero(metadata["utterance_id"].astype(str).to_numpy() == utterance_id)
    if len(matches) != 1:
        raise ValueError(f"Expected one matrix row for {utterance_id}; found {len(matches)}")
    matrix_row = int(matches[0])
    X_all = np.load(paths["X"], mmap_mode="r")
    if X_all.shape[0] != len(metadata):
        raise ValueError(f"X_eval rows ({X_all.shape[0]}) != metadata rows ({len(metadata)}).")
    x = np.asarray(X_all[matrix_row, col_idx], dtype=np.float32).reshape(1, -1)

    model = xgb.XGBClassifier()
    model.load_model(str(paths["model"]))
    model_prob = float(model.predict_proba(x)[0, 1])
    stored_prob = float(row["score_bonafide"])
    if abs(model_prob - stored_prob) > 1e-4:
        raise ValueError(
            f"Model score ({model_prob:.8f}) differs from saved prediction "
            f"({stored_prob:.8f}); verify feature matrix/order/model version."
        )

    explainer = shap.TreeExplainer(model, model_output="raw")
    raw_values = explainer.shap_values(x, check_additivity=False)
    if isinstance(raw_values, list):
        raw_values = raw_values[-1]
    shap_values = np.asarray(raw_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]
    shap_values = shap_values.reshape(-1)
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = np.asarray(base).reshape(-1)[-1]
    base = float(base)
    reconstructed_margin = base + float(shap_values.sum())
    reconstructed_probability = float(sigmoid(reconstructed_margin))
    booster = model.get_booster()
    # The final refitted model is trained for a fixed selected tree count and
    # therefore may not have XGBoost's best_iteration attribute (which is
    # normally created by early stopping). Use all saved boosting rounds.
    try:
        iteration_end = int(model.best_iteration) + 1
    except (AttributeError, TypeError, ValueError):
        iteration_end = int(booster.num_boosted_rounds())

    margin = float(
        booster.predict(
            xgb.DMatrix(x),
            output_margin=True,
            iteration_range=(0, iteration_end)
        )[0]
    )

    contributions = pd.DataFrame({
        "feature": feature_names,
        "feature_value": x.reshape(-1),
        "shap_value_log_odds": shap_values,
        "direction": np.where(shap_values > 0, "towards_bonafide",
                              np.where(shap_values < 0, "towards_spoof", "neutral"))
    })
    contributions["abs_shap"] = contributions["shap_value_log_odds"].abs()
    contributions = contributions.sort_values("abs_shap", ascending=False)
    csv_path = out_dir / f"{utterance_id}_feature_contributions.csv"
    contributions.to_csv(csv_path, index=False)

    # SHAP waterfall: exact local additive decomposition in raw-margin space.
    explanation = shap.Explanation(
        values=shap_values,
        base_values=base,
        data=x.reshape(-1),
        feature_names=feature_names
    )
    plt.figure(figsize=(11, 8))
    shap.plots.waterfall(explanation, max_display=args.top_k, show=False)
    plot_path = out_dir / f"{utterance_id}_waterfall.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Top positive/negative evidence and group totals.
    positive = contributions[contributions["shap_value_log_odds"] > 0].head(args.top_k)
    negative = (contributions[contributions["shap_value_log_odds"] < 0]
                .sort_values("shap_value_log_odds").head(args.top_k))
    group_rows = []
    for group in ("cqcc", "tkeo", "acoustic"):
        g = contributions[contributions["feature"].str.startswith(group + "__")]
        group_rows.append({
            "feature_group": group,
            "feature_count": int(len(g)),
            "sum_shap_log_odds": float(g["shap_value_log_odds"].sum()),
            "sum_abs_shap_log_odds": float(g["abs_shap"].sum()),
        })
    group_df = pd.DataFrame(group_rows)
    group_path = out_dir / f"{utterance_id}_group_contributions.csv"
    group_df.to_csv(group_path, index=False)

    def table(frame):
        if frame.empty:
            return "<p>None</p>"
        return frame[["feature", "feature_value", "shap_value_log_odds"]].to_html(
            index=False, float_format=lambda v: f"{v:.6g}", border=0, classes="data"
        )

    actual = str(row["key"])
    predicted = str(row["predicted_key"])
    outcome = str(row["error_type"])
    condition = " / ".join(str(row.get(c, "unknown")) for c in ("environment", "attack_id"))
    report = f"""<!doctype html><html><head><meta charset="utf-8"><title>SHAP explanation {html.escape(utterance_id)}</title>
<style>body{{font:15px Arial,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#222}}
h1,h2{{margin-top:28px}} .meta{{background:#f2f4f7;padding:16px;border-radius:8px}}
table.data{{border-collapse:collapse;width:100%}} .data th,.data td{{border-bottom:1px solid #ddd;padding:7px;text-align:left}}
img{{max-width:100%;border:1px solid #ddd}} .note{{color:#555;font-size:13px}}</style></head><body>
<h1>Individual XGBoost prediction explanation</h1>
<div class="meta"><b>Utterance:</b> {html.escape(utterance_id)}<br>
<b>Actual label:</b> {html.escape(actual)}<br><b>Predicted label:</b> {html.escape(predicted)}<br>
<b>Outcome:</b> {html.escape(outcome)}<br><b>Environment / attack:</b> {html.escape(condition)}<br>
<b>Bona-fide probability:</b> {model_prob:.8f}<br><b>Development-selected threshold:</b> {threshold:.8f}<br>
<b>Raw margin:</b> {margin:.8f}<br><b>SHAP reconstructed probability:</b> {reconstructed_probability:.8f}<br>
<b>SHAP additivity margin error:</b> {abs(reconstructed_margin-margin):.3g}</div>
<h2>Decision interpretation</h2>
<p>Positive SHAP contributions increase the model's raw margin for the bona-fide class.
Negative contributions decrease it and push the decision towards spoof. The final
classification is bona fide when its predicted probability is at least the threshold.</p>
<h2>SHAP waterfall</h2><img src="{html.escape(plot_path.name)}" alt="SHAP waterfall">
<h2>Strongest contributions towards bona fide</h2>{table(positive)}
<h2>Strongest contributions towards spoof</h2>{table(negative)}
<h2>Feature-group contribution totals</h2>{group_df.to_html(index=False, float_format=lambda v:f"{v:.6g}", border=0, classes="data")}
<p class="note">SHAP values are in raw-margin (log-odds) space. Feature contributions describe this model's behaviour, not causal physical explanations. Feature names such as c042 must be interpreted using the feature-extraction implementation.</p>
</body></html>"""
    html_path = out_dir / f"{utterance_id}_explanation.html"
    html_path.write_text(report, encoding="utf-8")

    summary = {
        "utterance_id": utterance_id,
        "actual_label": actual,
        "predicted_label": predicted,
        "error_type": outcome,
        "environment": row.get("environment"),
        "attack_id": row.get("attack_id"),
        "probability_bonafide": model_prob,
        "threshold_selected_on_dev": threshold,
        "raw_margin": margin,
        "iteration_end_exclusive": iteration_end,
        "shap_base_value": base,
        "sum_shap_values": float(shap_values.sum()),
        "shap_reconstructed_probability": reconstructed_probability,
        "raw_margin_additivity_error": abs(reconstructed_margin - margin),
        "waterfall_png": str(plot_path),
        "explanation_html": str(html_path),
        "feature_contributions_csv": str(csv_path),
        "feature_group_contributions_csv": str(group_path),
    }
    (out_dir / f"{utterance_id}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nOpen report: {html_path}")


if __name__ == "__main__":
    main()
