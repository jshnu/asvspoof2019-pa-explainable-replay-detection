#!/usr/bin/env python3
"""Generate the E7_FINAL paper figures and tables from saved project outputs.

Run from the ASVspoof-Explainable-Detection project root. The script does not
retrain models or recalculate SHAP values; it consumes the saved CSV/JSON/PNG
artifacts produced by the existing E7_FINAL evaluation and SHAP scripts.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

# Publication-oriented, reproducible figure settings.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.6,
})

NAVY = "#24557A"
TEAL = "#278C87"
ORANGE = "#D77A36"
RED = "#B84A4A"
GREY = "#6B7280"
PURPLE = "#6F5AA8"
GROUP_ORDER = ["CQCC", "TECC/TKEO", "RLFCC/acoustic"]
CASE_SPECS = [
    ("PA_E_0047911", "False accept"),
    ("PA_E_0110936", "False reject"),
    ("PA_E_0076063", "Correct accept"),
    ("PA_E_0023389", "Correct reject"),
]
ABLATION_ORDER = [
    "E1_CQCC", "E2_TKEO", "E3_ACOUSTIC", "E4_CQCC_TKEO",
    "E5_CQCC_ACOUSTIC", "E6_TKEO_ACOUSTIC", "E7_ALL",
]
ABLATION_LABELS = [
    "E1\nCQCC", "E2\nTECC", "E3\nAcoustic", "E4\nCQCC+TECC",
    "E5\nCQCC+Acoustic", "E6\nTECC+Acoustic", "E7\nAll groups",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate all E7_FINAL paper figures and CSV tables."
    )
    p.add_argument("--project-root", default=".", help="Project root (default: current directory).")
    p.add_argument("--ablation-csv", default=None,
                   help="Path to ablation_eer_tdcf_comparison.csv. If omitted, common project locations are searched.")
    p.add_argument("--final-model-dir", default="results/xgb_tuning/E7_FINAL/best_model",
                   help="Directory containing xgb_fusion.json, metrics.json and final_eval/.")
    p.add_argument("--global-shap-dir", default="results/explainability/E7_FINAL/global")
    p.add_argument("--class-conditional-dir", default="results/explainability/E7_FINAL/class_conditional")
    p.add_argument("--individual-shap-dir", default="results/explainability/E7_FINAL/individual")
    p.add_argument("--output-dir", default="results/paper_figures/E7_FINAL")
    p.add_argument("--top-k", type=int, default=15, help="Number of individual features in class-specific SHAP plots (default: 15).")
    p.add_argument("--no-zip", action="store_true", help="Do not create a ZIP of the generated package.")
    return p.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def find_ablation_csv(root: Path, explicit: str | None) -> Path:
    if explicit:
        path = resolve(root, explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Ablation CSV not found: {path}")
        return path
    candidates = [
        root / "ablation_eer_tdcf_comparison.csv",
        root / "results" / "ablation_eer_tdcf_comparison.csv",
        root / "results" / "xgb_ablation" / "ablation_eer_tdcf_comparison.csv",
        root / "results" / "xgb_ablation" / "summary" / "ablation_eer_tdcf_comparison.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = "\n".join(f"  - {x}" for x in candidates)
    raise FileNotFoundError(
        "Could not locate ablation_eer_tdcf_comparison.csv. Searched:\n" + searched +
        "\nPass --ablation-csv <path> to specify its location."
    )


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}. Found: {list(df.columns)}")


def read_csv(path: Path, label: str) -> pd.DataFrame:
    require_file(path, label)
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Could not read {label} at {path}: {exc}") from exc


def read_json(path: Path, label: str) -> dict:
    require_file(path, label)
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Could not read {label} at {path}: {exc}") from exc


def savefig(fig: plt.Figure, fig_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(fig_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def group_name(value: object) -> str:
    v = str(value).strip().lower()
    mapping = {
        "cqcc": "CQCC", "tkeo": "TECC/TKEO", "tecc": "TECC/TKEO",
        "tecc/tkeo": "TECC/TKEO", "acoustic": "RLFCC/acoustic",
        "rlfcc": "RLFCC/acoustic", "rlfcc/acoustic": "RLFCC/acoustic",
    }
    return mapping.get(v, str(value))


def dimension_aggregate(df: pd.DataFrame, code_col: str, pos: int,
                        names: dict[str, str]) -> pd.DataFrame:
    z = df.copy()
    z["dimension"] = z[code_col].astype(str).str[pos].str.lower()
    agg = z.groupby("dimension", as_index=False).agg(
        false_accepts=("false_accepts", "sum"),
        spoof_trials=("spoof_trials", "sum"),
    )
    agg["far_percent"] = 100.0 * agg.false_accepts / agg.spoof_trials
    agg["category"] = agg.dimension.map(names)
    return agg.sort_values("dimension").reset_index(drop=True)


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Project root does not exist: {root}")

    ablation_path = find_ablation_csv(root, args.ablation_csv)
    model_dir = resolve(root, args.final_model_dir)
    global_dir = resolve(root, args.global_shap_dir)
    class_dir = resolve(root, args.class_conditional_dir)
    individual_dir = resolve(root, args.individual_shap_dir)
    out_dir = resolve(root, args.output_dir)
    fig_dir, table_dir, source_dir = out_dir / "figures", out_dir / "tables", out_dir / "source_visuals"
    for d in (fig_dir, table_dir, source_dir):
        d.mkdir(parents=True, exist_ok=True)

    eval_dir = model_dir / "final_eval"
    # Core inputs.
    ab = read_csv(ablation_path, "E1-E7 ablation metrics")
    require_columns(ab, ["experiment", "split", "eer_percent", "min_tdcf", "roc_auc"], "Ablation metrics")
    eval_metrics_path = eval_dir / "eval_metrics.json"
    predictions_path = eval_dir / "eval_predictions.csv"
    eval_metrics = read_json(eval_metrics_path, "E7_FINAL evaluation metrics")
    pred = read_csv(predictions_path, "E7_FINAL evaluation predictions")
    require_columns(pred, ["utterance_id", "key", "score_bonafide"], "Evaluation predictions")

    global_groups = read_csv(global_dir / "shap_global_group_importance.csv", "Global SHAP group importance")
    require_columns(global_groups, ["feature_group", "share_of_total_abs_shap_percent", "mean_abs_shap_per_feature"], "Global SHAP group importance")
    bon = read_csv(class_dir / "shap_bonafide_feature_importance.csv", "Bona fide class SHAP feature importance")
    spo = read_csv(class_dir / "shap_spoof_feature_importance.csv", "Spoof class SHAP feature importance")
    groups = read_csv(class_dir / "shap_class_conditional_group_importance.csv", "Class-conditional SHAP group importance")
    require_columns(bon, ["feature", "mean_abs_shap_log_odds", "feature_group"], "Bona fide SHAP features")
    require_columns(spo, ["feature", "mean_abs_shap_log_odds", "feature_group"], "Spoof SHAP features")
    require_columns(groups, ["class", "feature_group", "mean_abs_shap_per_feature", "share_of_class_total_abs_shap_percent"], "Class-conditional SHAP groups")

    # Check key invariants before plotting.
    if len(pred) != int(eval_metrics.get("n_eval", len(pred))):
        raise ValueError(f"Prediction count ({len(pred)}) does not match eval_metrics.json n_eval ({eval_metrics.get('n_eval')}).")
    expected_n = int(eval_metrics.get("n_features", 1350))
    if expected_n != 1350:
        print(f"WARNING: final model metadata says {expected_n} features; expected E7 has 1,350.", file=sys.stderr)
    if not set(pred["key"].astype(str).str.lower().unique()).issubset({"bonafide", "spoof"}):
        raise ValueError("Unexpected class labels in eval_predictions.csv; expected 'bonafide' and 'spoof'.")

    # Save canonical input tables / source extracts.
    ab.to_csv(table_dir / "ablation_metrics_all_splits.csv", index=False)
    ab_eval = ab[ab["split"].astype(str).str.lower() == "eval"].copy()
    if ab_eval.empty:
        raise ValueError("No rows with split='eval' found in ablation_eer_tdcf_comparison.csv")
    ab_eval.to_csv(table_dir / "ablation_metrics_evaluation.csv", index=False)
    global_groups.to_csv(table_dir / "global_shap_group_importance.csv", index=False)
    groups.to_csv(table_dir / "class_conditional_shap_group_importance.csv", index=False)
    bon.nlargest(25, "mean_abs_shap_log_odds").to_csv(table_dir / "top25_shap_features_bonafide.csv", index=False)
    spo.nlargest(25, "mean_abs_shap_log_odds").to_csv(table_dir / "top25_shap_features_spoof.csv", index=False)

    # Figure 01: evaluation EER across E1-E7.
    ab_eval = ab_eval.set_index("experiment")
    missing_exp = [x for x in ABLATION_ORDER if x not in ab_eval.index]
    if missing_exp:
        raise ValueError(f"Ablation evaluation table is missing experiments: {missing_exp}")
    eval_ord = ab_eval.loc[ABLATION_ORDER]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    vals = eval_ord["eer_percent"].astype(float).to_numpy()
    bars = ax.bar(range(len(vals)), vals, color=[GREY] * 6 + [TEAL], width=.68)
    ax.set_xticks(range(len(vals)), ABLATION_LABELS)
    ax.set_ylabel("Equal error rate (%)")
    ax.set_title("Evaluation-set EER across feature ablation experiments")
    ax.set_ylim(0, max(vals) * 1.22)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + max(vals)*.018, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    savefig(fig, fig_dir, "fig01_ablation_eer_evaluation")

    # Figure 02: evaluation min-tDCF across E1-E7.
    vals = eval_ord["min_tdcf"].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    bars = ax.bar(range(len(vals)), vals, color=[GREY] * 6 + [TEAL], width=.68)
    ax.set_xticks(range(len(vals)), ABLATION_LABELS)
    ax.set_ylabel("Minimum t-DCF")
    ax.set_title("Evaluation-set min-tDCF across feature ablation experiments")
    ax.set_ylim(0, max(vals) * 1.22)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + max(vals)*.018, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    savefig(fig, fig_dir, "fig02_ablation_tdcf_evaluation")

    # Figure 03: original E7 vs final tuned E7; no abandoned internal-split attempt.
    orig = eval_ord.loc["E7_ALL"]
    tm = eval_metrics["threshold_independent_metrics"]
    min_tdcf = float(eval_metrics["official_min_tdcf"]["value"])
    comparison = pd.DataFrame([
        {"model": "Original E7", "EER (%)": float(orig["eer_percent"]),
         "min-tDCF": float(orig["min_tdcf"]), "ROC-AUC": float(orig["roc_auc"])},
        {"model": "Tuned E7 (E7_FINAL)", "EER (%)": float(tm["eer_percent"]),
         "min-tDCF": min_tdcf, "ROC-AUC": float(tm["roc_auc_bonafide_positive"])},
    ])
    comparison.to_csv(table_dir / "original_vs_final_e7_evaluation.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.7))
    for ax, metric, title, fmt in zip(
        axes, ["EER (%)", "min-tDCF", "ROC-AUC"], ["EER", "min-tDCF", "ROC-AUC"],
        ["{:.3f}%", "{:.5f}", "{:.6f}"]
    ):
        mvals = comparison[metric].to_numpy(float)
        if metric == "ROC-AUC":
            base = min(mvals) - max(0.00035, abs(max(mvals)-min(mvals))*2.0)
            bars = ax.bar(comparison.model, mvals-base, bottom=base, color=[GREY, TEAL], width=.58)
            ax.set_ylim(base, max(mvals) + max(0.00055, abs(max(mvals)-min(mvals))*2.5))
            ax.set_ylabel("ROC-AUC (truncated axis; values labelled)")
        else:
            bars = ax.bar(comparison.model, mvals, color=[GREY, TEAL], width=.58)
            ax.set_ylim(0, max(mvals)*1.25 if max(mvals) > 0 else 1)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=12)
        for b, v in zip(bars, mvals):
            ax.annotate(fmt.format(v), (b.get_x()+b.get_width()/2, v), xytext=(0,5),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8)
        ax.grid(axis="x", visible=False)
    savefig(fig, fig_dir, "fig03_original_vs_tuned_e7")

    # Figure 04: evaluation ROC (bona fide is the positive class).
    y = (pred["key"].astype(str).str.lower() == "bonafide").astype(int).to_numpy()
    scores = pd.to_numeric(pred["score_bonafide"], errors="coerce").to_numpy(float)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite score_bonafide values found in evaluation predictions.")
    fpr, tpr, _ = roc_curve(y, scores)
    roc_auc_value = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.plot(fpr, tpr, color=TEAL, lw=2, label=f"E7_FINAL (AUC = {roc_auc_value:.5f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color=GREY, lw=1, label="Chance")
    ax.set(xlim=(0, 1), ylim=(0, 1.02),
           xlabel="False positive rate (spoof misclassified as bona fide)",
           ylabel="True positive rate (bona fide correctly accepted)",
           title="ROC curve on the ASVspoof 2019 PA evaluation set")
    ax.legend(loc="lower right")
    savefig(fig, fig_dir, "fig04_roc_evaluation")

    # Figure 05: confusion matrix at development-selected fixed threshold.
    fx = eval_metrics["fixed_threshold"]
    cm = np.asarray(fx["confusion_matrix"], dtype=int)
    if cm.shape != (2, 2):
        raise ValueError(f"Expected 2x2 confusion matrix; got shape {cm.shape}")
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Predicted spoof", "Predicted bona fide"])
    ax.set_yticks([0, 1], ["Actual spoof", "Actual bona fide"])
    ax.set_title("E7_FINAL confusion matrix at the dev-selected threshold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()*.55 else "#18212B",
                    fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04, label="Number of recordings")
    savefig(fig, fig_dir, "fig05_confusion_matrix_fixed_dev_threshold")
    pd.DataFrame(cm, index=["actual_spoof", "actual_bonafide"],
                 columns=["predicted_spoof", "predicted_bonafide"]).to_csv(table_dir / "final_e7_confusion_matrix.csv")

    # Figure 06: global group-level SHAP attribution share.
    gg = global_groups.copy().sort_values("share_of_total_abs_shap_percent")
    color_map = {"CQCC": NAVY, "TECC/TKEO": TEAL, "RLFCC/acoustic": ORANGE}
    colors = [color_map.get(group_name(x), PURPLE) for x in gg.feature_group]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    bars = ax.barh(gg.feature_group, gg.share_of_total_abs_shap_percent, color=colors)
    ax.set_xlabel("Share of total mean absolute SHAP (%)")
    ax.set_title("Global SHAP attribution by feature group")
    ax.set_xlim(0, float(gg.share_of_total_abs_shap_percent.max())*1.25)
    for b, v in zip(bars, gg.share_of_total_abs_shap_percent):
        ax.text(v+.5, b.get_y()+b.get_height()/2, f"{v:.2f}%", va="center", fontsize=8)
    savefig(fig, fig_dir, "fig06_global_shap_group_share")

    # Figure 07: class-conditional group contribution share.
    groups = groups.copy()
    groups["feature_group"] = groups["feature_group"].map(group_name)
    groups["class"] = groups["class"].astype(str).str.lower()
    pshare = groups.pivot(index="feature_group", columns="class", values="share_of_class_total_abs_shap_percent")
    pshare = pshare.reindex(GROUP_ORDER)
    if not {"bonafide", "spoof"}.issubset(pshare.columns) or pshare.isna().any().any():
        raise ValueError("Class-conditional group-share table must contain CQCC, TECC/TKEO and RLFCC/acoustic for both classes.")
    x = np.arange(len(pshare)); width = .34
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    b1 = ax.bar(x-width/2, pshare["bonafide"], width, label="Bona fide", color=NAVY)
    b2 = ax.bar(x+width/2, pshare["spoof"], width, label="Spoof", color=ORANGE)
    ax.set_xticks(x, GROUP_ORDER)
    ax.set_ylabel("Share of class-specific absolute SHAP (%)")
    ax.set_title("Relative feature-group attribution by actual class")
    ax.legend(frameon=False)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+.6, f"{b.get_height():.1f}%", ha="center", fontsize=8)
    ax.set_ylim(0, float(pshare.max().max())*1.2)
    savefig(fig, fig_dir, "fig07_class_conditional_group_share")

    # Figure 08: mean absolute SHAP per feature, normalizing group size.
    pmean = groups.pivot(index="feature_group", columns="class", values="mean_abs_shap_per_feature").reindex(GROUP_ORDER)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    b1 = ax.bar(x-width/2, pmean["bonafide"], width, label="Bona fide", color=NAVY)
    b2 = ax.bar(x+width/2, pmean["spoof"], width, label="Spoof", color=ORANGE)
    ax.set_xticks(x, GROUP_ORDER)
    ax.set_ylabel("Mean absolute SHAP per feature (log-odds)")
    ax.set_title("Per-feature attribution magnitude by actual class")
    ax.legend(frameon=False)
    ax.set_ylim(0, float(pmean.max().max())*1.25)
    savefig(fig, fig_dir, "fig08_class_conditional_mean_abs_per_feature")

    # Figures 09–10: top class-specific SHAP features.
    def top_feature_plot(df: pd.DataFrame, cls: str, stem: str, color: str) -> pd.DataFrame:
        top = df.nlargest(args.top_k, "mean_abs_shap_log_odds").sort_values("mean_abs_shap_log_odds")
        fig, ax = plt.subplots(figsize=(8.4, max(5.2, .34*len(top))))
        bars = ax.barh(top.feature.astype(str), top.mean_abs_shap_log_odds, color=color)
        ax.set_xlabel("Mean absolute SHAP value (log-odds)")
        ax.set_title(f"Top {len(top)} features by mean absolute SHAP — {cls}")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, float(top.mean_abs_shap_log_odds.max())*1.18)
        for b, v in zip(bars, top.mean_abs_shap_log_odds):
            ax.text(v + float(top.mean_abs_shap_log_odds.max())*.01,
                    b.get_y()+b.get_height()/2, f"{v:.3f}", va="center", fontsize=7)
        savefig(fig, fig_dir, stem)
        return top

    top_b = top_feature_plot(bon, "bona fide", "fig09_top_features_bonafide", NAVY)
    top_s = top_feature_plot(spo, "spoof", "fig10_top_features_spoof", ORANGE)
    top_b.to_csv(table_dir / "top_features_bonafide.csv", index=False)
    top_s.to_csv(table_dir / "top_features_spoof.csv", index=False)

    # Figure 11: selected individual cases' signed group SHAP values.
    case_rows = []
    for uid, label in CASE_SPECS:
        path = individual_dir / f"{uid}_group_contributions.csv"
        d = read_csv(path, f"Individual group SHAP contributions for {uid}")
        require_columns(d, ["feature_group", "sum_shap_log_odds"], f"Individual group SHAP ({uid})")
        for _, r in d.iterrows():
            case_rows.append({
                "utterance_id": uid,
                "case": label,
                "feature_group": group_name(r["feature_group"]),
                "net_shap_log_odds": float(r["sum_shap_log_odds"]),
            })
        copy_if_exists(individual_dir / f"{uid}_waterfall.png", source_dir / f"{uid}_waterfall_original.png")
    case_df = pd.DataFrame(case_rows)
    case_df.to_csv(table_dir / "individual_case_group_shap.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4))
    for ax, (uid, label) in zip(axes.flat, CASE_SPECS):
        d = case_df[case_df.utterance_id == uid].set_index("feature_group").reindex(GROUP_ORDER).reset_index()
        colors = [NAVY if v >= 0 else RED for v in d.net_shap_log_odds]
        bars = ax.barh(d.feature_group, d.net_shap_log_odds, color=colors)
        ax.axvline(0, color="#333333", lw=.8)
        ax.set_title(f"{label} — {uid}", fontsize=9)
        ax.set_xlabel("Net SHAP contribution (log-odds)")
        for b, v in zip(bars, d.net_shap_log_odds):
            off = .05 if v >= 0 else -.05
            ax.text(v+off, b.get_y()+b.get_height()/2, f"{v:+.2f}",
                    ha="left" if v >= 0 else "right", va="center", fontsize=7)
        ax.grid(axis="y", visible=False)
    savefig(fig, fig_dir, "fig11_individual_case_group_contributions")

    # Figures 12–15: benchmark-metadata post-hoc FAR breakdowns.
    posthoc = eval_metrics.get("posthoc_metadata_analysis")
    if not isinstance(posthoc, dict):
        raise ValueError("eval_metrics.json has no posthoc_metadata_analysis object.")
    attack = pd.DataFrame(posthoc.get("attack_id_spoof_false_accept_rates", []))
    env = pd.DataFrame(posthoc.get("environment_spoof_false_accept_rates", []))
    require_columns(attack, ["attack_id", "spoof_trials", "false_accepts", "far_percent"], "Attack-wise post-hoc metrics")
    require_columns(env, ["environment", "spoof_trials", "false_accepts", "far_percent"], "Environment-wise post-hoc metrics")
    attack = attack.sort_values("attack_id").reset_index(drop=True)
    env = env.sort_values("environment").reset_index(drop=True)
    attack.to_csv(table_dir / "attack_id_false_accept_rates.csv", index=False)
    env.to_csv(table_dir / "environment_id_false_accept_rates.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    bars = ax.bar(attack.attack_id, attack.far_percent, color=ORANGE, width=.68)
    ax.set_ylabel("False acceptance rate (%)")
    ax.set_xlabel("Attack ID (attacker distance, replay-device quality)")
    ax.set_title("False acceptance rate by replay attack configuration")
    ax.set_ylim(0, float(attack.far_percent.max())*1.2)
    for b, v in zip(bars, attack.far_percent):
        ax.text(b.get_x()+b.get_width()/2, v+max(attack.far_percent)*.015,
                f"{v:.1f}", ha="center", fontsize=7)
    savefig(fig, fig_dir, "fig12_attack_id_far")

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    env_colors = [RED if v >= 10 else ORANGE if v >= 5 else TEAL for v in env.far_percent]
    bars = ax.bar(env.environment, env.far_percent, color=env_colors, width=.78)
    ax.set_ylabel("False acceptance rate (%)")
    ax.set_xlabel("Environment ID (room size, T60, talker-to-ASV distance)")
    ax.set_title("False acceptance rate across the 27 PA acoustic environments")
    ax.tick_params(axis="x", rotation=55)
    ax.set_ylim(0, float(env.far_percent.max())*1.18)
    savefig(fig, fig_dir, "fig13_environment_id_far")

    env2 = env.copy()
    attack2 = attack.copy()
    for df in (env2, attack2):
        df["false_accepts"] = pd.to_numeric(df.false_accepts, errors="raise").astype(int)
        df["spoof_trials"] = pd.to_numeric(df.spoof_trials, errors="raise").astype(int)
    room = dimension_aggregate(env2, "environment", 0, {"a": "2–5 m²", "b": "5–10 m²", "c": "10–20 m²"})
    t60 = dimension_aggregate(env2, "environment", 1, {"a": "50–200 ms", "b": "200–600 ms", "c": "600–1,000 ms"})
    talker_dist = dimension_aggregate(env2, "environment", 2, {"a": "10–50 cm", "b": "50–100 cm", "c": "100–150 cm"})
    attacker_dist = dimension_aggregate(attack2, "attack_id", 0, {"a": "10–50 cm", "b": "50–100 cm", "c": ">100 cm"})
    quality = dimension_aggregate(attack2, "attack_id", 1, {"a": "Perfect", "b": "High", "c": "Low"})
    for name, df in [
        ("environment_room_size_far", room), ("environment_t60_far", t60),
        ("environment_talker_asv_distance_far", talker_dist),
        ("attack_attacker_distance_far", attacker_dist),
        ("attack_replay_device_quality_far", quality),
    ]:
        df.to_csv(table_dir / f"{name}.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.8))
    for ax, df, title in zip(axes, [room, t60, talker_dist],
                             ["Room size", "Reverberation time (T60)", "Talker-to-ASV distance"]):
        bars = ax.bar(df.category, df.far_percent, color=[NAVY, TEAL, ORANGE], width=.65)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("FAR (%)")
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylim(0, float(df.far_percent.max())*1.24)
        for b, v in zip(bars, df.far_percent):
            ax.text(b.get_x()+b.get_width()/2, v+float(df.far_percent.max())*.015,
                    f"{v:.1f}", ha="center", fontsize=7)
        ax.grid(axis="x", visible=False)
    savefig(fig, fig_dir, "fig14_environment_dimension_far")

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.7))
    for ax, df, title in zip(axes, [attacker_dist, quality],
                             ["Attacker-to-talker distance", "Replay-device quality"]):
        bars = ax.bar(df.category, df.far_percent, color=[NAVY, TEAL, ORANGE], width=.65)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("FAR (%)")
        ax.tick_params(axis="x", rotation=15)
        ax.set_ylim(0, float(df.far_percent.max())*1.24)
        for b, v in zip(bars, df.far_percent):
            ax.text(b.get_x()+b.get_width()/2, v+float(df.far_percent.max())*.015,
                    f"{v:.1f}", ha="center", fontsize=7)
        ax.grid(axis="x", visible=False)
    savefig(fig, fig_dir, "fig15_attack_dimension_far")

    # Final metrics and confusion table.
    fixed = eval_metrics["fixed_threshold"]
    metric_rows = [
        ("EER (%)", tm["eer_percent"]),
        ("min-tDCF", min_tdcf),
        ("ROC-AUC", tm["roc_auc_bonafide_positive"]),
        ("Average precision", tm["average_precision_bonafide_positive"]),
        ("Fixed dev-selected threshold", fixed["value"]),
        ("FAR (%)", fixed["far_spoof_percent"]),
        ("FRR (%)", fixed["frr_bonafide_percent"]),
        ("Precision (bona fide)", fixed["precision_bonafide"]),
        ("Recall (bona fide)", fixed["recall_bonafide"]),
        ("F1 (bona fide)", fixed["f1_bonafide"]),
        ("Balanced accuracy", fixed["balanced_accuracy"]),
        ("Accuracy", fixed["accuracy"]),
        ("Evaluation recordings", eval_metrics["n_eval"]),
        ("Bona fide recordings", eval_metrics["class_counts"]["bonafide"]),
        ("Spoof recordings", eval_metrics["class_counts"]["spoof"]),
    ]
    pd.DataFrame(metric_rows, columns=["metric", "value"]).to_csv(table_dir / "final_e7_evaluation_metrics.csv", index=False)

    # Preserve upstream SHAP plots and the source sample manifests.
    source_copies = [
        (global_dir / "shap_global_beeswarm.png", source_dir / "global_shap_beeswarm_original.png"),
        (global_dir / "shap_global_bar.png", source_dir / "global_shap_bar_original.png"),
        (class_dir / "shap_bonafide_beeswarm.png", source_dir / "class_conditional_bonafide_beeswarm_original.png"),
        (class_dir / "shap_spoof_beeswarm.png", source_dir / "class_conditional_spoof_beeswarm_original.png"),
        (class_dir / "shap_bonafide_bar.png", source_dir / "class_conditional_bonafide_bar_original.png"),
        (class_dir / "shap_spoof_bar.png", source_dir / "class_conditional_spoof_bar_original.png"),
        (class_dir / "shap_class_group_share_comparison.png", source_dir / "class_group_share_original.png"),
        (class_dir / "shap_class_group_mean_abs_comparison.png", source_dir / "class_group_mean_abs_original.png"),
        (global_dir / "shap_global_sample.csv", table_dir / "global_shap_sample_ids.csv"),
        (class_dir / "shap_class_conditional_sample.csv", table_dir / "class_conditional_shap_sample_ids.csv"),
    ]
    for src, dst in source_copies:
        copy_if_exists(src, dst)

    # Copy the exact source script to output for reproducibility.
    script_path = Path(__file__).resolve()
    copy_if_exists(script_path, out_dir / "generate_e7_paper_figures.py")

    # Package README with scope and methodological cautions.
    readme = f"""# E7_FINAL paper figure and table package\n\nGenerated by `generate_e7_paper_figures.py`.\n\n## Inputs used\n\n- Ablation CSV: `{ablation_path}`\n- Final model directory: `{model_dir}`\n- Global SHAP directory: `{global_dir}`\n- Class-conditional SHAP directory: `{class_dir}`\n- Individual SHAP directory: `{individual_dir}`\n\n## Figures\n\n- fig01: E1–E7 evaluation EER.\n- fig02: E1–E7 evaluation min-tDCF.\n- fig03: Original E7 versus final tuned E7 (internal-validation experiment excluded).\n- fig04: Evaluation ROC curve (bona fide positive class).\n- fig05: Confusion matrix at the development-selected fixed threshold.\n- fig06: Global SHAP group attribution shares.\n- fig07: Class-conditional SHAP group attribution shares.\n- fig08: Class-conditional mean absolute SHAP per feature.\n- fig09–10: Top class-specific SHAP features.\n- fig11: Signed group contributions for four individual examples.\n- fig12: FAR by replay attack ID.\n- fig13: FAR by acoustic environment ID.\n- fig14: FAR aggregated by room size, T60 and talker-to-ASV distance.\n- fig15: FAR aggregated by attacker distance and replay-device quality.\n\nEach figure is exported as 300-dpi PNG and vector PDF. Original SHAP beeswarm, bar and waterfall plots are copied into `source_visuals/` when available.\n\n## Methodological notes\n\n1. The E7_FINAL model used the official development set for early stopping and hyperparameter selection. Development metrics are model-selection metrics and may be optimistic. The evaluation split was not used for tuning.\n2. The fixed operational threshold was selected on development data. The confusion matrix and FAR/FRR use this threshold. EER and min-tDCF are threshold-swept benchmark metrics.\n3. Global SHAP uses the saved 2,000-sample evaluation subset. Class-conditional SHAP uses separate balanced samples of 1,000 bona fide and 1,000 spoof recordings.\n4. Group attribution share depends on group dimension count; fig08 provides mean absolute attribution per feature as a complementary normalization.\n5. SHAP values explain model associations, not causal physical mechanisms.\n6. Attack/environment metadata is used only for retrospective benchmark error stratification, not as model input or a deployment-time explanation.\n7. The ROC-AUC subplot in fig03 uses a truncated vertical axis so the small difference is visible; use the numerical table for exact comparison.\n\n## Tables\n\nAll numeric tables used for the figures are in `tables/`. `key_results_for_paper.json` contains the core E7_FINAL metrics and SHAP summaries.\n"""
    (out_dir / "README_figures_and_tables.md").write_text(readme, encoding="utf-8")

    summary = {
        "final_model": eval_metrics.get("experiment", "E7_FINAL"),
        "evaluation": {
            **tm,
            "min_tdcf": min_tdcf,
            "n_eval": eval_metrics.get("n_eval"),
            "n_features": eval_metrics.get("n_features"),
        },
        "fixed_threshold": fixed,
        "global_group_importance": global_groups.to_dict(orient="records"),
        "class_conditional_group_importance": groups.to_dict(orient="records"),
        "source_paths": {
            "ablation_csv": str(ablation_path),
            "model_dir": str(model_dir),
            "global_shap_dir": str(global_dir),
            "class_conditional_dir": str(class_dir),
            "individual_shap_dir": str(individual_dir),
        },
    }
    (out_dir / "key_results_for_paper.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figures_png = sorted(fig_dir.glob("*.png"))
    figures_pdf = sorted(fig_dir.glob("*.pdf"))
    tables = sorted(table_dir.glob("*.csv"))
    print("\nE7_FINAL paper-figure generation completed.")
    print(f"Project root: {root}")
    print(f"Output:       {out_dir}")
    print(f"Figures:      {len(figures_png)} PNG and {len(figures_pdf)} PDF")
    print(f"Tables:       {len(tables)} CSV")
    print(f"Global SHAP sample reflected: {len(pd.read_csv(global_dir / 'shap_global_sample.csv')) if (global_dir / 'shap_global_sample.csv').exists() else 'see source metadata'}")
    print("Evaluation recordings:", len(pred))
    print("Final EER (%):", tm["eer_percent"])
    print("Final min-tDCF:", min_tdcf)
    print("Final ROC-AUC:", tm["roc_auc_bonafide_positive"])

    if not args.no_zip:
        zip_path = out_dir.parent / f"{out_dir.name}_figures_and_tables.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for path in out_dir.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(out_dir.parent))
        print(f"ZIP:          {zip_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
