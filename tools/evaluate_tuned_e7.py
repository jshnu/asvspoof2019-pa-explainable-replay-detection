"""Final, one-pass evaluation for the tuned ASVspoof 2019 PA E7 XGBoost model.

Loads the already-refitted model, evaluates the untouched official eval split,
uses the fixed min-tDCF threshold selected on official dev for operational
classification metrics, and calculates benchmark min-tDCF with the official
ASVspoof 2019 tDCF v1 implementation.

This script never trains or modifies the model and never selects a threshold
from evaluation labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
TDCF_DIR = Path(__file__).resolve().parent / "tdcf"
sys.path.insert(0, str(TDCF_DIR))
import eval_metrics as tdcf_metrics  # official ASVspoof 2019 tDCF v1

COST_MODEL = {
    "Pspoof": 0.05,
    "Ptar": (1.0 - 0.05) * 0.99,
    "Pnon": (1.0 - 0.05) * 0.01,
    "Cmiss_asv": 1.0,
    "Cfa_asv": 10.0,
    "Cmiss_cm": 1.0,
    "Cfa_cm": 10.0,
}
FEATURE_PREFIXES = ("cqcc__", "tkeo__", "acoustic__")


def read_threshold(model_dir: Path, supplied: float | None) -> tuple[float, str]:
    if supplied is not None:
        threshold = float(supplied)
        source = "command_line"
    else:
        metrics_path = model_dir / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(
                f"Cannot find {metrics_path}. Supply --threshold explicitly or provide metrics.json."
            )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("evaluation_split_used_for_tuning") is True:
            raise ValueError("Model metadata says the evaluation split was used for tuning; refusing final evaluation.")
        try:
            threshold = float(metrics["refit_dev_metrics"]["threshold_at_min_tdcf"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "metrics.json must contain refit_dev_metrics.threshold_at_min_tdcf"
            ) from exc
        source = str(metrics_path)
    if not np.isfinite(threshold):
        raise ValueError(f"Non-finite classification threshold: {threshold}")
    return threshold, source


def load_asv_scores(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Official ASV score file not found: {path}")
    arr = np.genfromtxt(path, dtype=str)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected three columns in {path}; got array shape {arr.shape}")
    trial_types = arr[:, 1]
    scores = arr[:, 2].astype(float)
    target = scores[trial_types == "target"]
    nontarget = scores[trial_types == "nontarget"]
    spoof = scores[trial_types == "spoof"]
    if min(target.size, nontarget.size, spoof.size) == 0:
        raise ValueError(f"Missing target/nontarget/spoof score group in {path}")
    return target, nontarget, spoof, {
        "target": int(target.size),
        "nontarget": int(nontarget.size),
        "spoof": int(spoof.size),
    }


def group_spoof_far(metadata: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, field: str) -> list[dict[str, Any]]:
    if field not in metadata.columns:
        return []
    values = metadata[field].fillna("unknown").astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    spoof_mask = y_true == 0
    for value in sorted(set(values[spoof_mask])):
        if value.strip() in {"", "-", "nan", "None"}:
            continue
        mask = spoof_mask & (values == value)
        count = int(mask.sum())
        false_accepts = int(np.sum(y_pred[mask] == 1))
        rows.append({
            field: value,
            "spoof_trials": count,
            "false_accepts": false_accepts,
            "far_percent": (false_accepts / count * 100.0) if count else None,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the saved tuned E7 model on ASVspoof 2019 PA eval.")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--feature-dir", type=Path, default=None,
                        help="Defaults to <project-root>/results/fusion_features")
    parser.add_argument("--model-dir", type=Path,
                        default=ROOT / "results" / "xgb_tuning" / "E7_FINAL" / "best_model")
    parser.add_argument("--asv-dir", type=Path,
                        default=ROOT.parent / "archive" / "PA" / "ASVspoof2019_PA_asv_scores")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Defaults to <model-dir>/final_eval")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Optional override. Normally loaded from the model's dev metrics.json.")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    feature_dir = (args.feature_dir or project_root / "results" / "fusion_features").resolve()
    model_dir = args.model_dir.resolve()
    asv_dir = args.asv_dir.resolve()
    output_dir = (args.output_dir or model_dir / "final_eval").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "xgb_fusion.json"
    schema_path = feature_dir / "feature_schema.json"
    eval_x_path = feature_dir / "X_eval.npy"
    eval_y_path = feature_dir / "y_eval.npy"
    metadata_path = feature_dir / "metadata_eval.csv"
    asv_path = asv_dir / "ASVspoof2019.PA.asv.eval.gi.trl.scores.txt"
    for path in (model_path, schema_path, eval_x_path, eval_y_path, metadata_path, asv_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    threshold, threshold_source = read_threshold(model_dir, args.threshold)

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    feature_names = schema.get("feature_names")
    if not isinstance(feature_names, list):
        raise ValueError("feature_schema.json does not contain feature_names")
    indices = [i for i, name in enumerate(feature_names) if str(name).startswith(FEATURE_PREFIXES)]
    selected_names = [feature_names[i] for i in indices]
    if len(indices) != 1350:
        raise ValueError(f"E7 requires exactly 1,350 features; found {len(indices)}")

    saved_names_path = model_dir / "feature_names.json"
    if saved_names_path.is_file():
        saved_names = json.loads(saved_names_path.read_text(encoding="utf-8"))
        if saved_names != selected_names:
            raise ValueError("Model feature_names.json does not match the E7 columns selected from feature_schema.json")

    # Verify labels and row order against the metadata generated with X_eval.npy.
    metadata = pd.read_csv(metadata_path, dtype={"utterance_id": str})
    required_meta = {"utterance_id", "key"}
    if not required_meta.issubset(metadata.columns):
        raise ValueError(f"{metadata_path} must contain columns {sorted(required_meta)}")
    if metadata["utterance_id"].duplicated().any():
        raise ValueError("metadata_eval.csv contains duplicate utterance IDs")
    keys = metadata["key"].astype(str).to_numpy()
    if not set(np.unique(keys)).issubset({"bonafide", "spoof"}):
        raise ValueError(f"Unexpected evaluation labels: {sorted(set(keys))}")
    y_meta = (keys == "bonafide").astype(np.int8)
    y_eval = np.load(eval_y_path).astype(np.int8)
    if len(y_eval) != len(metadata) or not np.array_equal(y_eval, y_meta):
        raise ValueError("y_eval.npy and metadata_eval.csv labels/order do not match")
    if set(np.unique(y_eval)) != {0, 1}:
        raise ValueError("Evaluation labels must contain spoof=0 and bonafide=1")

    X_full = np.load(eval_x_path, mmap_mode="r")
    if X_full.ndim != 2 or X_full.shape[1] != len(feature_names):
        raise ValueError(f"X_eval shape {X_full.shape} does not match schema width {len(feature_names)}")
    if X_full.shape[0] != len(y_eval):
        raise ValueError("X_eval row count does not match evaluation labels")
    X_eval = np.asarray(X_full[:, indices], dtype=np.float32)
    del X_full
    if not np.isfinite(X_eval).all():
        raise ValueError("Selected evaluation features contain NaN or infinite values")

    from xgboost import XGBClassifier
    model = XGBClassifier()
    model.load_model(str(model_path))
    expected_features = int(model.get_booster().num_features())
    if expected_features != X_eval.shape[1]:
        raise ValueError(f"Saved model expects {expected_features} features but eval matrix has {X_eval.shape[1]}")

    print(f"Model: {model_path}")
    print(f"Evaluation matrix: {X_eval.shape[0]} utterances x {X_eval.shape[1]} E7 features")
    print(f"Class counts: bona fide={int(np.sum(y_eval == 1))}, spoof={int(np.sum(y_eval == 0))}")
    print(f"Fixed bona-fide threshold from dev: {threshold:.12g} ({threshold_source})")
    scores = model.predict_proba(X_eval)[:, 1].astype(float)
    if not np.isfinite(scores).all():
        raise ValueError("Model returned non-finite evaluation scores")

    # Fixed-threshold operational metrics: threshold was selected on dev, not eval.
    y_pred = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_eval, y_pred, labels=[0, 1]).ravel()
    spoof_count = int(np.sum(y_eval == 0))
    bona_count = int(np.sum(y_eval == 1))
    eer, eer_threshold = tdcf_metrics.compute_eer(scores[y_eval == 1], scores[y_eval == 0])

    target_asv, nontarget_asv, spoof_asv, asv_counts = load_asv_scores(asv_path)
    asv_eer, asv_threshold = tdcf_metrics.compute_eer(target_asv, nontarget_asv)
    pfa_asv, pmiss_asv, pmiss_spoof_asv = tdcf_metrics.obtain_asv_error_rates(
        target_asv, nontarget_asv, spoof_asv, asv_threshold
    )
    tdcf_curve, tdcf_thresholds = tdcf_metrics.compute_tDCF(
        scores[y_eval == 1], scores[y_eval == 0],
        pfa_asv, pmiss_asv, pmiss_spoof_asv, COST_MODEL, False,
    )
    tdcf_index = int(np.argmin(tdcf_curve))

    attack_far = group_spoof_far(metadata, y_eval, y_pred, "attack_id")
    environment_far = group_spoof_far(metadata, y_eval, y_pred, "environment")
    predictions = metadata[["utterance_id", "key"]].copy()
    predictions["score_bonafide"] = scores
    predictions["predicted_key_at_dev_threshold"] = np.where(y_pred == 1, "bonafide", "spoof")
    predictions["false_accept"] = ((y_eval == 0) & (y_pred == 1))
    predictions["false_reject"] = ((y_eval == 1) & (y_pred == 0))
    predictions.to_csv(output_dir / "eval_predictions.csv", index=False)

    result: dict[str, Any] = {
        "experiment": "E7_ALL_tuned_XGBoost",
        "evaluation_split_used_for_tuning": False,
        "n_trials_in_prior_search": "see best_config.json",
        "n_eval": int(len(y_eval)),
        "n_features": int(X_eval.shape[1]),
        "class_counts": {"bonafide": bona_count, "spoof": spoof_count},
        "score_convention": "score_bonafide; larger scores indicate bona fide",
        "fixed_threshold": {
            "value": float(threshold),
            "selection": "official dev min-tDCF threshold from refitted model",
            "source": threshold_source,
            "confusion_matrix_order": "[[spoof_correct, spoof_false_accept], [bonafide_false_reject, bonafide_correct]]",
            "true_negative_spoof": int(tn),
            "false_accept_spoof_as_bonafide": int(fp),
            "false_reject_bonafide_as_spoof": int(fn),
            "true_positive_bonafide": int(tp),
            "far_spoof_percent": float(fp / spoof_count * 100.0) if spoof_count else None,
            "frr_bonafide_percent": float(fn / bona_count * 100.0) if bona_count else None,
            "precision_bonafide": float(precision_score(y_eval, y_pred, zero_division=0)),
            "recall_bonafide": float(recall_score(y_eval, y_pred, zero_division=0)),
            "f1_bonafide": float(f1_score(y_eval, y_pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_eval, y_pred)),
            "accuracy": float(accuracy_score(y_eval, y_pred)),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        },
        "threshold_independent_metrics": {
            "eer_percent": float(eer * 100.0),
            "eer_threshold_diagnostic_only": float(eer_threshold),
            "roc_auc_bonafide_positive": float(roc_auc_score(y_eval, scores)),
            "average_precision_bonafide_positive": float(average_precision_score(y_eval, scores)),
        },
        "official_min_tdcf": {
            "value": float(tdcf_curve[tdcf_index]),
            "cm_threshold_at_min_tdcf_diagnostic_only": float(tdcf_thresholds[tdcf_index]),
            "asv_eer_percent": float(asv_eer * 100.0),
            "asv_eer_threshold": float(asv_threshold),
            "asv_pfa_nontarget": float(pfa_asv),
            "asv_pmiss_target": float(pmiss_asv),
            "asv_pmiss_spoof": float(pmiss_spoof_asv),
            "asv_trial_counts": asv_counts,
            "cost_model": COST_MODEL,
            "implementation": "official ASVspoof 2019 tDCF_python_v1 eval_metrics.compute_tDCF",
            "note": "min-tDCF sweeps the evaluation CM score curve for benchmark comparison; its threshold is not the fixed operational threshold.",
        },
        "posthoc_metadata_analysis": {
            "attack_id_spoof_false_accept_rates": attack_far,
            "environment_spoof_false_accept_rates": environment_far,
            "note": "Benchmark metadata is used only to stratify evaluation errors after prediction; it is not a model input or deployment explanation.",
        },
        "artifacts": {
            "predictions_csv": str(output_dir / "eval_predictions.csv"),
            "metrics_json": str(output_dir / "eval_metrics.json"),
        },
    }
    metrics_path = output_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\nFINAL E7 EVALUATION")
    print(f"EER:                 {eer * 100:.4f}%")
    print(f"min-tDCF:            {tdcf_curve[tdcf_index]:.8f}")
    print(f"ROC-AUC:             {result['threshold_independent_metrics']['roc_auc_bonafide_positive']:.6f}")
    print(f"Fixed-dev FAR:       {result['fixed_threshold']['far_spoof_percent']:.4f}%")
    print(f"Fixed-dev FRR:       {result['fixed_threshold']['frr_bonafide_percent']:.4f}%")
    print(f"Fixed-dev threshold: {threshold:.12g}")
    print(f"\nSaved predictions: {output_dir / 'eval_predictions.csv'}")
    print(f"Saved metrics:     {metrics_path}")


if __name__ == "__main__":
    main()
