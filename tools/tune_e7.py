"""E7-only XGBoost hyperparameter search using official ASVspoof 2019 PA min-tDCF.

The evaluation split is deliberately never loaded. Each trial fits on ALL
official training data and uses the official development set for early stopping
and hyperparameter selection. This removes the internal training split and
maximizes the data available for fitting, but reuses dev for two decisions and
can increase development-set overfitting. The selected configuration is then
refit on all training data with the dev-selected tree count. The official tDCF
implementation is imported unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from xgboost.callback import EarlyStopping

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "tdcf"))
import eval_metrics as tdcf_metrics  # noqa: E402

COST_MODEL = {
    "Pspoof": 0.05,
    "Ptar": (1.0 - 0.05) * 0.99,
    "Pnon": (1.0 - 0.05) * 0.01,
    "Cmiss_asv": 1.0,
    "Cfa_asv": 10.0,
    "Cmiss_cm": 1.0,
    "Cfa_cm": 10.0,
}

TRIAL_FIELDS = [
    "trial", "status", "min_tdcf", "eer_percent", "roc_auc",
    "threshold_min_tdcf", "threshold_eer", "best_iteration",
    "effective_device", "training_updater", "elapsed_seconds", "error",
    "max_depth", "learning_rate", "min_child_weight", "subsample",
    "colsample_bytree", "gamma", "reg_alpha", "reg_lambda",
    "scale_pos_weight",
]


def _read_asv_scores(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Official ASV score file not found: {path}")
    a = np.genfromtxt(path, dtype=str)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"Expected 3 columns in {path}; got shape {a.shape}")
    trial_type = a[:, 1]
    scores = a[:, 2].astype(float)
    tar, non, spoof = (scores[trial_type == k] for k in ("target", "nontarget", "spoof"))
    if min(tar.size, non.size, spoof.size) == 0:
        raise ValueError(f"Missing target/nontarget/spoof score group in {path}")
    return tar, non, spoof, {
        "target": int(tar.size), "nontarget": int(non.size), "spoof": int(spoof.size)
    }


def _asv_operating_point(asv_path: Path) -> dict[str, Any]:
    tar, non, spoof, counts = _read_asv_scores(asv_path)
    asv_eer, asv_threshold = tdcf_metrics.compute_eer(tar, non)
    pfa, pmiss, pmiss_spoof = tdcf_metrics.obtain_asv_error_rates(
        tar, non, spoof, asv_threshold
    )
    return {
        "pfa_asv": float(pfa),
        "pmiss_asv": float(pmiss),
        "pmiss_spoof_asv": float(pmiss_spoof),
        "asv_eer": float(asv_eer),
        "asv_threshold": float(asv_threshold),
        "asv_counts": counts,
    }


def _load_dev_labels(feature_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    meta_path = feature_dir / "metadata_dev.csv"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Missing {meta_path}. The official tDCF calculation needs the dev CM labels."
        )
    meta = pd.read_csv(meta_path)
    required = {"utterance_id", "key"}
    if not required.issubset(meta.columns):
        raise ValueError(f"{meta_path} must contain columns {sorted(required)}")
    if meta["utterance_id"].duplicated().any():
        raise ValueError("Duplicate utterance_id values in metadata_dev.csv")
    if not set(meta["key"].astype(str).unique()).issubset({"bonafide", "spoof"}):
        raise ValueError("metadata_dev.csv key values must be bonafide or spoof")
    y = (meta["key"].astype(str).to_numpy() == "bonafide").astype(np.int8)
    return meta["utterance_id"].astype(str).to_numpy(), y


def _min_tdcf_and_threshold(
    y_true: np.ndarray, scores: np.ndarray, asv: dict[str, Any]
) -> tuple[float, float]:
    bona = scores[y_true == 1]
    spoof = scores[y_true == 0]
    curve, thresholds = tdcf_metrics.compute_tDCF(
        bona,
        spoof,
        asv["pfa_asv"],
        asv["pmiss_asv"],
        asv["pmiss_spoof_asv"],
        COST_MODEL,
        False,
    )
    idx = int(np.argmin(curve))
    return float(curve[idx]), float(thresholds[idx])


def _make_min_tdcf_metric(asv: dict[str, Any]):
    """Create an XGBoost sklearn-API metric minimized at every boosting round."""
    def min_tdcf_eval(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=np.int8)
        y_pred = np.asarray(y_pred, dtype=float)
        # The official v1 routine rejects fewer than three unique CM scores.
        # Early boosting rounds can temporarily produce tied/constant scores;
        # assign the no-useful-CM baseline cost until a valid score curve exists.
        if np.unique(y_pred).size < 3:
            return 1.0
        value, _ = _min_tdcf_and_threshold(y_true, y_pred, asv)
        return value

    # XGBoost uses the callable name in evals_result and callback metric selection.
    min_tdcf_eval.__name__ = "min_tdcf"
    return min_tdcf_eval


def _eer_and_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    eer, threshold = tdcf_metrics.compute_eer(scores[y_true == 1], scores[y_true == 0])
    return float(eer), float(threshold)


def _classification_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (scores >= threshold).astype(np.int8)
    return {
        "threshold": float(threshold),
        "precision_bonafide": float(precision_score(y, pred, zero_division=0)),
        "recall_bonafide": float(recall_score(y, pred, zero_division=0)),
        "f1_bonafide": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "confusion_matrix_labels_spoof_bonafide": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def _sample_params(rng: np.random.Generator, ratio: float) -> dict[str, Any]:
    # Log sampling for learning rate and regularization; broad but bounded ranges.
    return {
        "max_depth": int(rng.integers(3, 9)),
        "learning_rate": float(np.exp(rng.uniform(np.log(0.01), np.log(0.10)))),
        "min_child_weight": float(rng.choice([1, 2, 3, 5, 7, 10, 15])),
        "subsample": float(rng.uniform(0.65, 1.0)),
        "colsample_bytree": float(rng.uniform(0.55, 1.0)),
        "gamma": float(rng.choice([0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0])),
        "reg_alpha": float(rng.choice([0.0, 1e-4, 1e-3, 0.01, 0.1, 0.5, 1.0])),
        "reg_lambda": float(np.exp(rng.uniform(np.log(1.0), np.log(20.0)))),
        # Search around the observed class ratio rather than hard-coding 9.
        "scale_pos_weight": float(ratio * rng.uniform(0.60, 1.40)),
    }


def _load_completed(csv_path: Path) -> dict[int, dict[str, str]]:
    if not csv_path.is_file():
        return {}
    out: dict[int, dict[str, str]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                if row.get("status") == "ok":
                    out[int(row["trial"])] = row
            except (ValueError, TypeError):
                continue
    return out


def _append_trial(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRIAL_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in TRIAL_FIELDS})


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Tune E7 XGBoost on all training data, using official dev for early stopping and trial selection.")
    ap.add_argument("--feature-dir", type=Path, default=ROOT / "results" / "fusion_features")
    ap.add_argument("--asv-dir", type=Path, default=ROOT.parent / "archive" / "PA" / "ASVspoof2019_PA_asv_scores")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "results" / "xgb_tuning" / "E7_FINAL")
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-estimators", type=int, default=10000)
    ap.add_argument("--early-stopping-rounds", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--verbose", type=int, default=100)
    ap.add_argument("--force", action="store_true", help="Ignore prior successful trials and start a fresh search.")
    args = ap.parse_args()

    if args.trials < 1 or args.n_estimators < 1 or args.early_stopping_rounds < 1:
        ap.error("trials, n-estimators and early-stopping-rounds must be positive")
    feature_dir, asv_dir, out_dir = args.feature_dir.resolve(), args.asv_dir.resolve(), args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "trials.csv"
    best_json = out_dir / "best_config.json"

    # Strictly load train and dev only. No eval matrix, labels, metadata, or predictions are read.
    schema = json.loads((feature_dir / "feature_schema.json").read_text(encoding="utf-8"))
    names = schema["feature_names"]
    indices = [i for i, n in enumerate(names) if n.startswith(("cqcc__", "tkeo__", "acoustic__"))]
    if len(indices) != 1350:
        raise ValueError(f"E7 expected 1,350 features; found {len(indices)}")
    for filename in ("X_train.npy", "y_train.npy", "X_dev.npy", "y_dev.npy"):
        if not (feature_dir / filename).is_file():
            raise FileNotFoundError(feature_dir / filename)

    X_train_full = np.load(feature_dir / "X_train.npy", mmap_mode="r")
    X_dev_full = np.load(feature_dir / "X_dev.npy", mmap_mode="r")
    if X_train_full.shape[1] != len(names) or X_dev_full.shape[1] != len(names):
        raise ValueError("Feature matrix width does not match feature_schema.json")
    X_train = np.asarray(X_train_full[:, indices], dtype=np.float32)
    X_dev = np.asarray(X_dev_full[:, indices], dtype=np.float32)
    del X_train_full, X_dev_full
    y_train = np.load(feature_dir / "y_train.npy").astype(np.int8)
    y_dev_matrix = np.load(feature_dir / "y_dev.npy").astype(np.int8)
    _, y_dev_meta = _load_dev_labels(feature_dir)
    if not np.array_equal(y_dev_matrix, y_dev_meta):
        raise ValueError("y_dev.npy and metadata_dev.csv labels/order do not match")
    if not np.isfinite(X_train).all() or not np.isfinite(X_dev).all():
        raise ValueError("Train/dev feature matrices contain NaN or infinite values")
    if set(np.unique(y_train)) != {0, 1} or set(np.unique(y_dev_matrix)) != {0, 1}:
        raise ValueError("Train and dev must both contain spoof=0 and bona fide=1")

    asv_path = asv_dir / "ASVspoof2019.PA.asv.dev.gi.trl.scores.txt"
    asv = _asv_operating_point(asv_path)
    ratio = float(np.sum(y_train == 0) / np.sum(y_train == 1))
    rng = np.random.default_rng(args.seed)
    manifest_path = out_dir / "search_manifest.json"
    if args.force:
        results_csv.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
    completed = _load_completed(results_csv)

    # Deterministic candidate list allows interrupted runs to resume at the same trial IDs.
    candidates = [_sample_params(rng, ratio) for _ in range(args.trials)]
    manifest = {
        "seed": args.seed,
        "seed_policy": "same random_state for every hyperparameter trial and selected-model refit; candidate sampling uses the same fixed seed",
        "early_stopping_metric": "official dev min-tDCF (minimized)",
        "trial_selection_metrics": ["official dev min-tDCF", "dev EER", "dev AUC"],
        "early_stopping_set": "official dev",
        "dev_reuse_policy": "official dev is used both for early stopping within each trial and for hyperparameter ranking; development metrics are selection metrics, not an unbiased final performance estimate",
        "internal_validation_split": False,
        "final_refit_policy": "refit winning hyperparameters on all official training data using the winning trial best_iteration + 1 trees; no evaluation split is used",
        "trials": args.trials,
        "n_estimators": args.n_estimators,
        "early_stopping_rounds": args.early_stopping_rounds,
        "device": args.device,
        "feature_dir": str(feature_dir),
        "asv_score_file": str(asv_path),
        "candidates": candidates,
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise RuntimeError(
                "Existing search_manifest.json does not match this run. "
                "Use a new --output-dir or --force to start a fresh search."
            )
    else:
        _write_json(manifest_path, manifest)
    from xgboost import XGBClassifier

    min_tdcf_metric = _make_min_tdcf_metric(asv)

    print(f"E7 features: {len(indices)} | train: {len(y_train)} (all used for fitting) | dev: {len(y_dev_matrix)}")
    print("WARNING: official dev is used for BOTH early stopping and trial selection. Keep eval untouched; dev results may be optimistic.")
    print(f"Training class ratio spoof/bonafide: {ratio:.6f}")
    print(f"Dev ASV EER: {asv['asv_eer'] * 100:.4f}% | ASV threshold: {asv['asv_threshold']:.8g}")
    print(f"Device: {args.device} | trials: {args.trials} | output: {out_dir}")

    for trial_idx, params in enumerate(candidates, start=1):
        if trial_idx in completed:
            print(f"[{trial_idx}/{args.trials}] already completed; skipping")
            continue
        import time
        start = time.time()
        row: dict[str, Any] = {"trial": trial_idx, "status": "failed", **params}
        try:
            model = XGBClassifier(
                objective="binary:logistic",
                # Stop on official dev min-tDCF; dev is also used to rank trials.
                eval_metric=[min_tdcf_metric, "auc"],
                n_estimators=args.n_estimators,
                callbacks=[EarlyStopping(
                    rounds=args.early_stopping_rounds,
                    metric_name="min_tdcf",
                    data_name="validation_0",
                    maximize=False,
                    save_best=True,
                )],
                tree_method="hist",
                device=args.device,
                n_jobs=args.n_jobs,
                random_state=args.seed,
                verbosity=0,
                **params,
            )
            model.fit(X_train, y_train, eval_set=[(X_dev, y_dev_matrix)], verbose=args.verbose)
            config = json.loads(model.get_booster().save_config())
            learner = config["learner"]
            effective_device = learner["generic_param"]["device"]
            updater = learner["gradient_booster"]["gbtree_train_param"].get("updater", "")
            if args.device == "cuda" and (not effective_device.startswith("cuda") or "gpu" not in updater):
                raise RuntimeError(f"CUDA requested but effective device={effective_device}, updater={updater}")

            scores = model.predict_proba(X_dev)[:, 1]
            min_tdcf, tdcf_threshold = _min_tdcf_and_threshold(y_dev_matrix, scores, asv)
            eer, eer_threshold = _eer_and_threshold(y_dev_matrix, scores)
            row.update({
                "status": "ok",
                "min_tdcf": min_tdcf,
                "eer_percent": eer * 100.0,
                "roc_auc": float(roc_auc_score(y_dev_matrix, scores)),
                "threshold_min_tdcf": tdcf_threshold,
                "threshold_eer": eer_threshold,
                "best_iteration": int(model.best_iteration),
                "effective_device": effective_device,
                "training_updater": updater,
                "elapsed_seconds": round(time.time() - start, 2),
                "error": "",
            })
            print(
                f"[{trial_idx}/{args.trials}] min-tDCF={min_tdcf:.8f} "
                f"EER={eer*100:.4f}% AUC={row['roc_auc']:.6f} "
                f"best_iter={model.best_iteration}", flush=True
            )
            del model
        except Exception as exc:
            row.update({"error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": round(time.time() - start, 2)})
            print(f"[{trial_idx}/{args.trials}] FAILED: {row['error']}", flush=True)
            traceback.print_exc()
        _append_trial(results_csv, row)
        if row.get("status") == "ok":
            completed[trial_idx] = {k: str(v) for k, v in row.items()}

    if not completed:
        raise RuntimeError(f"No successful trials. See {results_csv}")

    successful = []
    for trial_id, row in completed.items():
        successful.append({k: (float(v) if k in {"min_tdcf", "eer_percent", "roc_auc"} else v) for k, v in row.items()})
    successful.sort(key=lambda r: (r["min_tdcf"], r["eer_percent"], -r["roc_auc"]))
    best = successful[0]
    best_params = {
        "max_depth": int(best["max_depth"]),
        "learning_rate": float(best["learning_rate"]),
        "min_child_weight": float(best["min_child_weight"]),
        "subsample": float(best["subsample"]),
        "colsample_bytree": float(best["colsample_bytree"]),
        "gamma": float(best["gamma"]),
        "reg_alpha": float(best["reg_alpha"]),
        "reg_lambda": float(best["reg_lambda"]),
        "scale_pos_weight": float(best["scale_pos_weight"]),
    }
    # Refit the selected configuration on ALL official training data using the
    # tree count selected during the dev-based early-stopping trial. This final
    # refit has a fixed tree count and does not perform another stopping pass.
    selected_tree_count = int(best["best_iteration"]) + 1
    final_model = XGBClassifier(
        objective="binary:logistic", eval_metric="auc",
        n_estimators=selected_tree_count,
        tree_method="hist", device=args.device, n_jobs=args.n_jobs,
        random_state=args.seed, verbosity=0, **best_params,
    )
    final_model.fit(X_train, y_train, verbose=args.verbose)
    final_scores = final_model.predict_proba(X_dev)[:, 1]
    final_min_tdcf, final_tdcf_threshold = _min_tdcf_and_threshold(y_dev_matrix, final_scores, asv)
    final_eer, final_eer_threshold = _eer_and_threshold(y_dev_matrix, final_scores)
    final_model_dir = out_dir / "best_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    final_model.save_model(final_model_dir / "xgb_fusion.json")
    selected_names = [names[i] for i in indices]
    (final_model_dir / "feature_names.json").write_text(json.dumps(selected_names, indent=2), encoding="utf-8")
    dev_ids, _ = _load_dev_labels(feature_dir)
    pd.DataFrame({
        "utterance_id": dev_ids,
        "key": np.where(y_dev_matrix == 1, "bonafide", "spoof"),
        "score_bonafide": final_scores,
        "predicted_key_at_min_tdcf": np.where(final_scores >= final_tdcf_threshold, "bonafide", "spoof"),
    }).to_csv(final_model_dir / "dev_predictions.csv", index=False)
    importance = pd.DataFrame({"feature": selected_names, "importance_gain_normalized": final_model.feature_importances_})
    importance.sort_values("importance_gain_normalized", ascending=False).to_csv(final_model_dir / "feature_importance.csv", index=False)

    summary = {
        "experiment": "E7_ALL",
        "selection_rule": "lowest dev official min-tDCF; ties broken by lower dev EER, then higher dev ROC-AUC",
        "early_stopping_metric": "official dev min-tDCF",
        "early_stopping_note": "Each trial fits on all official training data and early-stops on official dev. The same dev set is also used to rank trials, so dev performance is subject to selection optimism.",
        "internal_validation": None,
        "dev_reuse_warning": "Official dev is used for early stopping and hyperparameter selection. Do not report dev metrics as an unbiased estimate. The official evaluation split remains untouched.",
        "final_refit_note": "The selected configuration is refit on all official training data using the winning trial best_iteration + 1 tree count. No evaluation data is used.",
        "random_seed_policy": "All hyperparameter trials and the selected-model refit use the same random_state for paired comparisons.",
        "primary_metric": "dev min-tDCF",
        "secondary_metric": "dev EER",
        "feature_count": len(selected_names),
        "train_count": int(len(y_train)),
        "dev_count": int(len(y_dev_matrix)),
        "class_mapping": {"spoof": 0, "bonafide": 1},
        "score_convention": "higher score means more bona fide",
        "cost_model": COST_MODEL,
        "dev_asv_operating_point": asv,
        "winning_trial": int(best["trial"]),
        "winning_trial_metrics": {"min_tdcf": float(best["min_tdcf"]), "eer_percent": float(best["eer_percent"]), "roc_auc": float(best["roc_auc"])},
        "refit_dev_metrics": {
            "min_tdcf": final_min_tdcf,
            "eer_percent": final_eer * 100.0,
            "roc_auc": float(roc_auc_score(y_dev_matrix, final_scores)),
            "threshold_at_min_tdcf": final_tdcf_threshold,
            "eer_threshold_diagnostic": final_eer_threshold,
            "classification_at_min_tdcf": _classification_metrics(y_dev_matrix, final_scores, final_tdcf_threshold),
        },
        "selected_tree_count_from_winning_trial": selected_tree_count,
        "final_refit_tree_count": int(final_model.n_estimators),
        "best_parameters": best_params,
        "search": {"trials_requested": args.trials, "successful_trials": len(completed), "seed": args.seed},
        "artifacts": {
            "trials_csv": str(results_csv),
            "model": str(final_model_dir / "xgb_fusion.json"),
            "dev_predictions": str(final_model_dir / "dev_predictions.csv"),
        },
        "evaluation_split_used_for_tuning": False,
        "selection_limitations": "Because official dev is used for both early stopping and ranking 80 configurations, its selected metrics are susceptible to overfitting. The evaluation split is not loaded by this script.",
    }
    _write_json(best_json, summary)
    _write_json(final_model_dir / "metrics.json", summary)
    print("\nBEST E7 CONFIGURATION")
    print(json.dumps({"winning_trial": best["trial"], "dev_min_tdcf": best["min_tdcf"], "dev_eer_percent": best["eer_percent"], "parameters": best_params}, indent=2))
    print(f"\nSaved tuning summary: {best_json}")
    print(f"Saved selected model: {final_model_dir / 'xgb_fusion.json'}")


if __name__ == "__main__":
    main()
