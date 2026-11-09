from __future__ import annotations

import json
from pathlib import Path

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
    roc_curve,
)


def _eer_and_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    threshold = float(thresholds[idx])
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, threshold


def _metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(np.int8)
    eer, eer_threshold = _eer_and_threshold(y_true, scores)
    return {
        "eer": eer,
        "eer_percent": 100.0 * eer,
        "eer_threshold_on_this_split": eer_threshold,
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision_bonafide": float(average_precision_score(y_true, scores)),
        "threshold_used": float(threshold),
        "precision_bonafide": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_bonafide": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_bonafide": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix_labels_[spoof,bonafide]": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
    }


def _select_columns(feature_names: list[str], groups: tuple[str, ...]) -> tuple[list[int], list[str]]:
    unknown = sorted(set(groups) - {"cqcc", "tkeo", "acoustic"})
    if unknown:
        raise ValueError(f"Unsupported feature groups: {unknown}")
    indices = [i for i, name in enumerate(feature_names)
               if any(name.startswith(f"{group}__") for group in groups)]
    selected = [feature_names[i] for i in indices]
    if not indices:
        raise ValueError(f"No columns found for feature groups {groups}")
    return indices, selected


def train_xgb_fusion(
    feature_dir: str | Path,
    model_dir: str | Path,
    *,
    feature_groups: tuple[str, ...] | list[str] = ("cqcc", "tkeo", "acoustic"),
    n_estimators: int = 2000,
    max_depth: int = 5,
    learning_rate: float = 0.03,
    early_stopping_rounds: int = 100,
    n_jobs: int = 6,
    device: str = "cpu",
    random_state: int = 42,
    verbose: int = 100,
    use_class_weight: bool = True,
) -> dict:
    """Train on train, early-stop/select threshold on dev, evaluate once on eval.

    Positive class is bona fide (1); spoof is 0. The model's raw score therefore
    follows ASVspoof's convention: larger values indicate bona fide speech.
    """
    from xgboost import XGBClassifier

    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")

    feature_dir = Path(feature_dir)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    schema_path = feature_dir / "feature_schema.json"
    if not schema_path.is_file():
        raise FileNotFoundError(f"Feature schema missing: {schema_path}. Run build-fusion first.")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    all_names = schema["feature_names"]
    indices, selected_names = _select_columns(all_names, tuple(feature_groups))

    required = ["X_train.npy", "y_train.npy", "X_dev.npy", "y_dev.npy"]
    missing = [name for name in required if not (feature_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing fusion matrices: {missing}. Run build-fusion for train and dev.")

    X_train_full = np.load(feature_dir / "X_train.npy", mmap_mode="r")
    X_dev_full = np.load(feature_dir / "X_dev.npy", mmap_mode="r")
    y_train = np.load(feature_dir / "y_train.npy").astype(np.int8)
    y_dev = np.load(feature_dir / "y_dev.npy").astype(np.int8)
    if X_train_full.shape[1] != len(all_names) or X_dev_full.shape[1] != len(all_names):
        raise ValueError("Fusion matrix width does not match feature_schema.json")
    X_train = np.asarray(X_train_full[:, indices], dtype=np.float32)
    X_dev = np.asarray(X_dev_full[:, indices], dtype=np.float32)
    if not np.isfinite(X_train).all() or not np.isfinite(X_dev).all():
        raise ValueError("Train/dev fusion matrix contains non-finite values")
    if len(np.unique(y_train)) != 2 or len(np.unique(y_dev)) != 2:
        raise ValueError("Both train and dev splits must contain bona fide and spoof trials")

    n_positive = int(np.sum(y_train == 1))
    n_negative = int(np.sum(y_train == 0))
    scale_pos_weight = (n_negative / n_positive) if use_class_weight and n_positive else 1.0
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=5.0,
        reg_alpha=0.0,
        gamma=0.0,
        scale_pos_weight=float(scale_pos_weight),
        tree_method="hist",
        device=device,
        n_jobs=int(n_jobs),
        random_state=int(random_state),
        early_stopping_rounds=int(early_stopping_rounds),
    )
    model.fit(X_train, y_train, eval_set=[(X_dev, y_dev)], verbose=verbose)
    # Verify the effective device/updater, rather than trusting the requested setting.
    booster_config = json.loads(model.get_booster().save_config())
    learner = booster_config["learner"]
    effective_device = learner["generic_param"]["device"]
    updater = learner["gradient_booster"]["gbtree_train_param"].get("updater", "")
    if device == "cuda" and (not effective_device.startswith("cuda") or "gpu" not in updater):
        raise RuntimeError(
            f"CUDA training requested but XGBoost used device={effective_device}, updater={updater}"
        )
    print(f"XGBoost trained on {effective_device} (updater={updater})", flush=True)
    dev_scores = model.predict_proba(X_dev)[:, 1]
    dev_eer, dev_threshold = _eer_and_threshold(y_dev, dev_scores)
    dev_metrics = _metrics(y_dev, dev_scores, dev_threshold)

    results = {
        "feature_groups": list(feature_groups),
        "n_features": len(selected_names),
        "feature_names": selected_names,
        "class_mapping": {"spoof": 0, "bonafide": 1},
        "score_convention": "higher score means more bona fide",
        "training_counts": {"bonafide": n_positive, "spoof": n_negative},
        "scale_pos_weight": float(scale_pos_weight),
        "best_iteration": int(getattr(model, "best_iteration", n_estimators - 1)),
        "dev": dev_metrics,
        "threshold_selected_on_dev": float(dev_threshold),
        "xgboost_parameters": model.get_params(),
        "requested_device": device,
        "effective_device": effective_device,
        "training_updater": updater,
    }

    dev_meta_path = feature_dir / "metadata_dev.csv"
    if dev_meta_path.is_file():
        dev_meta = pd.read_csv(dev_meta_path)
        pd.DataFrame({
            "utterance_id": dev_meta["utterance_id"],
            "key": dev_meta["key"],
            "score_bonafide": dev_scores,
            "predicted_key": np.where(dev_scores >= dev_threshold, "bonafide", "spoof"),
        }).to_csv(model_dir / "dev_predictions.csv", index=False)

    eval_x_path, eval_y_path = feature_dir / "X_eval.npy", feature_dir / "y_eval.npy"
    if eval_x_path.is_file() and eval_y_path.is_file():
        X_eval_full = np.load(eval_x_path, mmap_mode="r")
        y_eval = np.load(eval_y_path).astype(np.int8)
        if X_eval_full.shape[1] != len(all_names):
            raise ValueError("Eval matrix width does not match feature_schema.json")
        X_eval = np.asarray(X_eval_full[:, indices], dtype=np.float32)
        if not np.isfinite(X_eval).all():
            raise ValueError("Eval fusion matrix contains non-finite values")
        eval_scores = model.predict_proba(X_eval)[:, 1]
        eval_metrics = _metrics(y_eval, eval_scores, dev_threshold)
        results["eval"] = eval_metrics
        results["eval_eer_threshold_diagnostic_only"] = _eer_and_threshold(y_eval, eval_scores)[1]
        eval_meta_path = feature_dir / "metadata_eval.csv"
        if eval_meta_path.is_file():
            eval_meta = pd.read_csv(eval_meta_path)
            pd.DataFrame({
                "utterance_id": eval_meta["utterance_id"],
                "key": eval_meta["key"],
                "attack_id": eval_meta.get("attack_id", "-"),
                "score_bonafide": eval_scores,
                "predicted_key": np.where(eval_scores >= dev_threshold, "bonafide", "spoof"),
            }).to_csv(model_dir / "eval_predictions.csv", index=False)
            spoof_mask = y_eval == 0
            attack_rows = []
            for attack_id in sorted(set(eval_meta.loc[spoof_mask, "attack_id"].astype(str))):
                mask = spoof_mask & (eval_meta["attack_id"].astype(str).to_numpy() == attack_id)
                if np.any(mask):
                    attack_rows.append({
                        "attack_id": attack_id,
                        "spoof_trials": int(np.sum(mask)),
                        "false_accept_rate_at_dev_threshold": float(np.mean(eval_scores[mask] >= dev_threshold)),
                    })
            if attack_rows:
                pd.DataFrame(attack_rows).to_csv(model_dir / "eval_attack_breakdown.csv", index=False)

    model_path = model_dir / "xgb_fusion.json"
    model.save_model(model_path)
    (model_dir / "feature_names.json").write_text(
        json.dumps(selected_names, indent=2), encoding="utf-8"
    )
    importance = pd.DataFrame({
        "feature": selected_names,
        "importance_gain_normalized": model.feature_importances_,
    }).sort_values("importance_gain_normalized", ascending=False)
    importance.to_csv(model_dir / "feature_importance.csv", index=False)
    (model_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return {
        "model_path": str(model_path),
        "metrics_path": str(model_dir / "metrics.json"),
        "feature_importance_path": str(model_dir / "feature_importance.csv"),
        "dev_eer_percent": 100.0 * dev_eer,
        "eval_eer_percent": results.get("eval", {}).get("eer_percent"),
        "best_iteration": results["best_iteration"],
        "n_features": len(selected_names),
        "feature_groups": list(feature_groups),
        "effective_device": effective_device,
    }
