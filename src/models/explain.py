from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def explain_xgb_model(
    feature_dir: str | Path,
    model_dir: str | Path,
    *,
    split: str = "dev",
    max_samples: int = 1000,
    random_state: int = 42,
) -> dict:
    """Compute SHAP TreeExplainer summaries on a bounded sample."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is optional. Install it with `pip install '.[explain]'` "
            "or `pip install shap`."
        ) from exc
    from xgboost import XGBClassifier

    feature_dir = Path(feature_dir)
    model_dir = Path(model_dir)
    model_path = model_dir / "xgb_fusion.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"XGBoost model not found: {model_path}")
    x_path = feature_dir / f"X_{split}.npy"
    if not x_path.is_file():
        raise FileNotFoundError(f"Feature matrix not found: {x_path}")
    schema = json.loads((feature_dir / "feature_schema.json").read_text(encoding="utf-8"))
    selected_names = json.loads((model_dir / "feature_names.json").read_text(encoding="utf-8"))
    all_names = schema["feature_names"]
    index_lookup = {name: i for i, name in enumerate(all_names)}
    indices = [index_lookup[name] for name in selected_names]

    X_full = np.load(x_path, mmap_mode="r")
    n = min(int(max_samples), X_full.shape[0])
    if n < 1:
        raise ValueError("No samples available for SHAP")
    rng = np.random.default_rng(random_state)
    rows = np.sort(rng.choice(X_full.shape[0], size=n, replace=False))
    X = np.asarray(X_full[rows][:, indices], dtype=np.float32)

    model = XGBClassifier()
    model.load_model(model_path)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    mean_abs = np.mean(np.abs(values), axis=0)
    importance = pd.DataFrame({
        "feature": selected_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)
    importance_path = model_dir / f"shap_importance_{split}.csv"
    importance.to_csv(importance_path, index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shap.summary_plot(values, X, feature_names=selected_names, show=False, max_display=25)
    plot_path = model_dir / f"shap_summary_{split}.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close()
    return {
        "samples_explained": n,
        "importance_csv": str(importance_path),
        "summary_plot": str(plot_path),
        "top_features": importance.head(20).to_dict(orient="records"),
    }
