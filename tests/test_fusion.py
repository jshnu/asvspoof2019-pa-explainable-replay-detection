import json
import numpy as np
import pandas as pd
import pytest

from src.pipeline.cache import cache_path, save_feature_cache
from src.pipeline.fusion import aggregate_feature_matrix, build_fusion_split


def test_aggregate_feature_matrix_statistics():
    x = np.array([[1, 2, 3], [2, 4, 6]], dtype=float)
    result = aggregate_feature_matrix(x, ["mean", "std", "p50"])
    assert result.shape == (6,)
    assert np.allclose(result[:2], [2, 4])
    assert np.allclose(result[4:], [2, 4])


def test_build_fusion_split_creates_fixed_rows(tmp_path):
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    manifest = pd.DataFrame([
        {"utterance_id": "u1", "key": "bonafide", "attack_id": "-", "speaker_id": "s1", "environment": "E1"},
        {"utterance_id": "u2", "key": "spoof", "attack_id": "A01", "speaker_id": "s2", "environment": "E2"},
    ])
    manifest.to_csv(manifest_dir / "PA_train.csv", index=False)
    cache_root = tmp_path / "cache"
    for uid, val, label in [("u1", 1.0, "bonafide"), ("u2", 2.0, "spoof")]:
        save_feature_cache(
            cache_path(cache_root, "acoustic", "train", uid),
            np.full((3, 4), val),
            {"utterance_id": uid, "label": label},
        )
    out = tmp_path / "fusion"
    summary = build_fusion_split(
        manifest_dir / "PA_train.csv", cache_root, out, "train",
        feature_groups=["acoustic"], stats=["mean", "std"],
    )
    X = np.load(out / "X_train.npy")
    y = np.load(out / "y_train.npy")
    assert X.shape == (2, 6)
    assert y.tolist() == [1, 0]
    assert summary["features"] == 6
    assert json.loads((out / "feature_schema.json").read_text())["feature_groups"] == ["acoustic"]


def test_build_fusion_fails_on_missing_cache(tmp_path):
    manifest = pd.DataFrame([{"utterance_id": "u1", "key": "bonafide", "attack_id": "-"}])
    manifest_path = tmp_path / "PA_dev.csv"
    manifest.to_csv(manifest_path, index=False)
    with pytest.raises(FileNotFoundError):
        build_fusion_split(manifest_path, tmp_path / "cache", tmp_path / "out", "dev", feature_groups=["tkeo"])


def test_xgb_training_smoke(tmp_path):
    pytest.importorskip("xgboost")
    from src.models.xgb_fusion import train_xgb_fusion

    rng = np.random.default_rng(5)
    # Synthetic separable data, positive class = bona fide.
    Xtr = rng.normal(size=(100, 4)).astype(np.float32)
    ytr = np.array([1] * 40 + [0] * 60, dtype=np.int8)
    Xtr[:40, 0] += 2.0
    Xdv = rng.normal(size=(40, 4)).astype(np.float32)
    ydv = np.array([1] * 16 + [0] * 24, dtype=np.int8)
    Xdv[:16, 0] += 2.0
    Xev = rng.normal(size=(40, 4)).astype(np.float32)
    yev = np.array([1] * 16 + [0] * 24, dtype=np.int8)
    Xev[:16, 0] += 2.0
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    np.save(feature_dir / "X_train.npy", Xtr)
    np.save(feature_dir / "y_train.npy", ytr)
    np.save(feature_dir / "X_dev.npy", Xdv)
    np.save(feature_dir / "y_dev.npy", ydv)
    np.save(feature_dir / "X_eval.npy", Xev)
    np.save(feature_dir / "y_eval.npy", yev)
    names = [f"acoustic__mean__c{i:03d}" for i in range(4)]
    (feature_dir / "feature_schema.json").write_text(json.dumps({"feature_names": names}))
    pd.DataFrame({"utterance_id": [f"d{i}" for i in range(40)], "key": ["bonafide"] * 16 + ["spoof"] * 24}).to_csv(feature_dir / "metadata_dev.csv", index=False)
    pd.DataFrame({"utterance_id": [f"e{i}" for i in range(40)], "key": ["bonafide"] * 16 + ["spoof"] * 24, "attack_id": ["-"] * 16 + ["A01"] * 24}).to_csv(feature_dir / "metadata_eval.csv", index=False)

    result = train_xgb_fusion(
        feature_dir, tmp_path / "model", feature_groups=["acoustic"],
        n_estimators=30, early_stopping_rounds=5, max_depth=2, n_jobs=1, verbose=False,
    )
    assert (tmp_path / "model" / "xgb_fusion.json").is_file()
    assert result["n_features"] == 4
    assert "eval_eer_percent" in result


def test_group_matrices_combine_with_identity_checks(tmp_path):
    from src.pipeline.fusion import build_group_feature_matrix, combine_group_feature_matrices

    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    pd.DataFrame([
        {"utterance_id": "a", "key": "bonafide", "attack_id": "-"},
        {"utterance_id": "b", "key": "spoof", "attack_id": "A01"},
    ]).to_csv(manifest_dir / "PA_dev.csv", index=False)
    cache_root = tmp_path / "cache"
    for group, dim in [("cqcc", 2), ("tkeo", 3)]:
        for uid, value, label in [("a", 1.0, "bonafide"), ("b", -1.0, "spoof")]:
            save_feature_cache(
                cache_path(cache_root, group, "dev", uid),
                np.full((dim, 4), value),
                {"utterance_id": uid, "label": label},
            )
        build_group_feature_matrix(
            manifest_dir / "PA_dev.csv", cache_root, tmp_path / "groups", "dev", group,
            stats=["mean", "std"],
        )
    result = combine_group_feature_matrices(
        tmp_path / "groups", tmp_path / "combined", ["dev"], ["cqcc", "tkeo"]
    )
    X = np.load(tmp_path / "combined" / "X_dev.npy")
    y = np.load(tmp_path / "combined" / "y_dev.npy")
    assert X.shape == (2, (2 + 3) * 2)
    assert y.tolist() == [1, 0]
    assert result[0]["features"] == 10
