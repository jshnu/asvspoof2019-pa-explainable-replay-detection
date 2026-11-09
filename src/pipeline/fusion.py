from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.pipeline.cache import cache_path, load_feature_cache

SUPPORTED_GROUPS = ("cqcc", "tkeo", "acoustic")
SUPPORTED_STATS = ("mean", "std", "p10", "p50", "p90")


def aggregate_feature_matrix(features: np.ndarray, stats: tuple[str, ...] | list[str]) -> np.ndarray:
    """Convert a (feature_dim, frames) matrix to a fixed utterance vector."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError(f"Expected non-empty (dimensions, frames) matrix, got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Feature matrix contains non-finite values")
    result = []
    for stat in stats:
        if stat == "mean":
            values = np.mean(x, axis=1)
        elif stat == "std":
            values = np.std(x, axis=1)
        elif stat == "p10":
            values = np.percentile(x, 10, axis=1)
        elif stat == "p50":
            values = np.percentile(x, 50, axis=1)
        elif stat == "p90":
            values = np.percentile(x, 90, axis=1)
        else:
            raise ValueError(f"Unsupported aggregation statistic: {stat}")
        result.append(np.asarray(values, dtype=np.float32))
    if not result:
        raise ValueError("At least one aggregation statistic is required")
    vector = np.concatenate(result)
    if not np.isfinite(vector).all():
        raise FloatingPointError("Aggregated feature vector contains non-finite values")
    return vector


def _load_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}. Run the manifest command first.")
    df = pd.read_csv(path, dtype={"utterance_id": str})
    required = {"utterance_id", "key", "attack_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Manifest {path} is missing columns: {missing}")
    if df["utterance_id"].duplicated().any():
        raise ValueError(f"Manifest contains duplicate utterance IDs: {path}")
    invalid = sorted(set(df["key"].astype(str)) - {"bonafide", "spoof"})
    if invalid:
        raise ValueError(f"Unexpected labels in {path}: {invalid}")
    return df


def build_fusion_split(
    manifest_path: str | Path,
    cache_root: str | Path,
    output_dir: str | Path,
    split: str,
    *,
    feature_groups: tuple[str, ...] | list[str] = SUPPORTED_GROUPS,
    stats: tuple[str, ...] | list[str] = SUPPORTED_STATS,
    overwrite: bool = False,
) -> dict:
    """Aggregate frame caches into one row per utterance, using disk-backed arrays."""
    groups = tuple(feature_groups)
    stats = tuple(stats)
    if not groups or len(set(groups)) != len(groups):
        raise ValueError("feature_groups must be non-empty and unique")
    unknown = sorted(set(groups) - set(SUPPORTED_GROUPS))
    if unknown:
        raise ValueError(f"Unsupported feature groups: {unknown}")
    if not stats or len(set(stats)) != len(stats):
        raise ValueError("stats must be non-empty and unique")
    unknown_stats = sorted(set(stats) - set(SUPPORTED_STATS))
    if unknown_stats:
        raise ValueError(f"Unsupported aggregation statistics: {unknown_stats}")

    manifest = _load_manifest(Path(manifest_path))
    cache_root = Path(cache_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve dimensions from the first utterance, then verify all caches exist
    # before allocating output. This avoids leaving a half-built matrix on a
    # common missing-cache error.
    dimensions: dict[str, int] = {}
    missing_paths = []
    for group in groups:
        first = None
        for uid in manifest["utterance_id"].astype(str):
            candidate = cache_path(cache_root, group, split, uid)
            if candidate.is_file():
                first = candidate
                break
        if first is None:
            raise FileNotFoundError(f"No {group} feature caches found for split={split}")
        first_features, _ = load_feature_cache(first)
        if first_features.ndim != 2:
            raise ValueError(f"Cache {first} must contain a 2-D feature matrix")
        dimensions[group] = int(first_features.shape[0])
        for uid in manifest["utterance_id"].astype(str):
            candidate = cache_path(cache_root, group, split, uid)
            if not candidate.is_file():
                missing_paths.append(str(candidate))
                if len(missing_paths) >= 20:
                    break
        if missing_paths:
            break
    if missing_paths:
        raise FileNotFoundError(
            f"Missing feature caches (showing up to 20):\n" + "\n".join(missing_paths)
        )

    names: list[str] = []
    group_slices: dict[str, list[int]] = {}
    cursor = 0
    for group in groups:
        width = dimensions[group] * len(stats)
        group_slices[group] = [cursor, cursor + width]
        for stat in stats:
            names.extend([f"{group}__{stat}__c{i:03d}" for i in range(dimensions[group])])
        cursor += width

    schema = {
        "feature_groups": list(groups), "aggregation_stats": list(stats),
        "dimensions_per_group": dimensions, "group_slices": group_slices,
        "feature_names": names, "n_features": len(names),
    }
    schema_path = output_dir / "feature_schema.json"
    if schema_path.exists() and not overwrite:
        old = json.loads(schema_path.read_text(encoding="utf-8"))
        comparable = {k: schema[k] for k in schema if k != "feature_names"}
        old_comparable = {k: old.get(k) for k in comparable}
        if old_comparable != comparable:
            raise ValueError(
                f"Feature schema mismatch in {schema_path}; use --overwrite or a new output directory"
            )
    else:
        schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    x_path = output_dir / f"X_{split}.npy"
    y_path = output_dir / f"y_{split}.npy"
    metadata_path = output_dir / f"metadata_{split}.csv"
    if not overwrite and x_path.exists() and y_path.exists() and metadata_path.exists():
        raise FileExistsError(
            f"Fusion matrix already exists for {split}: {x_path}. Use overwrite=True to rebuild."
        )
    tmp_x = output_dir / f"X_{split}.tmp.npy"
    X = np.lib.format.open_memmap(
        tmp_x, mode="w+", dtype=np.float32, shape=(len(manifest), len(names))
    )
    y = np.empty(len(manifest), dtype=np.int8)
    metadata_rows = []
    try:
        for row_idx, row in enumerate(tqdm(
            manifest.to_dict(orient="records"), total=len(manifest), desc=f"Fuse {split}"
        )):
            uid = str(row["utterance_id"])
            pieces = []
            for group in groups:
                path = cache_path(cache_root, group, split, uid)
                frame_features, cache_metadata = load_feature_cache(path)
                if frame_features.shape[0] != dimensions[group]:
                    raise ValueError(
                        f"Feature dimension mismatch for {uid}/{group}: "
                        f"{frame_features.shape[0]} vs {dimensions[group]}"
                    )
                cached_uid = cache_metadata.get("utterance_id")
                if cached_uid is not None and str(cached_uid) != uid:
                    raise ValueError(f"Cache identity mismatch: expected {uid}, got {cached_uid}")
                pieces.append(aggregate_feature_matrix(frame_features, stats))
            X[row_idx] = np.concatenate(pieces)
            y[row_idx] = 1 if str(row["key"]) == "bonafide" else 0
            metadata_rows.append({
                "utterance_id": uid,
                "key": str(row["key"]),
                "attack_id": str(row.get("attack_id", "-")),
                "speaker_id": str(row.get("speaker_id", "")),
                "environment": str(row.get("environment", "")),
            })
        X.flush()
        del X
        tmp_x.replace(x_path)
        np.save(y_path, y)
        pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
    except Exception:
        try:
            del X
        except UnboundLocalError:
            pass
        if tmp_x.exists():
            tmp_x.unlink()
        raise

    return {
        "split": split, "utterances": len(manifest), "features": len(names),
        "feature_groups": list(groups), "X_path": str(x_path),
        "y_path": str(y_path), "metadata_path": str(metadata_path),
        "schema_path": str(schema_path),
        "class_counts": {"bonafide": int(np.sum(y == 1)), "spoof": int(np.sum(y == 0))},
    }


def build_fusion_splits(
    manifest_dir: str | Path,
    cache_root: str | Path,
    output_dir: str | Path,
    splits: tuple[str, ...] | list[str],
    *,
    feature_groups: tuple[str, ...] | list[str] = SUPPORTED_GROUPS,
    stats: tuple[str, ...] | list[str] = SUPPORTED_STATS,
    overwrite: bool = False,
) -> list[dict]:
    results = []
    for split in splits:
        results.append(build_fusion_split(
            Path(manifest_dir) / f"PA_{split}.csv", cache_root, output_dir, split,
            feature_groups=feature_groups, stats=stats, overwrite=overwrite,
        ))
    return results


def build_group_feature_matrix(
    manifest_path: str | Path,
    cache_root: str | Path,
    output_dir: str | Path,
    split: str,
    feature_group: str,
    *,
    stats: tuple[str, ...] | list[str] = SUPPORTED_STATS,
    overwrite: bool = False,
) -> dict:
    """Aggregate one feature group's frame caches, allowing cache deletion afterwards."""
    if feature_group not in SUPPORTED_GROUPS:
        raise ValueError(f"Unsupported feature group: {feature_group}")
    stats = tuple(stats)
    if not stats or len(set(stats)) != len(stats) or set(stats) - set(SUPPORTED_STATS):
        raise ValueError(f"Invalid aggregation statistics: {stats}")
    manifest = _load_manifest(Path(manifest_path))
    cache_root, output_dir = Path(cache_root), Path(output_dir)
    group_dir = output_dir / feature_group
    group_dir.mkdir(parents=True, exist_ok=True)

    first = None
    missing = []
    for uid in manifest["utterance_id"].astype(str):
        p = cache_path(cache_root, feature_group, split, uid)
        if p.is_file():
            first = p
            break
    if first is None:
        raise FileNotFoundError(f"No {feature_group} cache files found for {split}")
    first_features, _ = load_feature_cache(first)
    if first_features.ndim != 2:
        raise ValueError(f"Expected 2-D frame features in {first}")
    dim = int(first_features.shape[0])
    for uid in manifest["utterance_id"].astype(str):
        p = cache_path(cache_root, feature_group, split, uid)
        if not p.is_file():
            missing.append(str(p))
            if len(missing) >= 20:
                break
    if missing:
        raise FileNotFoundError("Missing caches (first 20):\n" + "\n".join(missing))

    names = [f"{feature_group}__{stat}__c{i:03d}" for stat in stats for i in range(dim)]
    schema = {
        "feature_group": feature_group, "aggregation_stats": list(stats),
        "frame_feature_dimensions": dim, "feature_names": names,
        "n_features": len(names),
    }
    schema_path = group_dir / "feature_schema.json"
    if schema_path.exists() and not overwrite:
        old = json.loads(schema_path.read_text(encoding="utf-8"))
        if old != schema:
            raise ValueError(f"Feature schema mismatch at {schema_path}; use --overwrite")
    else:
        schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    x_path, y_path = group_dir / f"X_{split}.npy", group_dir / f"y_{split}.npy"
    metadata_path = group_dir / f"metadata_{split}.csv"
    if not overwrite and any(p.exists() for p in (x_path, y_path, metadata_path)):
        raise FileExistsError(f"Group feature matrix already exists for {feature_group}/{split}; use --overwrite")
    tmp_x = group_dir / f"X_{split}.tmp.npy"
    X = np.lib.format.open_memmap(tmp_x, mode="w+", dtype=np.float32,
                                  shape=(len(manifest), len(names)))
    y = np.empty(len(manifest), dtype=np.int8)
    metadata_rows = []
    try:
        for idx, row in enumerate(tqdm(manifest.to_dict(orient="records"),
                                      total=len(manifest), desc=f"Aggregate {feature_group}/{split}")):
            uid = str(row["utterance_id"])
            features, meta = load_feature_cache(cache_path(cache_root, feature_group, split, uid))
            if features.shape[0] != dim:
                raise ValueError(f"Feature dimension mismatch for {uid}: {features.shape[0]} != {dim}")
            if meta.get("utterance_id") is not None and str(meta["utterance_id"]) != uid:
                raise ValueError(f"Cache identity mismatch for {uid}")
            X[idx] = aggregate_feature_matrix(features, stats)
            y[idx] = 1 if str(row["key"]) == "bonafide" else 0
            metadata_rows.append({
                "utterance_id": uid, "key": str(row["key"]),
                "attack_id": str(row.get("attack_id", "-")),
                "speaker_id": str(row.get("speaker_id", "")),
                "environment": str(row.get("environment", "")),
            })
        X.flush()
        del X
        tmp_x.replace(x_path)
        np.save(y_path, y)
        pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
    except Exception:
        try:
            del X
        except UnboundLocalError:
            pass
        if tmp_x.exists():
            tmp_x.unlink()
        raise
    return {
        "split": split, "feature_group": feature_group,
        "utterances": len(manifest), "n_features": len(names),
        "X_path": str(x_path), "y_path": str(y_path),
        "metadata_path": str(metadata_path), "schema_path": str(schema_path),
        "class_counts": {"bonafide": int(np.sum(y == 1)), "spoof": int(np.sum(y == 0))},
    }


def combine_group_feature_matrices(
    group_feature_dir: str | Path,
    output_dir: str | Path,
    splits: tuple[str, ...] | list[str],
    feature_groups: tuple[str, ...] | list[str],
    *,
    overwrite: bool = False,
) -> list[dict]:
    """Join per-group utterance matrices horizontally after strict ID/label checks."""
    groups = tuple(feature_groups)
    if not groups or len(set(groups)) != len(groups) or set(groups) - set(SUPPORTED_GROUPS):
        raise ValueError(f"Invalid feature groups: {groups}")
    group_feature_dir, output_dir = Path(group_feature_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    group_schemas = {}
    group_dims = {}
    common_stats = None
    for group in groups:
        p = group_feature_dir / group / "feature_schema.json"
        if not p.is_file():
            raise FileNotFoundError(f"Group schema missing: {p}")
        schema = json.loads(p.read_text(encoding="utf-8"))
        group_schemas[group] = schema
        group_dims[group] = int(schema["n_features"])
        current_stats = schema.get("aggregation_stats")
        if common_stats is None:
            common_stats = current_stats
        elif current_stats != common_stats:
            raise ValueError(
                f"Aggregation-stat mismatch: {group} uses {current_stats}, expected {common_stats}"
            )

    names = []
    slices = {}
    cursor = 0
    for group in groups:
        gnames = group_schemas[group]["feature_names"]
        if len(gnames) != group_dims[group]:
            raise ValueError(f"Invalid feature schema for {group}")
        names.extend(gnames)
        slices[group] = [cursor, cursor + group_dims[group]]
        cursor += group_dims[group]
    combined_schema = {
        "feature_groups": list(groups),
        "aggregation_stats": common_stats,
        "group_dimensions": group_dims,
        "group_slices": slices,
        "feature_names": names,
        "n_features": len(names),
    }
    schema_path = output_dir / "feature_schema.json"
    if schema_path.exists() and not overwrite:
        old = json.loads(schema_path.read_text(encoding="utf-8"))
        if old != combined_schema:
            raise ValueError(f"Combined schema mismatch at {schema_path}; use --overwrite")
    else:
        schema_path.write_text(json.dumps(combined_schema, indent=2), encoding="utf-8")

    results = []
    for split in splits:
        matrices, labels, metas = [], [], []
        for group in groups:
            base = group_feature_dir / group
            required = [base / f"X_{split}.npy", base / f"y_{split}.npy", base / f"metadata_{split}.csv"]
            missing = [str(p) for p in required if not p.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing group matrices for {group}/{split}: {missing}")
            matrices.append(np.load(required[0], mmap_mode="r"))
            labels.append(np.load(required[1]))
            metas.append(pd.read_csv(required[2], dtype={"utterance_id": str}))
        n_rows = matrices[0].shape[0]
        if any(m.shape[0] != n_rows for m in matrices):
            raise ValueError(f"Row-count mismatch across feature groups for split={split}")
        for i in range(1, len(groups)):
            if not np.array_equal(labels[0], labels[i]):
                raise ValueError(f"Label/order mismatch between {groups[0]} and {groups[i]} for {split}")
            if not np.array_equal(metas[0]["utterance_id"].astype(str), metas[i]["utterance_id"].astype(str)):
                raise ValueError(f"Utterance ordering mismatch between {groups[0]} and {groups[i]} for {split}")
            if not np.array_equal(metas[0]["key"].astype(str), metas[i]["key"].astype(str)):
                raise ValueError(f"Label metadata mismatch between {groups[0]} and {groups[i]} for {split}")

        x_path = output_dir / f"X_{split}.npy"
        y_path = output_dir / f"y_{split}.npy"
        metadata_path = output_dir / f"metadata_{split}.csv"
        if not overwrite and any(p.exists() for p in (x_path, y_path, metadata_path)):
            raise FileExistsError(f"Combined fusion matrix already exists for {split}; use --overwrite")
        tmp_x = output_dir / f"X_{split}.tmp.npy"
        X = np.lib.format.open_memmap(tmp_x, mode="w+", dtype=np.float32,
                                      shape=(n_rows, len(names)))
        col = 0
        for matrix in matrices:
            width = matrix.shape[1]
            X[:, col:col + width] = matrix
            col += width
        X.flush()
        del X
        tmp_x.replace(x_path)
        np.save(y_path, labels[0].astype(np.int8))
        metas[0].to_csv(metadata_path, index=False)
        results.append({"split": split, "utterances": n_rows, "features": len(names),
                        "X_path": str(x_path), "y_path": str(y_path),
                        "metadata_path": str(metadata_path)})
    return results
