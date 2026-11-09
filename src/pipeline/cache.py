from __future__ import annotations

from pathlib import Path
import hashlib
import json
import numpy as np


def cache_path(cache_root: str | Path, feature_name: str, split: str, utterance_id: str) -> Path:
    return Path(cache_root) / "features" / feature_name / split / f"{utterance_id}.npz"


def save_feature_cache(
    path: str | Path,
    features: np.ndarray,
    metadata: dict,
) -> Path:
    """
    Save one utterance's features atomically.

    The temporary file is placed beside the destination, then renamed, so an
    interrupted process does not leave a partially written final cache entry.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp_path,
        features=np.asarray(features),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    tmp_path.replace(path)
    return path


def load_feature_cache(path: str | Path) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as data:
        features = data["features"]
        metadata = json.loads(str(data["metadata_json"].item()))
    return features, metadata


def fingerprint_config(config: dict) -> str:
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
