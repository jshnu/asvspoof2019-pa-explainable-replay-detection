from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator
import pandas as pd

from src.pipeline.cache import cache_path, save_feature_cache


def iter_audio_rows(manifest: pd.DataFrame) -> Iterator[dict]:
    """
    Yield one manifest row at a time. Does not load waveforms or feature arrays.
    """
    for row in manifest.to_dict(orient="records"):
        yield row


def run_feature_group(
    manifest: pd.DataFrame,
    feature_name: str,
    split: str,
    cache_root: str | Path,
    extractor: Callable[[str], object],
    overwrite: bool = False,
) -> dict:
    """
    Generic Phase 2 extraction runner.

    `extractor` accepts an audio path and returns a NumPy-compatible feature
    object. Each utterance is processed and saved independently. This function
    is infrastructure only; actual CQCC/TKEO/acoustic extractors are added later.
    """
    import numpy as np

    written = 0
    skipped = 0
    failed = []

    for row in iter_audio_rows(manifest):
        utterance_id = str(row["utterance_id"])
        destination = cache_path(cache_root, feature_name, split, utterance_id)

        if destination.exists() and not overwrite:
            skipped += 1
            continue

        try:
            features = np.asarray(extractor(str(row["audio_path"])))
            save_feature_cache(
                destination,
                features,
                metadata={
                    "utterance_id": utterance_id,
                    "split": split,
                    "feature_group": feature_name,
                    "label": str(row["key"]),
                    "attack_id": str(row["attack_id"]),
                },
            )
            written += 1
        except Exception as exc:
            failed.append({
                "utterance_id": utterance_id,
                "audio_path": str(row["audio_path"]),
                "error": f"{type(exc).__name__}: {exc}",
            })

    return {
        "feature_group": feature_name,
        "split": split,
        "written": written,
        "skipped_cached": skipped,
        "failed": failed,
    }
