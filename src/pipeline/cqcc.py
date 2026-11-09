from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.audio import load_audio
from src.features.cqt import extract_cqcc
from src.pipeline.cache import (
    cache_path,
    load_feature_cache,
    save_feature_cache,
    fingerprint_config,
)


def _cqcc_parameters(expected_sample_rate: int) -> dict:
    return {
        "feature_group": "cqcc",
        "sample_rate": expected_sample_rate,
        "bins": 96,
        "fmax": expected_sample_rate / 2,
        "fmin": expected_sample_rate / (2 ** 10),
        "d": 16,
        "cf": 29,
        "delta_window": 3,
        "extractor_version": 1,
    }


def _extract_one_task(task: dict) -> dict:
    """
    Process-safe worker. It must remain a module-level function for Windows
    multiprocessing 'spawn'. Each worker owns one utterance at a time.
    """
    # Limit BLAS/OpenMP threads inside each process. Without this, N worker
    # processes can each create several math-library threads and oversubscribe
    # the CPU badly.
    try:
        from threadpoolctl import threadpool_limits
        thread_limit = threadpool_limits(limits=1)
    except ImportError:
        thread_limit = None

    waveform = None
    features = None
    try:
        destination = cache_path(
            task["cache_root"], "cqcc", task["split"], task["utterance_id"]
        )

        if destination.is_file() and not task["overwrite"]:
            try:
                _, metadata = load_feature_cache(destination)
                if metadata.get("config_fingerprint") == task["config_fingerprint"]:
                    return {"status": "skipped", "utterance_id": task["utterance_id"]}
            except Exception:
                # Corrupt or stale cache entries are regenerated.
                pass

        waveform, sample_rate = load_audio(
            task["audio_path"],
            expected_sample_rate=task["expected_sample_rate"],
        )
        if int(sample_rate) != task["expected_sample_rate"]:
            raise ValueError(
                f"Expected {task['expected_sample_rate']} Hz, got {sample_rate} Hz"
            )

        params = task["params"]
        features = extract_cqcc(
            waveform,
            sr=int(sample_rate),
            bins=params["bins"],
            fmax=params["fmax"],
            fmin=params["fmin"],
            d=params["d"],
            cf=params["cf"],
            delta_window=params["delta_window"],
        )
        features = np.asarray(features, dtype=np.float64)

        if features.ndim != 2 or features.shape[0] != 90:
            raise ValueError(f"Expected CQCC shape (90, frames), got {features.shape}")
        if features.shape[1] == 0 or not np.isfinite(features).all():
            raise ValueError("CQCC output is empty or contains non-finite values")

        metadata = {
            "utterance_id": task["utterance_id"],
            "split": task["split"],
            "feature_group": "cqcc",
            "label": task["key"],
            "attack_id": task["attack_id"],
            "audio_path": task["audio_path"],
            "sample_rate": int(sample_rate),
            "waveform_samples": int(waveform.size),
            "duration_seconds": float(waveform.size / sample_rate),
            "feature_shape": list(features.shape),
            "feature_dtype": str(features.dtype),
            "config": params,
            "config_fingerprint": task["config_fingerprint"],
        }
        save_feature_cache(destination, features, metadata)
        return {
            "status": "written",
            "utterance_id": task["utterance_id"],
            "frames": int(features.shape[1]),
            "cache_bytes": int(destination.stat().st_size),
        }

    except Exception as exc:
        return {
            "status": "failed",
            "utterance_id": task["utterance_id"],
            "audio_path": task["audio_path"],
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        waveform = None
        features = None
        if thread_limit is not None:
            thread_limit.restore_original_limits()


def run_cqcc_extraction(
    manifest: pd.DataFrame,
    split: str,
    cache_root: str | Path,
    *,
    expected_sample_rate: int = 16000,
    overwrite: bool = False,
    limit: int | None = None,
    workers: int = 1,
    audio_loader: Callable = load_audio,
    feature_extractor: Callable = extract_cqcc,
) -> dict:
    """
    Extract CQCC features with process-level parallelism.

    workers=1 keeps the injectable serial path used by unit tests. For workers>1,
    independent utterances are processed by separate processes, each with its
    own waveform/features and a single-threaded BLAS/OpenMP pool.
    """
    required = {"utterance_id", "audio_path", "key", "attack_id"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    if workers < 1:
        raise ValueError("workers must be at least 1")

    rows = manifest if limit is None else manifest.head(limit)
    params = _cqcc_parameters(expected_sample_rate)
    config_fingerprint = fingerprint_config(params)

    tasks = []
    for row in rows.to_dict(orient="records"):
        tasks.append({
            "utterance_id": str(row["utterance_id"]),
            "audio_path": str(row["audio_path"]),
            "key": str(row["key"]),
            "attack_id": str(row["attack_id"]),
            "split": split,
            "cache_root": str(cache_root),
            "expected_sample_rate": expected_sample_rate,
            "overwrite": overwrite,
            "params": params,
            "config_fingerprint": config_fingerprint,
        })

    written = 0
    skipped = 0
    failed: list[dict] = []
    total_cache_bytes = 0

    if workers == 1:
        # Serial path retains dependency injection for deterministic unit tests.
        results = []
        for task in tqdm(tasks, total=len(tasks), desc=f"CQCC {split}"):
            waveform = None
            features = None
            try:
                destination = cache_path(cache_root, "cqcc", split, task["utterance_id"])
                if destination.is_file() and not overwrite:
                    try:
                        _, cached_metadata = load_feature_cache(destination)
                        if cached_metadata.get("config_fingerprint") == config_fingerprint:
                            results.append({"status": "skipped"})
                            continue
                    except Exception:
                        pass

                waveform, sample_rate = audio_loader(
                    task["audio_path"], expected_sample_rate=expected_sample_rate
                )
                features = feature_extractor(
                    waveform,
                    sr=int(sample_rate),
                    bins=96,
                    fmax=sample_rate / 2,
                    fmin=sample_rate / (2 ** 10),
                    d=16,
                    cf=29,
                    delta_window=3,
                )
                features = np.asarray(features, dtype=np.float64)
                if int(sample_rate) != expected_sample_rate:
                    raise ValueError(f"Expected {expected_sample_rate} Hz, got {sample_rate} Hz")
                if features.ndim != 2 or features.shape[0] != 90:
                    raise ValueError(f"Expected CQCC shape (90, frames), got {features.shape}")
                if features.shape[1] == 0 or not np.isfinite(features).all():
                    raise ValueError("CQCC output is empty or contains non-finite values")

                metadata = {
                    "utterance_id": task["utterance_id"],
                    "split": split,
                    "feature_group": "cqcc",
                    "label": task["key"],
                    "attack_id": task["attack_id"],
                    "audio_path": task["audio_path"],
                    "sample_rate": int(sample_rate),
                    "waveform_samples": int(waveform.size),
                    "duration_seconds": float(waveform.size / sample_rate),
                    "feature_shape": list(features.shape),
                    "feature_dtype": str(features.dtype),
                    "config": params,
                    "config_fingerprint": config_fingerprint,
                }
                destination = cache_path(cache_root, "cqcc", split, task["utterance_id"])
                save_feature_cache(destination, features, metadata)
                results.append({"status": "written", "cache_bytes": destination.stat().st_size})
            except Exception as exc:
                results.append({
                    "status": "failed",
                    "utterance_id": task["utterance_id"],
                    "audio_path": task["audio_path"],
                    "error": f"{type(exc).__name__}: {exc}",
                })
            finally:
                waveform = None
                features = None

        for result in results:
            if result["status"] == "written":
                written += 1
                total_cache_bytes += result.get("cache_bytes", 0)
            elif result["status"] == "skipped":
                skipped += 1
            else:
                failed.append(result)
    else:
        # ProcessPoolExecutor uses spawn on Windows. The worker is module-level,
        # and each task is self-contained/pickleable.
        with ProcessPoolExecutor(max_workers=workers) as executor:
            iterator = executor.map(_extract_one_task, tasks, chunksize=1)
            for result in tqdm(iterator, total=len(tasks), desc=f"CQCC {split} ({workers} workers)"):
                if result["status"] == "written":
                    written += 1
                    total_cache_bytes += result.get("cache_bytes", 0)
                elif result["status"] == "skipped":
                    skipped += 1
                else:
                    failed.append(result)

    return {
        "feature_group": "cqcc",
        "split": split,
        "total_requested": len(tasks),
        "written": written,
        "skipped_cached": skipped,
        "failed_count": len(failed),
        "failed": failed,
        "workers": workers,
        "cache_bytes_written": total_cache_bytes,
        "cache_directory": str(Path(cache_root) / "features" / "cqcc" / split),
        "config_fingerprint": config_fingerprint,
    }


def extract_cqcc_split(
    manifest_path: str | Path,
    split: str,
    cache_root: str | Path,
    **kwargs,
) -> dict:
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path, dtype={"utterance_id": str})
    return run_cqcc_extraction(
        manifest=manifest,
        split=split,
        cache_root=cache_root,
        **kwargs,
    )
