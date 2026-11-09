from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.audio import load_audio
from src.features.tkeo import extract_tecc
from src.pipeline.cache import (
    cache_path,
    load_feature_cache,
    save_feature_cache,
    fingerprint_config,
)


def _tkeo_parameters(expected_sample_rate: int, feature_config: dict | None = None) -> dict:
    cfg = dict(feature_config or {})
    return {
        "feature_group": "tkeo",
        "algorithm": "TECC",
        "sample_rate": int(expected_sample_rate),
        "preemphasis": float(cfg.get("preemphasis", 0.97)),
        "n_filters": int(cfg.get("n_filters", 40)),
        "fmin": float(cfg.get("fmin", 10.0)),
        "fmax": float(cfg.get("fmax", expected_sample_rate / 2.0)),
        "bandwidth_hz": cfg.get("bandwidth_hz", None),
        "frame_length_ms": float(cfg.get("frame_length_ms", 20.0)),
        "frame_shift_ms": float(cfg.get("frame_shift_ms", 10.0)),
        "n_ceps": int(cfg.get("n_ceps", 40)),
        "delta_window": int(cfg.get("delta_window", 2)),
        "eps": float(cfg.get("eps", 1e-12)),
        "extractor_version": 1,
        "bandwidth_note": (
            "If null, bandwidth equals linear filter-center spacing; "
            "paper does not report bandwidth for the 40-filter Table 2 setting."
        ),
    }


def _extract_tkeo_task(task: dict) -> dict:
    try:
        from threadpoolctl import threadpool_limits
        thread_limit = threadpool_limits(limits=1)
    except ImportError:
        thread_limit = None

    waveform = features = None
    try:
        destination = cache_path(
            task["cache_root"], "tkeo", task["split"], task["utterance_id"]
        )
        if destination.is_file() and not task["overwrite"]:
            try:
                _, metadata = load_feature_cache(destination)
                if metadata.get("config_fingerprint") == task["config_fingerprint"]:
                    return {"status": "skipped", "utterance_id": task["utterance_id"]}
            except Exception:
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
        features = extract_tecc(
            waveform,
            sample_rate=int(sample_rate),
            preemphasis=params["preemphasis"],
            n_filters=params["n_filters"],
            fmin=params["fmin"],
            fmax=params["fmax"],
            bandwidth_hz=params["bandwidth_hz"],
            frame_length_ms=params["frame_length_ms"],
            frame_shift_ms=params["frame_shift_ms"],
            n_ceps=params["n_ceps"],
            delta_window=params["delta_window"],
            eps=params["eps"],
        )
        features = np.asarray(features, dtype=np.float64)
        expected_dim = 3 * params["n_ceps"]
        if features.ndim != 2 or features.shape[0] != expected_dim:
            raise ValueError(
                f"Expected TECC shape ({expected_dim}, frames), got {features.shape}"
            )
        if features.shape[1] == 0 or not np.isfinite(features).all():
            raise ValueError("TECC output is empty or contains non-finite values")

        metadata = {
            "utterance_id": task["utterance_id"],
            "split": task["split"],
            "feature_group": "tkeo",
            "feature_type": "TECC",
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


def run_tkeo_extraction(
    manifest: pd.DataFrame,
    split: str,
    cache_root: str | Path,
    *,
    expected_sample_rate: int = 16000,
    feature_config: dict | None = None,
    overwrite: bool = False,
    limit: int | None = None,
    workers: int = 1,
    audio_loader: Callable = load_audio,
    feature_extractor: Callable = extract_tecc,
) -> dict:
    required = {"utterance_id", "audio_path", "key", "attack_id"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive integer")

    rows = manifest if limit is None else manifest.head(limit)
    params = _tkeo_parameters(expected_sample_rate, feature_config)
    config_fingerprint = fingerprint_config(params)

    tasks = [{
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
    } for row in rows.to_dict(orient="records")]

    written = skipped = total_cache_bytes = 0
    failed: list[dict] = []

    def consume(result: dict) -> None:
        nonlocal written, skipped, total_cache_bytes
        if result["status"] == "written":
            written += 1
            total_cache_bytes += result.get("cache_bytes", 0)
        elif result["status"] == "skipped":
            skipped += 1
        else:
            failed.append(result)

    if workers == 1:
        for task in tqdm(tasks, total=len(tasks), desc=f"TECC {split}"):
            waveform = features = None
            try:
                destination = cache_path(cache_root, "tkeo", split, task["utterance_id"])
                if destination.is_file() and not overwrite:
                    try:
                        _, metadata = load_feature_cache(destination)
                        if metadata.get("config_fingerprint") == config_fingerprint:
                            consume({"status": "skipped"})
                            continue
                    except Exception:
                        pass

                waveform, sample_rate = audio_loader(
                    task["audio_path"], expected_sample_rate=expected_sample_rate
                )
                if int(sample_rate) != expected_sample_rate:
                    raise ValueError(
                        f"Expected {expected_sample_rate} Hz, got {sample_rate} Hz"
                    )
                features = feature_extractor(
                    waveform,
                    sample_rate=int(sample_rate),
                    preemphasis=params["preemphasis"],
                    n_filters=params["n_filters"],
                    fmin=params["fmin"],
                    fmax=params["fmax"],
                    bandwidth_hz=params["bandwidth_hz"],
                    frame_length_ms=params["frame_length_ms"],
                    frame_shift_ms=params["frame_shift_ms"],
                    n_ceps=params["n_ceps"],
                    delta_window=params["delta_window"],
                    eps=params["eps"],
                )
                features = np.asarray(features, dtype=np.float64)
                expected_dim = 3 * params["n_ceps"]
                if features.ndim != 2 or features.shape[0] != expected_dim:
                    raise ValueError(
                        f"Expected TECC shape ({expected_dim}, frames), got {features.shape}"
                    )
                if features.shape[1] == 0 or not np.isfinite(features).all():
                    raise ValueError("TECC output is empty or contains non-finite values")

                metadata = {
                    "utterance_id": task["utterance_id"],
                    "split": split,
                    "feature_group": "tkeo",
                    "feature_type": "TECC",
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
                save_feature_cache(destination, features, metadata)
                consume({
                    "status": "written",
                    "cache_bytes": int(destination.stat().st_size),
                })
            except Exception as exc:
                consume({
                    "status": "failed",
                    "utterance_id": task["utterance_id"],
                    "audio_path": task["audio_path"],
                    "error": f"{type(exc).__name__}: {exc}",
                })
            finally:
                waveform = None
                features = None
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            iterator = executor.map(_extract_tkeo_task, tasks, chunksize=1)
            for result in tqdm(
                iterator,
                total=len(tasks),
                desc=f"TECC {split} ({workers} workers)",
            ):
                consume(result)

    return {
        "feature_group": "tkeo",
        "feature_type": "TECC",
        "split": split,
        "total_requested": len(tasks),
        "written": written,
        "skipped_cached": skipped,
        "failed_count": len(failed),
        "failed": failed,
        "workers": workers,
        "cache_bytes_written": total_cache_bytes,
        "cache_directory": str(Path(cache_root) / "features" / "tkeo" / split),
        "config": params,
        "config_fingerprint": config_fingerprint,
    }


def extract_tkeo_split(
    manifest_path: str | Path,
    split: str,
    cache_root: str | Path,
    **kwargs,
) -> dict:
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path, dtype={"utterance_id": str})
    return run_tkeo_extraction(
        manifest=manifest,
        split=split,
        cache_root=cache_root,
        **kwargs,
    )
