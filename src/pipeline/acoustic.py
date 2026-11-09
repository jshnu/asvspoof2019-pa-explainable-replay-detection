from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.audio import load_audio
from src.features.acoustic import extract_rlfcc
from src.pipeline.cache import (
    cache_path,
    load_feature_cache,
    save_feature_cache,
    fingerprint_config,
)


def _acoustic_parameters(expected_sample_rate: int, feature_config: dict | None = None) -> dict:
    cfg = dict(feature_config or {})
    return {
        "feature_group": "acoustic",
        "algorithm": "RLFCC",
        "sample_rate": int(expected_sample_rate),
        "frame_length_ms": float(cfg.get("frame_length_ms", 25.0)),
        "frame_shift_ms": float(cfg.get("frame_shift_ms", 12.5)),
        "lp_order": int(cfg.get("lp_order", 12)),
        "n_fft": int(cfg.get("n_fft", 512)),
        "n_filters": int(cfg.get("n_filters", 20)),
        "n_ceps": int(cfg.get("n_ceps", 20)),
        "delta_window": int(cfg.get("delta_window", 2)),
        "eps": float(cfg.get("eps", 1e-10)),
        "extractor_version": 1,
        "parameter_note": (
            "PA frame/shift and LP order from Wei et al. (2022) Table 2; "
            "20-filter/20-coefficient LFCC setup inherited from the paper's LFCC description."
        ),
    }


def _extract_acoustic_task(task: dict) -> dict:
    try:
        from threadpoolctl import threadpool_limits
        limits = threadpool_limits(limits=1)
    except ImportError:
        limits = None
    waveform = features = None
    try:
        destination = cache_path(task["cache_root"], "acoustic", task["split"], task["utterance_id"])
        if destination.is_file() and not task["overwrite"]:
            try:
                _, metadata = load_feature_cache(destination)
                if metadata.get("config_fingerprint") == task["config_fingerprint"]:
                    return {"status": "skipped", "utterance_id": task["utterance_id"]}
            except Exception:
                pass

        waveform, sample_rate = load_audio(
            task["audio_path"], expected_sample_rate=task["expected_sample_rate"]
        )
        if int(sample_rate) != task["expected_sample_rate"]:
            raise ValueError(f"Expected {task['expected_sample_rate']} Hz, got {sample_rate} Hz")
        p = task["params"]
        features = extract_rlfcc(
            waveform,
            sample_rate=int(sample_rate),
            frame_length_ms=p["frame_length_ms"],
            frame_shift_ms=p["frame_shift_ms"],
            lp_order=p["lp_order"],
            n_fft=p["n_fft"],
            n_filters=p["n_filters"],
            n_ceps=p["n_ceps"],
            delta_window=p["delta_window"],
            eps=p["eps"],
        )
        features = np.asarray(features, dtype=np.float64)
        expected_dim = 3 * p["n_ceps"]
        if features.ndim != 2 or features.shape[0] != expected_dim or features.shape[1] == 0:
            raise ValueError(f"Expected RLFCC shape ({expected_dim}, frames), got {features.shape}")
        if not np.isfinite(features).all():
            raise ValueError("RLFCC output contains non-finite values")
        metadata = {
            "utterance_id": task["utterance_id"], "split": task["split"],
            "feature_group": "acoustic", "feature_type": "RLFCC",
            "label": task["key"], "attack_id": task["attack_id"],
            "audio_path": task["audio_path"], "sample_rate": int(sample_rate),
            "waveform_samples": int(waveform.size),
            "duration_seconds": float(waveform.size / sample_rate),
            "feature_shape": list(features.shape), "feature_dtype": str(features.dtype),
            "config": p, "config_fingerprint": task["config_fingerprint"],
        }
        save_feature_cache(destination, features, metadata)
        return {"status": "written", "utterance_id": task["utterance_id"],
                "frames": int(features.shape[1]), "cache_bytes": int(destination.stat().st_size)}
    except Exception as exc:
        return {"status": "failed", "utterance_id": task["utterance_id"],
                "audio_path": task["audio_path"], "error": f"{type(exc).__name__}: {exc}"}
    finally:
        waveform = None
        features = None
        if limits is not None:
            limits.restore_original_limits()


def run_acoustic_extraction(
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
    feature_extractor: Callable = extract_rlfcc,
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
    params = _acoustic_parameters(expected_sample_rate, feature_config)
    config_fingerprint = fingerprint_config(params)
    tasks = [{
        "utterance_id": str(row["utterance_id"]), "audio_path": str(row["audio_path"]),
        "key": str(row["key"]), "attack_id": str(row["attack_id"]),
        "split": split, "cache_root": str(cache_root),
        "expected_sample_rate": int(expected_sample_rate), "overwrite": overwrite,
        "params": params, "config_fingerprint": config_fingerprint,
    } for row in rows.to_dict(orient="records")]

    written = skipped = total_bytes = 0
    failed: list[dict] = []

    def consume(result: dict) -> None:
        nonlocal written, skipped, total_bytes
        if result["status"] == "written":
            written += 1
            total_bytes += result.get("cache_bytes", 0)
        elif result["status"] == "skipped":
            skipped += 1
        else:
            failed.append(result)

    if workers == 1:
        # Serial path supports injected loaders/extractors for deterministic tests.
        for task in tqdm(tasks, total=len(tasks), desc=f"RLFCC {split}"):
            waveform = features = None
            try:
                destination = cache_path(cache_root, "acoustic", split, task["utterance_id"])
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
                    raise ValueError(f"Expected {expected_sample_rate} Hz, got {sample_rate} Hz")
                features = feature_extractor(
                    waveform, sample_rate=int(sample_rate),
                    frame_length_ms=params["frame_length_ms"],
                    frame_shift_ms=params["frame_shift_ms"],
                    lp_order=params["lp_order"], n_fft=params["n_fft"],
                    n_filters=params["n_filters"], n_ceps=params["n_ceps"],
                    delta_window=params["delta_window"], eps=params["eps"],
                )
                features = np.asarray(features, dtype=np.float64)
                expected_dim = 3 * params["n_ceps"]
                if features.ndim != 2 or features.shape[0] != expected_dim or features.shape[1] == 0:
                    raise ValueError(f"Expected RLFCC shape ({expected_dim}, frames), got {features.shape}")
                if not np.isfinite(features).all():
                    raise ValueError("RLFCC output contains non-finite values")
                metadata = {
                    "utterance_id": task["utterance_id"], "split": split,
                    "feature_group": "acoustic", "feature_type": "RLFCC",
                    "label": task["key"], "attack_id": task["attack_id"],
                    "audio_path": task["audio_path"], "sample_rate": int(sample_rate),
                    "waveform_samples": int(waveform.size),
                    "duration_seconds": float(waveform.size / sample_rate),
                    "feature_shape": list(features.shape), "feature_dtype": str(features.dtype),
                    "config": params, "config_fingerprint": config_fingerprint,
                }
                save_feature_cache(destination, features, metadata)
                consume({"status": "written", "cache_bytes": int(destination.stat().st_size)})
            except Exception as exc:
                consume({"status": "failed", "utterance_id": task["utterance_id"],
                         "audio_path": task["audio_path"],
                         "error": f"{type(exc).__name__}: {exc}"})
            finally:
                waveform = None
                features = None
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in tqdm(pool.map(_extract_acoustic_task, tasks, chunksize=1),
                               total=len(tasks), desc=f"RLFCC {split} ({workers} workers)"):
                consume(result)

    return {
        "feature_group": "acoustic", "feature_type": "RLFCC", "split": split,
        "total_requested": len(tasks), "written": written, "skipped_cached": skipped,
        "failed_count": len(failed), "failed": failed, "workers": workers,
        "cache_bytes_written": total_bytes,
        "cache_directory": str(Path(cache_root) / "features" / "acoustic" / split),
        "config_fingerprint": config_fingerprint,
    }


def extract_acoustic_split(
    manifest_path: str | Path,
    split: str,
    cache_root: str | Path,
    *,
    expected_sample_rate: int = 16000,
    feature_config: dict | None = None,
    overwrite: bool = False,
    limit: int | None = None,
    workers: int = 1,
) -> dict:
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path, dtype={"utterance_id": str})
    return run_acoustic_extraction(
        manifest, split, cache_root,
        expected_sample_rate=expected_sample_rate,
        feature_config=feature_config, overwrite=overwrite, limit=limit, workers=workers,
    )
