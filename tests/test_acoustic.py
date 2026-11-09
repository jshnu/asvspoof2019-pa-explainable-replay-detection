import numpy as np
import pandas as pd

from src.features.acoustic import extract_rlfcc, lp_residual_frames
from src.pipeline.acoustic import run_acoustic_extraction
from src.pipeline.cache import cache_path, load_feature_cache


def test_lp_residual_frames_shape_and_finiteness():
    rng = np.random.default_rng(17)
    x = rng.normal(0, 0.05, 16000)
    residual = lp_residual_frames(x)
    assert residual.shape == (79, 400)
    assert np.isfinite(residual).all()


def test_rlfcc_shape_and_finiteness():
    rng = np.random.default_rng(19)
    x = rng.normal(0, 0.05, 16000)
    features, details = extract_rlfcc(x, return_details=True)
    assert features.shape == (60, 79)
    assert np.isfinite(features).all()
    assert details["lp_order"] == 12
    assert details["frame_length_ms"] == 25.0


def test_rlfcc_handles_silence():
    features = extract_rlfcc(np.zeros(16000, dtype=np.float64))
    assert features.shape == (60, 79)
    assert np.isfinite(features).all()


def test_acoustic_runner_writes_and_resumes(tmp_path):
    manifest = pd.DataFrame([{
        "utterance_id": "PA_D_ACOUSTIC_TEST",
        "audio_path": "fake.flac",
        "key": "bonafide",
        "attack_id": "-",
    }])
    calls = {"audio": 0, "features": 0}

    def fake_loader(path, expected_sample_rate):
        calls["audio"] += 1
        return np.zeros(16000, dtype=np.float32), expected_sample_rate

    def fake_extractor(waveform, **kwargs):
        calls["features"] += 1
        return np.zeros((60, 79), dtype=np.float64)

    first = run_acoustic_extraction(
        manifest, "dev", tmp_path, audio_loader=fake_loader,
        feature_extractor=fake_extractor, workers=1,
    )
    assert first["written"] == 1
    assert first["failed_count"] == 0

    path = cache_path(tmp_path, "acoustic", "dev", "PA_D_ACOUSTIC_TEST")
    features, metadata = load_feature_cache(path)
    assert features.shape == (60, 79)
    assert metadata["feature_type"] == "RLFCC"

    second = run_acoustic_extraction(
        manifest, "dev", tmp_path, audio_loader=fake_loader,
        feature_extractor=fake_extractor, workers=1,
    )
    assert second["skipped_cached"] == 1
    assert calls == {"audio": 1, "features": 1}
