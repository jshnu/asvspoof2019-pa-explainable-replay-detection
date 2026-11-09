import numpy as np
import pandas as pd
import pytest

from src.features.tkeo import (
    teager_energy_operator,
    extract_tecc,
)
from src.pipeline.tkeo import run_tkeo_extraction
from src.pipeline.cache import cache_path, load_feature_cache


def test_teager_energy_operator_matches_single_tone_identity():
    amplitude = 0.8
    omega = 0.31
    n = np.arange(1000, dtype=np.float64)
    signal = amplitude * np.cos(omega * n)

    energy = teager_energy_operator(signal)
    expected = amplitude ** 2 * np.sin(omega) ** 2

    assert energy.shape == (998,)
    assert np.allclose(energy[20:-20], expected, rtol=1e-10, atol=1e-10)


def test_teager_energy_operator_rejects_too_short_input():
    with pytest.raises(ValueError):
        teager_energy_operator(np.array([1.0, 2.0]))


def test_tecc_output_shape_finite_and_config():
    rng = np.random.default_rng(123)
    waveform = rng.normal(0, 0.05, 16000).astype(np.float64)

    features, details = extract_tecc(
        waveform,
        sample_rate=16000,
        n_filters=12,
        n_ceps=12,
        return_details=True,
    )

    assert features.shape == (36, 98)
    assert np.isfinite(features).all()
    assert details["n_filters"] == 12
    assert details["frame_length_samples"] == 320
    assert details["frame_shift_samples"] == 160


def test_tecc_rejects_nonfinite_waveform():
    waveform = np.zeros(1000)
    waveform[10] = np.nan

    with pytest.raises(ValueError):
        extract_tecc(waveform)


def test_tkeo_runner_writes_and_resumes(tmp_path):
    manifest = pd.DataFrame([{
        "utterance_id": "PA_D_TKEO_TEST",
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
        return np.zeros((120, 98), dtype=np.float64)

    first = run_tkeo_extraction(
        manifest,
        "dev",
        tmp_path,
        audio_loader=fake_loader,
        feature_extractor=fake_extractor,
        workers=1,
    )
    assert first["written"] == 1
    assert first["failed_count"] == 0

    path = cache_path(tmp_path, "tkeo", "dev", "PA_D_TKEO_TEST")
    features, metadata = load_feature_cache(path)
    assert features.shape == (120, 98)
    assert metadata["feature_type"] == "TECC"
    assert metadata["label"] == "bonafide"

    second = run_tkeo_extraction(
        manifest,
        "dev",
        tmp_path,
        audio_loader=fake_loader,
        feature_extractor=fake_extractor,
        workers=1,
    )
    assert second["skipped_cached"] == 1
    assert calls == {"audio": 1, "features": 1}
