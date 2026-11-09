import numpy as np
import pandas as pd

from src.pipeline.cqcc import run_cqcc_extraction
from src.pipeline.cache import cache_path, load_feature_cache


def test_cqcc_runner_writes_and_resumes(tmp_path):
    manifest = pd.DataFrame([{
        "utterance_id": "PA_D_TEST",
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
        return np.zeros((90, 20), dtype=np.float64)

    first = run_cqcc_extraction(
        manifest, "dev", tmp_path,
        audio_loader=fake_loader,
        feature_extractor=fake_extractor,
        workers=1,
    )
    assert first["written"] == 1
    assert first["failed_count"] == 0
    cached_path = cache_path(tmp_path, "cqcc", "dev", "PA_D_TEST")
    features, metadata = load_feature_cache(cached_path)
    assert features.shape == (90, 20)
    assert metadata["label"] == "bonafide"

    second = run_cqcc_extraction(
        manifest, "dev", tmp_path,
        audio_loader=fake_loader,
        feature_extractor=fake_extractor,
        workers=1,
    )
    assert second["skipped_cached"] == 1
    assert calls == {"audio": 1, "features": 1}
