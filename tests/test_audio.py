from pathlib import Path
import numpy as np
import soundfile as sf

from src.data.audio import inspect_audio


def test_inspect_audio_valid_file(tmp_path: Path):
    p = tmp_path / "valid.wav"
    sf.write(p, np.zeros(16000, dtype=np.float32), 16000)
    result = inspect_audio(p)
    assert result["exists"] is True
    assert result["readable"] is True
    assert result["sample_rate"] == 16000
    assert result["channels"] == 1
    assert result["issues"] == []
