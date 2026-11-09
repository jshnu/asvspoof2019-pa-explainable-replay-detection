from pathlib import Path
from src.data.manifest import build_manifest, check_split_overlap


def test_manifest_paths_and_overlap(tmp_path: Path):
    protocol = tmp_path / "train.txt"
    protocol.write_text(
        "spk1 PA_1 - - bonafide\n"
        "spk2 PA_2 - AA spoof\n",
        encoding="utf-8",
    )
    df = build_manifest(protocol, tmp_path / "audio", "train")
    assert len(df) == 2
    assert df.iloc[0]["audio_path"].endswith("PA_1.flac")

    overlap = check_split_overlap({"train": df, "dev": df.iloc[[1]].copy()})
    assert overlap["has_overlap"] is True
