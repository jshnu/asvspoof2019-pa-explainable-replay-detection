from __future__ import annotations

from pathlib import Path
import pandas as pd

from .protocol import read_protocol


def build_manifest(protocol_path: str | Path, audio_dir: str | Path, split: str) -> pd.DataFrame:
    protocol_path = Path(protocol_path)
    audio_dir = Path(audio_dir)
    trials = read_protocol(protocol_path)

    rows = []
    for trial in trials:
        rows.append({
            "split": split,
            **trial.to_dict(),
            "audio_path": str(audio_dir / f"{trial.utterance_id}.flac"),
        })
    return pd.DataFrame(rows)


def write_manifest(df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def check_split_overlap(manifests: dict[str, pd.DataFrame]) -> dict:
    ids = {split: set(df["utterance_id"].astype(str)) for split, df in manifests.items()}
    overlaps = {}
    splits = list(ids)
    for i, left in enumerate(splits):
        for right in splits[i + 1:]:
            common = sorted(ids[left] & ids[right])
            if common:
                overlaps[f"{left}__{right}"] = common[:20]
    return {
        "has_overlap": bool(overlaps),
        "overlaps_preview": overlaps,
        "counts": {split: len(values) for split, values in ids.items()},
    }
