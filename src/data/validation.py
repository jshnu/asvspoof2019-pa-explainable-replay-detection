from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm

from .audio import inspect_audio
from .manifest import build_manifest


def validate_split(
    protocol_path: Path,
    audio_dir: Path,
    split: str,
    expected_sample_rate: int,
    allowed_channels: tuple[int, ...],
    min_duration_seconds: float,
    max_duration_seconds: float | None,
    decode_check: bool,
    output_dir: Path,
) -> dict:
    manifest = build_manifest(protocol_path, audio_dir, split)
    class_counts = manifest["key"].value_counts().to_dict()

    records = []
    for row in tqdm(
        manifest.itertuples(index=False),
        total=len(manifest),
        desc=f"Validate {split}",
    ):
        record = inspect_audio(
            row.audio_path,
            expected_sample_rate=expected_sample_rate,
            allowed_channels=allowed_channels,
            min_duration_seconds=min_duration_seconds,
            max_duration_seconds=max_duration_seconds,
            decode_check=decode_check,
        )
        record["utterance_id"] = row.utterance_id
        record["key"] = row.key
        record["attack_id"] = row.attack_id
        records.append(record)

    report_df = pd.DataFrame(records)
    issue_counter = Counter()
    for issues in report_df["issues"]:
        issue_counter.update(issues)

    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / f"PA_{split}_audio_validation.csv"
    report_df.to_csv(details_path, index=False)

    summary = {
        "split": split,
        "protocol_path": str(protocol_path),
        "audio_dir": str(audio_dir),
        "num_trials": int(len(manifest)),
        "class_counts": {k: int(v) for k, v in class_counts.items()},
        "files_found": int(report_df["exists"].sum()),
        "files_readable": int(report_df["readable"].sum()),
        "files_with_issues": int((report_df["issues"].map(len) > 0).sum()),
        "issue_counts": dict(issue_counter),
        "sample_rate_counts": {
            str(k): int(v)
            for k, v in report_df["sample_rate"].value_counts(dropna=False).items()
        },
        "channel_counts": {
            str(k): int(v)
            for k, v in report_df["channels"].value_counts(dropna=False).items()
        },
        "duration_seconds": {
            "min": float(report_df["duration_seconds"].min()),
            "median": float(report_df["duration_seconds"].median()),
            "max": float(report_df["duration_seconds"].max()),
        },
        "details_csv": str(details_path),
    }

    summary_path = output_dir / f"PA_{split}_validation.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
