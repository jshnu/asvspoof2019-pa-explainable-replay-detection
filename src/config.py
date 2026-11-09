from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml


@dataclass(frozen=True)
class SplitPaths:
    split: str
    protocol_path: Path
    audio_dir: Path


@dataclass(frozen=True)
class ProjectConfig:
    raw: dict[str, Any]
    config_path: Path
    project_root: Path
    dataset_root: Path
    access_dir: Path
    processed_dir: Path
    manifest_dir: Path
    validation_dir: Path
    cache_dir: Path

    @property
    def access(self) -> str:
        return str(self.raw["project"].get("access", "PA"))

    @property
    def expected_sample_rate(self) -> int:
        return int(self.raw["audio"].get("expected_sample_rate", 16000))

    @property
    def allowed_channels(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self.raw["audio"].get("allowed_channels", [1, 2]))

    def split_paths(self, split: str) -> SplitPaths:
        if split not in ("train", "dev", "eval"):
            raise ValueError(f"Unsupported split: {split}")
        data_cfg = self.raw["data"]
        overrides = data_cfg.get("overrides", {})
        protocol_override = overrides.get(f"{split}_protocol")
        audio_override = overrides.get(f"{split}_audio_dir")

        protocol_rel = protocol_override or data_cfg["protocols"][split]
        audio_rel = audio_override or data_cfg["audio_dirs"][split]

        return SplitPaths(
            split=split,
            protocol_path=self.access_dir / protocol_rel,
            audio_dir=self.access_dir / audio_rel,
        )


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    required = ("project", "data", "audio", "output")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Missing top-level configuration sections: {missing}")

    project_root = config_path.parent.parent
    dataset_root = Path(raw["data"]["root"]).expanduser()
    access_dir = dataset_root / raw["data"].get("access_dir", raw["project"].get("access", "PA"))

    def resolve_output(value: str) -> Path:
        p = Path(value).expanduser()
        return p if p.is_absolute() else project_root / p

    output = raw["output"]
    return ProjectConfig(
        raw=raw,
        config_path=config_path,
        project_root=project_root,
        dataset_root=dataset_root,
        access_dir=access_dir,
        processed_dir=resolve_output(output["processed_dir"]),
        manifest_dir=resolve_output(output["manifest_dir"]),
        validation_dir=resolve_output(output["validation_dir"]),
        cache_dir=resolve_output(output["cache_dir"]),
    )
