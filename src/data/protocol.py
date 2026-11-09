from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path


VALID_KEYS = {"bonafide", "spoof"}


@dataclass(frozen=True)
class Trial:
    speaker_id: str
    utterance_id: str
    environment: str
    attack_id: str
    key: str

    def to_dict(self) -> dict:
        return asdict(self)


def read_protocol(path: str | Path) -> list[Trial]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Protocol file not found: {path}")

    trials: list[Trial] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) != 5:
                raise ValueError(
                    f"{path}:{line_no}: expected 5 columns, received {len(fields)}"
                )
            trial = Trial(*fields)
            if trial.key not in VALID_KEYS:
                raise ValueError(
                    f"{path}:{line_no}: unexpected class key {trial.key!r}; "
                    f"expected one of {sorted(VALID_KEYS)}"
                )
            if trial.utterance_id in seen_ids:
                raise ValueError(
                    f"{path}:{line_no}: duplicate utterance ID {trial.utterance_id}"
                )
            seen_ids.add(trial.utterance_id)
            trials.append(trial)

    if not trials:
        raise ValueError(f"Protocol contains no trials: {path}")
    return trials
