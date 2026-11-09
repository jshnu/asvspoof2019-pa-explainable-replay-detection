from pathlib import Path
import pytest

from src.data.protocol import read_protocol


def test_reads_valid_protocol(tmp_path: Path):
    p = tmp_path / "protocol.txt"
    p.write_text(
        "spk1 PA_000001 - - bonafide\n"
        "spk2 PA_000002 - AA spoof\n",
        encoding="utf-8",
    )
    trials = read_protocol(p)
    assert len(trials) == 2
    assert trials[0].utterance_id == "PA_000001"
    assert trials[0].key == "bonafide"
    assert trials[1].attack_id == "AA"


def test_rejects_invalid_column_count(tmp_path: Path):
    p = tmp_path / "bad.txt"
    p.write_text("spk1 PA_1 - bonafide\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 5 columns"):
        read_protocol(p)


def test_rejects_duplicate_ids(tmp_path: Path):
    p = tmp_path / "dup.txt"
    p.write_text(
        "spk1 PA_1 - - bonafide\n"
        "spk2 PA_1 - AA spoof\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate utterance"):
        read_protocol(p)
