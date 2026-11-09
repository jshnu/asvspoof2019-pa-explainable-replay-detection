#!/usr/bin/env python
"""
Paired statistical comparison of ASVspoof 2019 PA condition-analysis predictions.

Primary comparison: E7_ALL vs E5_CQCC_ACOUSTIC.
Also compares E7 against E1-E6.

Inputs:
  results/condition_analysis/E*_joined_eval_predictions.csv

Each input must contain:
  utterance_id, key, environment, attack_id, score_bonafide, threshold

The threshold column must already be selected on the development set.
A spoof trial is a false accept when score_bonafide >= threshold.

Outputs:
  paired_stats_overall.csv
  paired_stats_by_attack.csv
  paired_stats_by_environment.csv
  paired_stats_by_environment_attack.csv
  paired_stats_metadata.json

Uses paired bootstrap resampling over the same spoof utterances and exact
two-sided McNemar tests. Holm correction is applied within each comparison
and analysis family.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest

EXPERIMENTS = {
    "E1_CQCC": "E1_CQCC_joined_eval_predictions.csv",
    "E2_TKEO": "E2_TKEO_joined_eval_predictions.csv",
    "E3_ACOUSTIC": "E3_ACOUSTIC_joined_eval_predictions.csv",
    "E4_CQCC_TKEO": "E4_CQCC_TKEO_joined_eval_predictions.csv",
    "E5_CQCC_ACOUSTIC": "E5_CQCC_ACOUSTIC_joined_eval_predictions.csv",
    "E6_TKEO_ACOUSTIC": "E6_TKEO_ACOUSTIC_joined_eval_predictions.csv",
    "E7_ALL": "E7_ALL_joined_eval_predictions.csv",
}
BASELINE = "E7_ALL"
COMPARATORS = [k for k in EXPERIMENTS if k != BASELINE]
REQUIRED = {"utterance_id", "key", "environment", "attack_id",
            "score_bonafide", "threshold"}

def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (n - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted

def load_predictions(root: Path):
    frames = {}
    for exp, filename in EXPERIMENTS.items():
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        df = pd.read_csv(path)
        missing = REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        if df["utterance_id"].duplicated().any():
            raise ValueError(f"Duplicate utterance_id in {path}")
        df = df[list(REQUIRED)].copy()
        df["is_spoof"] = df["key"].astype(str).str.lower().eq("spoof")
        df["false_accept"] = (
            df["is_spoof"] &
            (pd.to_numeric(df["score_bonafide"]) >=
             pd.to_numeric(df["threshold"]))
        )
        frames[exp] = df
    # Verify the utterance sets and metadata are aligned.
    reference = frames[BASELINE].set_index("utterance_id")
    for exp, df in frames.items():
        current = df.set_index("utterance_id")
        if set(current.index) != set(reference.index):
            raise ValueError(f"Utterance IDs differ for {exp}")
        for col in ["key", "environment", "attack_id"]:
            if not current[col].sort_index().equals(reference[col].sort_index()):
                raise ValueError(f"{col} differs between {BASELINE} and {exp}")
    return frames

def paired_bootstrap_diff(a, b, n_boot, rng):
    """CI for FAR(a)-FAR(b), resampling paired trial rows."""
    n = len(a)
    if n == 0:
        return np.nan, np.nan, np.nan
    diff = float(np.mean(a) - np.mean(b))
    # The paired per-trial difference is in {-1, 0, +1}. A bootstrap
    # replicate is therefore obtained efficiently from a multinomial draw
    # over the three observed difference categories.
    d = a.astype(np.int8) - b.astype(np.int8)
    p_minus = np.mean(d == -1)
    p_zero = np.mean(d == 0)
    p_plus = np.mean(d == 1)
    counts = rng.multinomial(n, [p_minus, p_zero, p_plus], size=n_boot)
    reps = (counts[:, 2] - counts[:, 0]) / n
    lo, hi = np.quantile(reps, [0.025, 0.975])
    return diff, float(lo), float(hi)

def compare_group(frames, group_cols, n_boot, seed):
    rows = []
    rng = np.random.default_rng(seed)
    # Create common metadata and only spoof trials for FAR comparison.
    ref_cols = list(dict.fromkeys(["utterance_id", "key", "environment", "attack_id"] + group_cols))
    ref = frames[BASELINE][ref_cols]
    spoof_ids = set(ref.loc[ref.key.astype(str).str.lower().eq("spoof"), "utterance_id"])
    for group_key, group_df in ref[ref.utterance_id.isin(spoof_ids)].groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        ids = group_df["utterance_id"].tolist()
        arrays = {}
        for exp, df in frames.items():
            indexed = df.set_index("utterance_id")
            arrays[exp] = indexed.loc[ids, "false_accept"].to_numpy(dtype=bool)
        for comp in COMPARATORS:
            a = arrays[BASELINE]
            b = arrays[comp]
            n = len(a)
            fa_base, fa_comp = int(a.sum()), int(b.sum())
            diff, lo, hi = paired_bootstrap_diff(a.astype(float), b.astype(float), n_boot, rng)
            # Exact McNemar: discordant paired outcomes only.
            base_only = int(np.sum(a & ~b))
            comp_only = int(np.sum(~a & b))
            discordant = base_only + comp_only
            p = (binomtest(base_only, discordant, 0.5, alternative="two-sided").pvalue
                 if discordant else 1.0)
            row = {
                "comparison": f"{BASELINE}_vs_{comp}",
                "baseline": BASELINE,
                "comparator": comp,
                "spoof_trials": n,
                "baseline_false_accepts": fa_base,
                "comparator_false_accepts": fa_comp,
                "baseline_far_percent": 100 * fa_base / n if n else np.nan,
                "comparator_far_percent": 100 * fa_comp / n if n else np.nan,
                "far_difference_pp_baseline_minus_comparator": 100 * diff,
                "paired_bootstrap_ci95_low_pp": 100 * lo,
                "paired_bootstrap_ci95_high_pp": 100 * hi,
                "baseline_only_errors": base_only,
                "comparator_only_errors": comp_only,
                "mcnemar_discordant_pairs": discordant,
                "mcnemar_p": p,
            }
            for col, value in zip(group_cols, group_key):
                row[col] = value
            rows.append(row)
    result = pd.DataFrame(rows)
    # Holm correction within each comparator across all groups in this family.
    result["mcnemar_p_holm"] = np.nan
    for comp, idx in result.groupby("comparator").groups.items():
        result.loc[idx, "mcnemar_p_holm"] = holm_adjust(result.loc[idx, "mcnemar_p"].to_numpy())
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path,
        default=Path("results/condition_analysis"),
        help="Directory containing *_joined_eval_predictions.csv")
    parser.add_argument("--output-dir", type=Path,
        default=Path("results/paired_condition_statistics"))
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = load_predictions(args.input_dir)
    analyses = [
        ("overall", []),
        ("by_attack", ["attack_id"]),
        ("by_environment", ["environment"]),
        ("by_environment_attack", ["environment", "attack_id"]),
    ]
    for offset, (name, cols) in enumerate(analyses):
        if cols:
            out = compare_group(frames, cols, args.bootstrap, args.seed + offset)
        else:
            # Add one constant group for overall.
            overall_frames = {}
            for exp, df in frames.items():
                spoof = df[df["is_spoof"]].copy()
                spoof["all_trials"] = "ALL"
                overall_frames[exp] = spoof
            out = compare_group(overall_frames, ["all_trials"], args.bootstrap,
                                args.seed + offset)
            out = out.drop(columns=["all_trials"])
        out.to_csv(args.output_dir / f"paired_stats_{name}.csv", index=False)
        print(f"Saved {name}: {len(out)} rows")
    metadata = {
        "baseline": BASELINE,
        "comparators": COMPARATORS,
        "threshold": "Development-selected threshold already stored per utterance",
        "decision": "false_accept = spoof and score_bonafide >= threshold",
        "difference_sign": "positive means E7 has higher FAR (worse) than comparator",
        "bootstrap": {"replicates": args.bootstrap, "seed": args.seed,
                      "resampling": "paired utterance-level bootstrap within each group"},
        "test": "two-sided exact McNemar test; Holm correction within comparator and output family",
        "note": "Confidence intervals and p-values quantify sampling uncertainty on this evaluation set; they do not establish generalisation to other datasets."
    }
    (args.output_dir / "paired_stats_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Done. Outputs: {args.output_dir.resolve()}")

if __name__ == "__main__":
    main()
