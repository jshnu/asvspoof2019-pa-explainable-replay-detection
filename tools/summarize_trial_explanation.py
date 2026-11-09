#!/usr/bin/env python3
"""Create a concise, human-readable summary from explain_one_trial.py outputs.

Run after tools/explain_one_trial.py:
  python tools/summarize_trial_explanation.py --utterance-id PA_E_...

The wording describes the extracted descriptor, not a causal explanation.
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import pandas as pd


GROUP_INFO = {
    "cqcc": {
        "name": "CQCC spectral features",
        "meaning": "constant-Q cepstral representation of the utterance spectrum",
        "dims": 90,
        "family_size": 30,
        "families": ("static", "delta", "delta-delta"),
    },
    "tkeo": {
        "name": "TECC temporal-energy features",
        "meaning": "cepstral representation of Teager-energy patterns across Gabor subbands",
        "dims": 120,
        "family_size": 40,
        "families": ("static", "delta", "delta-delta"),
    },
    "acoustic": {
        "name": "RLFCC residual-acoustic features",
        "meaning": "cepstral representation of the linear-prediction residual spectrum",
        "dims": 60,
        "family_size": 20,
        "families": ("static", "delta", "delta-delta"),
    },
}

STAT_WORDING = {
    "mean": "average across frames",
    "std": "frame-to-frame variability",
    "p10": "10th-percentile frame value",
    "p50": "median frame value",
    "p90": "90th-percentile frame value",
}


def describe_feature(name: str) -> str:
    m = re.fullmatch(r"(cqcc|tkeo|acoustic)__(mean|std|p10|p50|p90)__c(\d+)", name)
    if not m:
        return name.replace("__", " / ")
    group, stat, raw_idx = m.groups()
    info = GROUP_INFO[group]
    idx = int(raw_idx)
    if idx >= info["dims"]:
        return name
    family_idx, coefficient = divmod(idx, info["family_size"])
    family = info["families"][family_idx]
    return (
        f"{info['name']}: {family} coefficient {coefficient} "
        f"({STAT_WORDING[stat]})"
    )


def compact_direction(frame: pd.DataFrame, direction: str, n: int) -> list[dict]:
    if direction == "bonafide":
        chosen = frame[frame["shap_value_log_odds"] > 0].nlargest(
            n, "shap_value_log_odds"
        )
    else:
        chosen = frame[frame["shap_value_log_odds"] < 0].nsmallest(
            n, "shap_value_log_odds"
        )
    rows = []
    for _, r in chosen.iterrows():
        rows.append({
            "description": describe_feature(str(r["feature"])),
            "raw_feature": str(r["feature"]),
            "contribution": float(r["shap_value_log_odds"]),
            "value": float(r["feature_value"]),
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--utterance-id", required=True)
    p.add_argument("--input-dir", default="results/individual_explanations")
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    root = Path(args.input_dir)
    uid = args.utterance_id
    summary_path = root / f"{uid}_summary.json"
    feature_path = root / f"{uid}_feature_contributions.csv"
    group_path = root / f"{uid}_group_contributions.csv"
    for path in (summary_path, feature_path, group_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Run explain_one_trial.py first.")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    features = pd.read_csv(feature_path)
    groups = pd.read_csv(group_path)

    probability = float(summary["probability_bonafide"])
    threshold = float(summary["threshold_selected_on_dev"])
    predicted = str(summary["predicted_label"])
    actual = str(summary["actual_label"])
    outcome = str(summary["error_type"])
    score_pct = 100 * probability

    if predicted == "bonafide":
        decision_line = (
            f"The model classified this recording as bona fide "
            f"({score_pct:.1f}% model score; decision threshold {100*threshold:.1f}%)."
        )
    else:
        decision_line = (
            f"The model classified this recording as spoof "
            f"({100-score_pct:.1f}% spoof-side model score; "
            f"bona-fide threshold {100*threshold:.1f}%)."
        )

    toward_bona = compact_direction(features, "bonafide", args.top_k)
    toward_spoof = compact_direction(features, "spoof", args.top_k)

    group_rows = []
    for _, r in groups.iterrows():
        group = str(r["feature_group"])
        if group not in GROUP_INFO:
            continue
        value = float(r["sum_shap_log_odds"])
        group_rows.append({
            "group": GROUP_INFO[group]["name"],
            "direction": "towards bona fide" if value > 0 else (
                "towards spoof" if value < 0 else "neutral"
            ),
            "net_contribution": value,
        })
    group_rows.sort(key=lambda x: abs(x["net_contribution"]), reverse=True)

    def bullets(rows):
        if not rows:
            return ["No non-zero contributions in this direction."]
        return [
            f"{r['description']} — contribution {r['contribution']:+.3f}"
            for r in rows
        ]

    lines = [
        f"INDIVIDUAL REPLAY-SPOOF EXPLANATION — {uid}",
        "",
        decision_line,
        f"Actual label: {actual} | Outcome: {outcome.replace('_', ' ')}",
        "",
        "EVIDENCE PUSHING TOWARDS THE MODEL'S BONA-FIDE CLASS",
        *[f"- {x}" for x in bullets(toward_bona)],
        "",
        "EVIDENCE PUSHING TOWARDS THE MODEL'S SPOOF CLASS",
        *[f"- {x}" for x in bullets(toward_spoof)],
        "",
        "FEATURE-GROUP BALANCE (net SHAP contribution, log-odds)",
        *[
            f"- {r['group']}: {r['direction']} ({r['net_contribution']:+.3f})"
            for r in group_rows
        ],
        "",
        "Interpretation note: SHAP contributions show how this XGBoost model "
        "used the measured features for this one decision. They do not prove "
        "the physical cause of a replay artefact. The percentage is the model's "
        "bona-fide output score, not a calibrated probability of truth.",
        "Feature descriptions identify the implemented signal representation, "
        "coefficient family and across-frame statistic; they are not claims "
        "that a particular acoustic phenomenon was directly measured.",
    ]
    txt_path = root / f"{uid}_concise_explanation.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    def html_list(rows):
        if not rows:
            return "<li>None</li>"
        return "".join(
            f"<li>{html.escape(r['description'])} "
            f"<b>({r['contribution']:+.3f})</b></li>" for r in rows
        )

    group_html = "".join(
        f"<li>{html.escape(r['group'])}: {html.escape(r['direction'])} "
        f"({r['net_contribution']:+.3f})</li>" for r in group_rows
    )
    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Concise explanation: {html.escape(uid)}</title>
<style>
body{{font:16px system-ui,Arial;max-width:850px;margin:36px auto;padding:0 22px;color:#20242a;line-height:1.55}}
h1{{font-size:24px}} h2{{font-size:17px;margin-top:26px}}
.decision{{padding:14px 16px;background:#f2f4f7;border-radius:8px}}
small,.note{{color:#5c6570}}
li{{margin:8px 0}}
</style></head><body>
<h1>Why the model made this decision</h1>
<p><small>Utterance: {html.escape(uid)} · Actual: {html.escape(actual)} ·
Outcome: {html.escape(outcome.replace('_',' '))}</small></p>
<p class="decision">{html.escape(decision_line)}</p>
<h2>Evidence pushing towards bona fide</h2><ul>{html_list(toward_bona)}</ul>
<h2>Evidence pushing towards spoof</h2><ul>{html_list(toward_spoof)}</ul>
<h2>Overall feature-group balance</h2><ul>{group_html}</ul>
<p class="note">SHAP explains how this model used the measured features for this
recording; it does not establish the physical cause of a replay artefact.
The percentage is the model output score, not a calibrated probability of truth.
Feature descriptions identify the implemented representation and statistic,
not a directly measured physical phenomenon.</p>
</body></html>"""
    html_path = root / f"{uid}_concise_explanation.html"
    html_path.write_text(html_doc, encoding="utf-8")

    print(f"Saved:\n  {txt_path}\n  {html_path}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
