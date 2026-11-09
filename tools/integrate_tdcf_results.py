"""Merge independently computed PA min-tDCF outputs into each XGBoost metrics.json."""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import pandas as pd

EXPERIMENT_DIRS = {
    'E1_CQCC': 'E1_CQCC',
    'E2_TKEO': 'E2_TKEO',
    'E3_ACOUSTIC': 'E3_ACOUSTIC',
    'E4_CQCC_TKEO': 'E4_CQCC_TKEO',
    'E5_CQCC_ACOUSTIC': 'E5_CQCC_ACOUSTIC',
    'E6_TKEO_ACOUSTIC': 'E6_TKEO_ACOUSTIC',
    'E7_ALL': 'E7_ALL',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--tdcf-csv', type=Path, default=None)
    ap.add_argument('--no-backup', action='store_true')
    args = ap.parse_args()
    root = args.project_root.resolve()
    csv_path = (args.tdcf_csv or root / 'results' / 'tdcf' / 'pa_min_tdcf_summary.csv').resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    tdf = pd.read_csv(csv_path)
    required = {'split','experiment','min_tdcf','cm_threshold_at_min_tdcf','asv_eer_percent','asv_threshold','asv_pfa_nontarget','asv_pmiss_target','asv_pmiss_spoof'}
    missing = required - set(tdf.columns)
    if missing:
        raise ValueError(f'Missing columns in t-DCF CSV: {sorted(missing)}')
    if tdf.duplicated(['split','experiment']).any():
        raise ValueError('Duplicate split/experiment rows in t-DCF CSV')
    if set(tdf['split']) != {'dev','eval'}:
        raise ValueError('Expected exactly dev and eval splits')

    for exp, dirname in EXPERIMENT_DIRS.items():
        path = root / 'results' / 'xgb_ablation' / dirname / 'metrics.json'
        if not path.exists():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text(encoding='utf-8'))
        for split in ('dev','eval'):
            rows = tdf[(tdf.split == split) & (tdf.experiment == exp)]
            if len(rows) != 1:
                raise ValueError(f'Expected one row for {split}/{exp}; found {len(rows)}')
            r = rows.iloc[0]
            data.setdefault(split, {})['min_tdcf'] = {
                'value': float(r.min_tdcf),
                'cm_threshold': float(r.cm_threshold_at_min_tdcf),
                'asv_eer_percent': float(r.asv_eer_percent),
                'asv_eer_threshold': float(r.asv_threshold),
                'asv_pfa_nontarget': float(r.asv_pfa_nontarget),
                'asv_pmiss_target': float(r.asv_pmiss_target),
                'asv_pmiss_spoof': float(r.asv_pmiss_spoof),
                'normalization': str(r.normalization),
                'implementation': 'official ASVspoof 2019 tDCF_python_v1 eval_metrics.compute_tDCF',
            }
        if not args.no_backup:
            backup = path.with_suffix('.json.pre_tdcf.bak')
            if not backup.exists():
                shutil.copy2(path, backup)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(f'Updated {path.relative_to(root)}')

    # Create a compact comparison table combining the existing classifier metrics and t-DCF.
    rows = []
    for exp, dirname in EXPERIMENT_DIRS.items():
        d = json.loads((root / 'results' / 'xgb_ablation' / dirname / 'metrics.json').read_text(encoding='utf-8'))
        for split in ('dev','eval'):
            m = d[split]
            rows.append({
                'experiment': exp, 'split': split,
                'n_features': d.get('n_features'),
                'eer_percent': m.get('eer_percent'),
                'roc_auc': m.get('roc_auc'),
                'precision_bonafide': m.get('precision_bonafide'),
                'recall_bonafide': m.get('recall_bonafide'),
                'f1_bonafide': m.get('f1_bonafide'),
                'balanced_accuracy': m.get('balanced_accuracy'),
                'min_tdcf': m.get('min_tdcf', {}).get('value'),
            })
    out = root / 'results' / 'tdcf' / 'ablation_eer_tdcf_comparison.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'Created {out}')

if __name__ == '__main__':
    main()
