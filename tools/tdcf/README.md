# ASVspoof 2019 t-DCF component

This directory contains the ASVspoof 2019 Python t-DCF evaluation implementation
used by this project.

## Provenance and license

- Upstream project: https://github.com/nesl/asvspoof2019
- Upstream component: `tDCF_python_v1`
- License: CC BY-NC-SA 4.0
- Preserved license notice: [`LICENSE`](LICENSE)
- Upstream package note: [`README.txt`](README.txt)

This component is third-party code and is excluded from the repository-root MIT
License. See [`../../THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md).

## How this repository uses it

The project-specific wrappers import `eval_metrics.py` from this directory:

- `tools/compute_pa_tdcf.py`
- `tools/tune_e7.py`
- `tools/evaluate_tuned_e7.py`

Run those wrappers from the repository root or through the complete
reproducibility notebook. The legacy command shown in the upstream `README.txt`
refers to `evaluate_tDCF_asvspoof19.py`, which is not included in this adapted
repository layout; use the project wrappers above instead.

The evaluation-plan reference is:
https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf
