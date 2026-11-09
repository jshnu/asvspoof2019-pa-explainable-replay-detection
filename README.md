# ASVspoof 2019 PA — Explainable Replay-Attack Detection

This repository contains the Python implementation and reproducibility workflow for replay-attack countermeasure experiments on the ASVspoof 2019 Physical Access (PA) corpus.

The implementation extracts three utterance-level feature groups—CQCC, TECC/TKEO, and RLFCC/acoustic—then evaluates seven feature-group ablations (E1–E7) with XGBoost. A separate E7_FINAL workflow tunes the fused model using the official training and development splits, evaluates it once on the official evaluation split, and produces SHAP explanations and post-hoc condition analyses.

The notebooks orchestrate the existing source modules and scripts in this repository. They do not reimplement the feature extractors, model trainer, t-DCF calculation, tuning procedure, or SHAP algorithms.

## Repository contents

Keep the complete source tree when publishing this repository, especially:

- `src/data/` — protocol parsing, manifests, audio access, and validation.
- `src/features/` — CQCC, TKEO/TECC, and acoustic feature implementations.
- `src/pipeline/` — extraction, cache, aggregation, and fusion pipelines.
- `src/models/` — XGBoost model and explanation support.
- `tools/` — tuning, evaluation, official PA min-tDCF, SHAP, condition analysis, paired statistics, and paper-figure scripts.
- `tests/` — project unit tests.
- `notebooks/` — the complete reproducibility workflow and the reduced pipeline smoke test.
- `configs/` — shared, machine-independent configuration templates.

Do not publish a partial source archive that omits `src/data/` or `src/features/`. The notebooks check for these modules and stop when required source files are missing.

## Requirements

- Python `>=3.10,<3.13` (see `pyproject.toml`).
- A supported operating system and enough free disk space for temporary frame-level feature caches and generated outputs.
- The ASVspoof 2019 PA audio/protocol files and the official PA ASV score files. These data files are not included in this repository.
- For GPU acceleration, a CUDA-capable NVIDIA GPU, a compatible driver, and an XGBoost installation whose build has CUDA support. A CPU-only run is supported by selecting `XGB_DEVICE = "cpu"` in the notebook.

## Installation

From the repository root, create and activate a virtual environment.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,explain,notebook]"
```

### Linux / macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,explain,notebook]'
```

The `dev` extra installs pytest, the `explain` extra installs SHAP and Matplotlib, and the `notebook` extra installs JupyterLab and ipykernel. The project dependencies are declared in `pyproject.toml`; `requirements.txt` contains the core runtime dependencies.

To use the notebooks, open the repository in JupyterLab, Jupyter Notebook, or VS Code with the virtual environment selected as the notebook kernel. Install `jupyter`/`ipykernel` in that environment if your editor requires them.

### XGBoost device selection

The notebook defaults to `XGB_DEVICE = "cuda"` to use a CUDA-enabled XGBoost build. The notebook checks the installed XGBoost build before training. If CUDA is unavailable or your XGBoost build is CPU-only, set `XGB_DEVICE = "cpu"` in the first configuration cell. GPU and CPU runs may differ slightly in performance and numerical results.

## Dataset setup

Obtain the ASVspoof 2019 PA corpus and the official ASV score files from their authorized distribution. Do not commit the corpus or score files to this repository.

The notebook expects `DATASET_ROOT` to be the **parent directory containing `PA/`**. The default layout is:

```text
<DATASET_ROOT>/
└── PA/
    ├── ASVspoof2019_PA_train/
    │   └── flac/
    ├── ASVspoof2019_PA_dev/
    │   └── flac/
    ├── ASVspoof2019_PA_eval/
    │   └── flac/
    ├── ASVspoof2019_PA_cm_protocols/
    │   ├── ASVspoof2019.PA.cm.train.trn.txt
    │   ├── ASVspoof2019.PA.cm.dev.trl.txt
    │   └── ASVspoof2019.PA.cm.eval.trl.txt
    └── ASVspoof2019_PA_asv_scores/
        ├── ASVspoof2019.PA.asv.dev.gi.trl.scores.txt
        └── ASVspoof2019.PA.asv.eval.gi.trl.scores.txt
```

Set the dataset root before starting Jupyter, or edit `DATASET_ROOT` in the first configuration cell of the notebook.

### Windows PowerShell

```powershell
$env:ASVSPOOF_DATASET_ROOT = "D:\datasets\ASVspoof2019_archive"
```

### Linux / macOS

```bash
export ASVSPOOF_DATASET_ROOT="/data/ASVspoof2019_archive"
```

If the ASV score directory is stored separately from `PA/ASVspoof2019_PA_asv_scores/`, set `ASVSPOOF_ASV_SCORE_DIR` to the directory containing the two `.scores.txt` files. If the notebook is launched from outside the repository tree, optionally set `ASVSPOOF_PROJECT_ROOT` to the repository root.

The shared `configs/data.yaml` contains a placeholder data root. The full notebook creates a local `configs/data.notebook.local.yaml` with the configured path; `.gitignore` excludes this machine-specific file.

## Recommended run order

### 1. Pipeline smoke test

Open and run:

```text
notebooks/ASVspoof_2019_PA_Pipeline_Smoke_Test.ipynb
```

This test selects a balanced subset of the official manifests and invokes the real project implementations for:

- manifest generation;
- CQCC, TKEO/TECC, and acoustic extraction;
- frame-cache aggregation and feature fusion;
- E1–E7 XGBoost training;
- the PA min-tDCF scripts;
- a reduced E7_FINAL tuning run and final evaluation;
- global, class-conditional, and individual SHAP;
- condition analysis and paired statistics.

The smoke test writes to a unique `results/_pipeline_smoke_*` directory and uses a temporary configuration. It is an integration check, not a benchmark run. Its subset metrics must not be reported as ASVspoof results. It does not reproduce the full 80-trial tuning search or full-corpus workload.

### 2. Complete reproducibility workflow

After the smoke test passes, open and run:

```text
notebooks/ASVspoof_2019_PA_Complete_Reproducibility.ipynb
```

The complete notebook runs the project unit tests, creates full protocol manifests, extracts and aggregates all feature groups, fuses the 1,350-dimensional E7 representation, trains E1–E7, computes official PA min-tDCF, runs the 80-trial E7_FINAL tuning workflow, evaluates the selected model on the official evaluation split, generates SHAP explanations, performs condition and paired-statistics analyses, and creates the publication figures and tables.

The default production settings are defined in the first configuration cell. In particular, the notebook uses 2,000 estimators for E1–E7, an 80-trial E7_FINAL search with up to 10,000 estimators and 200-round early stopping, and the configured SHAP sample sizes. These are computationally intensive stages. Adjust worker counts and device settings for the available hardware without changing the research settings unless the experiment is intentionally being modified.

The full notebook removes a feature group's temporary frame cache only after its aggregate matrix has been written and checked. Set `DELETE_FRAME_CACHES_AFTER_AGGREGATION = False` only when you intentionally want to retain those caches and have sufficient free storage.

## Outputs and reproducibility

The workflow writes generated artifacts under `data/processed/` and `results/`, including manifests, feature matrices, models, metrics, predictions, explanation outputs, figures, and tables. These generated directories are excluded by `.gitignore` by default; do not commit the full raw corpus, large frame caches, or machine-specific local configuration files.

The complete notebook writes `results/reproducibility/run_metadata.json` with software versions, selected execution settings, manifest counts, and feature dimensions. Preserve this metadata together with the tuning search manifest and the result tables when archiving a run.

GPU training can vary slightly across hardware, drivers, and library builds; exact bitwise-identical results across different machines are not guaranteed. The smoke-test metrics are only for checking that the code paths execute.

## Troubleshooting

- **Project root not found:** launch Jupyter from the repository or set `ASVSPOOF_PROJECT_ROOT`.
- **Dataset files missing:** verify that `ASVSPOOF_DATASET_ROOT` points to the parent of `PA/`, and that the protocol/audio folder names match `configs/data.yaml`.
- **ASV score files missing:** verify `ASVSPOOF_ASV_SCORE_DIR` or put the two official score files in `PA/ASVspoof2019_PA_asv_scores/`.
- **CUDA unavailable:** set `XGB_DEVICE = "cpu"`, or install an XGBoost build compatible with the local CUDA setup.
- **SHAP or Matplotlib missing:** install with `python -m pip install -e ".[dev,explain,notebook]"`.
- **Insufficient storage:** retain the default cache cleanup, ensure the project output location has sufficient free space, and avoid running multiple extraction jobs concurrently.

For a reproducibility issue, include the failing notebook cell, the complete error/traceback, `run_metadata.json` when available, and the relevant package versions. Do not include private local paths or corpus audio in issue reports.


## Citing this repository

Before the first public release, copy `CITATION.cff.template` to `CITATION.cff` and replace the author and repository URL placeholders with the correct information. GitHub uses the root-level `CITATION.cff` to display a **Cite this repository** prompt.

For a persistent citation in a paper, publish a versioned GitHub Release and archive that release with Zenodo. Cite the resulting version-specific DOI rather than only citing the moving `main` branch. See `PAPER_REPOSITORY_CITATION.md` for a citation template.

## Third-party code and licensing

The ASVspoof 2019 PA corpus and official ASV score files are not included. Obtain them from the authorized ASVspoof/Edinburgh DataShare distribution and follow the dataset's applicable access and licence terms. See `THIRD_PARTY_NOTICES.md` for the evaluation-metric attribution.

This source archive does not add a project-wide `LICENSE` file. Before describing the repository as open source or granting reuse rights, review the provenance and licence terms of all included code, particularly `tools/tdcf/eval_metrics.py`, then add a licence that you are authorized to apply. A public GitHub repository without a licence remains subject to default copyright rules.
