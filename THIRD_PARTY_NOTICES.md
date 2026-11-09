# Third-party notices and data provenance

## ASVspoof 2019 PA evaluation metric

The minimum t-DCF calculation in this project uses the ASVspoof 2019 Python t-DCF v1 evaluation implementation in `tools/tdcf/eval_metrics.py`. The metric and its evaluation settings are defined by the ASVspoof 2019 challenge evaluation plan. Please consult the upstream implementation and evaluation plan, and preserve any upstream notices and licence terms when redistributing this file.

- ASVspoof 2019 challenge evaluation plan: https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf
- Public ASVspoof 2019 implementation repository containing t-DCF Python resources: https://github.com/nesl/asvspoof2019/tree/master/tDCF_python_v1
- t-DCF paper: T. Kinnunen et al., “t-DCF: a Detection Cost Function for the Tandem Assessment of Spoofing Countermeasures and Automatic Speaker Verification,” *Proc. Odyssey 2018*, pp. 312–319. https://www.isca-speech.org/archive/Odyssey_2018/pdfs/68.pdf

## Dataset

The ASVspoof 2019 PA audio, protocols, keys, and ASV score files are not redistributed in this repository. Obtain data from the authorized sources and follow the applicable dataset licence/access terms.

- ASVspoof 2019 challenge page: https://www.asvspoof.org/index2019.html
- Edinburgh DataShare record: https://doi.org/10.7488/ds/2555

## Python dependencies

Third-party Python packages are installed from their package distributions and retain their own licences. See `pyproject.toml` for the dependency list.
