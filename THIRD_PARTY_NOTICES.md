# Third-Party Notices and Licensing

This repository contains original project code as well as separately licensed
third-party components and adaptations. Read this file together with the root
`LICENSE` and the component-level license/notice files before reusing code.

## Project-authored code

Original code authored specifically for this project is licensed under the MIT
License, except for the files and components listed below. See the root
`LICENSE` for the exact scope and terms.

## CQCC / constant-Q transform implementation

### EURECOM CQCC package

- **Upstream package:** ASVspoof 2019 baseline, `baseline/CQCC_v1.0/`
- **Upstream source:** https://github.com/nesl/asvspoof2019/tree/master/baseline/CQCC_v1.0
- **Original notice:** The package README identifies Copyright (C) 2016 EURECOM,
  France, and states that the package is licensed under CC BY-NC-SA 4.0.
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0
  International (CC BY-NC-SA 4.0).
- **License reference:** `licenses/CQCC_EURECOM_LICENSE.txt`

### CQT Toolbox 2013 routines

The project’s `src/features/cqt.py` documents a Python port/adaptation of the
following MATLAB routines from the upstream CQCC package:

- `CQT_toolbox_2013/nsgcqwin.m`
- `CQT_toolbox_2013/nsgtf_real.m`
- `CQT_toolbox_2013/winfuns.m`

Relevant upstream sources:

- https://github.com/nesl/asvspoof2019/blob/master/baseline/CQCC_v1.0/CQT_toolbox_2013/nsgcqwin.m
- https://github.com/nesl/asvspoof2019/blob/master/baseline/CQCC_v1.0/CQT_toolbox_2013/nsgtf_real.m
- https://github.com/nesl/asvspoof2019/blob/master/baseline/CQCC_v1.0/CQT_toolbox_2013/winfuns.m

The original source headers identify Copyright (C) 2013 Nicki Holighaus and,
for relevant routines, Christian Schörkhuber. The upstream CQCC package README
states that the CQT Toolbox is licensed under CC BY-NC-SA 3.0 Unported.

- **Original license:** Creative Commons Attribution-NonCommercial-ShareAlike
  3.0 Unported (CC BY-NC-SA 3.0).
- **License reference:** `licenses/CQT_TOOLBOX_LICENSE.txt`

### Scope and adaptation notice

`src/features/cqt.py` is a Python port/adaptation, not the original MATLAB
source. It is excluded from the root MIT License. The adapted material is
identified as distributed under CC BY-NC-SA 4.0. CC BY-NC-SA 3.0 Section
4(b)(ii) permits an adaptation to be distributed under a later version with the
same license elements; CC BY-NC-SA 4.0 retains Attribution, NonCommercial, and
ShareAlike. The original notices and source links above are retained for both
upstream components. The Python implementation contains modifications and
should not be represented as the untouched upstream MATLAB code.

For the complete license terms, consult the canonical legal codes linked in the
component license-reference files. This statement is a provenance notice, not
legal advice; the project maintainers should obtain legal review if the licensing
boundary or commercial-use implications are material.

## ASVspoof 2019 t-DCF evaluation implementation

- **Component:** ASVspoof 2019 Python t-DCF implementation, version 1.0
- **Location:** `tools/tdcf/`
- **Upstream source:** https://github.com/nesl/asvspoof2019/tree/master/tDCF_python_v1
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0
  International (CC BY-NC-SA 4.0).
- **License text/notice:** `tools/tdcf/LICENSE`
- **Upstream package note:** `tools/tdcf/README.txt`

This third-party component is excluded from the root MIT License. The original
license notice is retained. The project wrapper scripts import `eval_metrics.py`
from this directory; see `tools/tdcf/README.md` for the way this repository uses
it.

## Dataset and official score files

The ASVspoof 2019 PA audio corpus, protocol files, and official ASV score files
are not included in this repository. Obtain them from their authorized
sources and follow their own access, attribution, and license terms. This
repository does not grant rights to redistribute those materials.

## Dependencies

Python packages listed in `pyproject.toml` and `requirements.txt` are installed
from their respective distributions. They are not copied into this repository
and retain their own licenses and notices.
