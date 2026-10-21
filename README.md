# SRE-PCC: scale-residual correction for monthly electricity carbon responsibility

This repository contains the reproducibility materials associated with the manuscript
**Scale-Residual Analysis and Physics-Consistent Correction of Monthly Electricity Carbon Responsibility under Fixed Regional Boundaries**.

The repository preserves the two experimentally frozen implementations as separate projects:

- `full_st_v22/`: SRE-PCC Full-ST method, prior ablations, the 696-hour February test, period-transfer and wind tests, and a lossy-AC accounting audit.
- `sota_v37/`: four **task-adapted** state-construction comparator families (SF*, PM*, PM-DSE and PI-GNN), 120 off-target labeled calibration scenarios, and 13 frozen comparator runs.
- `audit/`: cross-project integration audit and the frozen February comparison table.
- `tools/`: verification and reproduction entry points.

## Setup and verification

The project can be used in a fresh Python 3.11–3.13 environment. The requirements file specifies dependency minimum versions; it is not an archival record of the exact historical solver environment.

```bash
python -m pip install -r requirements_unified.txt
python tools/run.py verify-tests
python tools/run.py full-st-verify
```

The first command runs independent test suites, SHA256 checks on frozen SOTA artifacts and cross-project consistency checks. It **does not retrain** the comparator models. Full-ST experiments may be rerun with `python tools/run.py full-st-reproduce`; this can take longer. SOTA frozen candidate evaluation can be replayed with `python tools/run.py sota-phase-b`. See each subproject README before undertaking training runs.

## Reported reference values

The February 2020 experiment covers 696 hours, with MAE in kgCO₂/MWh:

| Method | MAE | Training information |
|:--|--:|:--|
| Direct-M | 140.160 | Monthly aggregate observations |
| SRE-PCC Full-ST | 1.117 | Independent historical prior; no labeled hidden-state training |
| SF* adapted | 0.769 | 120 labeled off-target calibration scenarios |
| PM* adapted | 0.649 | 120 labeled off-target calibration scenarios |
| PM-DSE adapted | 1.170 | 120 labeled off-target calibration scenarios |
| PI-GNN adapted | 0.990 ± 0.186 | 120 labeled off-target calibration scenarios; 10 random seeds |

The compared methods differ in calibration resources and projection details; the figures should be interpreted with the protocol descriptions in `audit/INTEGRATION_AUDIT.md`. The comparator results are from frozen artifacts, not from retraining performed during repository assembly.

## Research data and attribution

Official RTS-GMLC release 0.2 data are distributed in the permitted study subset under `full_st_v22/data/raw/`. The dataset's required DOE/NREL/ALLIANCE use notice remains in `full_st_v22/data/raw/RTS_GMLC_DATA_USE_DISCLAIMER.md` and must be retained in any redistributed copy of the data.

Upstream dataset: https://github.com/GridMod/RTS-GMLC/tree/0.2

Citation: C. Barrows et al., *The IEEE Reliability Test System: A Proposed 2019 Update*, IEEE Transactions on Power Systems 35(1), 119–127 (2020). DOI: 10.1109/TPWRS.2019.2925557.

## Release scope

The frozen algorithm sources, data, benchmark outputs and test artifacts are preserved. Draft manuscript files and internal Word QA reports were excluded intentionally from this public-facing package. `MANIFEST_PUBLIC_SHA256.txt` records the content hashes of files in this assembled release. Historical source-archive hashes under `audit/` document the earlier archived bundles.

**Source-code license:** no explicit license has yet been granted for the original research code. A code license should be selected by the authors/institution before third-party reuse is authorized. The upstream RTS-GMLC data rights are governed separately by their retained notice.
