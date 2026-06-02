# Battery Cycle ML

A machine learning system for predicting battery degradation behavior, built on data from the [Battery Archive](https://www.batteryarchive.org/).

## Overview

This project builds a structured data pipeline and ML framework for battery cycle life prediction. The initial prototype covers nine datasets spanning multiple chemistries (LFP, NMC, NCA, LCO), form factors, temperatures, and cycling protocols.

**Current status:** EDA and data schema design.

## Datasets

Data is sourced from the Battery Archive and is **not included in this repository** (see [Data Setup](#data-setup) below). The following datasets are supported:

| Dataset | Chemistry | Cells | Median cycles |
|---------|-----------|-------|---------------|
| CALCE | LCO | 7 | 1,815 |
| HNEI | NMC/LCO | 15 | 1,103 |
| Michigan Expansion | NMC | 40 | ~500 |
| Michigan Formation | NMC | 18 | ~519 |
| Oxford | LCO | 8 | 74 |
| SNL LFP | LFP | — | — |
| SNL NCA | NCA | — | — |
| SNL NMC | NMC | 86 total | 1,307 |
| UL-Purdue | NCA | 22 | 322 |

## Project Structure

```
battery-cycle-ml/
├── battery_archive_data/   # ← not committed; see Data Setup
├── eda_output/             # ← not committed; regenerate with eda.py
├── eda.py                  # Exploratory data analysis script
├── schema.py               # Core data model (Dataset → Battery → Cycle)
└── README.md
```

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for reproducible Python environments. No manual `pip install` or virtualenv management needed.

### Install uv

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Data Setup

Download the Battery Archive datasets and place the zip files in `battery_archive_data/`:

```
battery_archive_data/
├── CALCE.zip
├── HNEI.zip
├── Michigan Expansion.zip
├── Michigan Formation.zip
├── Oxford.zip
├── SNL LFP.zip
├── SNL NCA.zip
├── SNL NMC.zip
└── UL-Purdue.zip
```

Data is available at [https://www.batteryarchive.org/](https://www.batteryarchive.org/).

## Usage

All scripts use [inline script dependencies](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies) — uv installs everything automatically on first run.

```powershell
# Run EDA and generate figures
uv run eda.py

# Outputs written to eda_output/
```

## License

Code: MIT. Data: subject to Battery Archive terms of use.
