#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pandas",
#   "numpy",
#   "pydantic",
# ]
# ///
"""
Battery Archive data loader.

Reads zip files from battery_archive_data/ and constructs the
Dataset → Battery → Cycle hierarchy defined in schema.py.

Usage
-----
    from loader import load_all, load_dataset

    datasets = load_all()                  # all zips → list[Dataset]
    snl_lfp = load_dataset("SNL LFP.zip") # single zip → Dataset
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from schema import (
    Battery,
    Chemistry,
    Cycle,
    Dataset,
    FailureMechanism,
    FormFactor,
)

DATA_DIR = Path(__file__).parent / "battery_archive_data"

# ── Filename parsing ──────────────────────────────────────────────────────────

# Maps filename token → Chemistry enum
_CHEMISTRY_TOKENS: list[tuple[str, Chemistry]] = [
    ("NMC_LCO", Chemistry.NMC_LCO),
    ("LFP",     Chemistry.LFP),
    ("NCA",     Chemistry.NCA),
    ("NMC",     Chemistry.NMC),
    ("LCO",     Chemistry.LCO),
]

# Maps filename token → FormFactor enum
_FORM_FACTOR_TOKENS: list[tuple[str, FormFactor]] = [
    ("18650", FormFactor.CYLINDRICAL_18650),
    ("pouch", FormFactor.POUCH),
    ("prism", FormFactor.PRISMATIC),
]

# Maps filename fragment → FailureMechanism enum (UL-Purdue only)
_FAILURE_TOKENS: list[tuple[str, FailureMechanism]] = [
    ("-EX",  FailureMechanism.EXTERNAL_SHORT),
    ("-OV",  FailureMechanism.OVERVOLTAGE),
    ("-NA",  FailureMechanism.NORMAL_AGING),
    ("DPA",  FailureMechanism.DEGRADATION_PROTOCOL_A),
]

# Zip filename → dataset name
_ZIP_TO_DATASET: dict[str, str] = {
    "CALCE.zip":              "CALCE",
    "HNEI.zip":               "HNEI",
    "Michigan Expansion.zip": "Michigan_Expansion",
    "Michigan Formation.zip": "Michigan_Formation",
    "Oxford.zip":             "Oxford",
    "SNL LFP.zip":            "SNL_LFP",
    "SNL NCA.zip":            "SNL_NCA",
    "SNL NMC.zip":            "SNL_NMC",
    "UL-Purdue.zip":          "UL-Purdue",
}


def _parse_cell_metadata(entry_name: str) -> dict:
    """
    Extract all metadata encoded in a Battery Archive filename.

    Example filename:
        SNL_18650_LFP_25C_0-100_0.5-1C_a_cycle_data.csv
        MICH_BLForm1_pouch_NMC_45C_0-100_1-1C_a_cycle_data.csv
        UL-PUR_N10-EX9_18650_NCA_23C_0-100_0.5-0.5C_i_cycle_data.csv
    """
    stem = Path(entry_name).stem  # drop .csv

    chemistry: Optional[Chemistry] = None
    for token, value in _CHEMISTRY_TOKENS:
        if token in stem:
            chemistry = value
            break

    form_factor: Optional[FormFactor] = None
    for token, value in _FORM_FACTOR_TOKENS:
        if token.lower() in stem.lower():
            form_factor = value
            break

    temp_match = re.search(r'_(-?\d+)C_', stem)
    temperature_c: Optional[float] = float(temp_match.group(1)) if temp_match else None

    crate_match = re.search(r'_([\d.]+)-([\d.]+)C_', stem)
    charge_rate_c: Optional[float] = float(crate_match.group(1)) if crate_match else None
    discharge_rate_c: Optional[float] = float(crate_match.group(2)) if crate_match else None

    failure_mechanism: Optional[FailureMechanism] = None
    for token, value in _FAILURE_TOKENS:
        if token in stem:
            failure_mechanism = value
            break

    has_formation = "Formation" in entry_name or "BLForm" in stem

    return {
        "chemistry": chemistry,
        "form_factor": form_factor,
        "temperature_c": temperature_c,
        "charge_rate_c": charge_rate_c,
        "discharge_rate_c": discharge_rate_c,
        "failure_mechanism": failure_mechanism,
        "has_formation_cycles": has_formation,
    }


# ── Cycle parsing ─────────────────────────────────────────────────────────────

_CYCLE_COL_MAP = {
    "cycle_index":          ["Cycle_Index", "Cycle Index"],
    "charge_capacity_ah":   ["Charge_Capacity (Ah)", "Charge_Capacity(Ah)"],
    "discharge_capacity_ah":["Discharge_Capacity (Ah)", "Discharge_Capacity(Ah)"],
    "charge_energy_wh":     ["Charge_Energy (Wh)", "Charge_Energy(Wh)"],
    "discharge_energy_wh":  ["Discharge_Energy (Wh)", "Discharge_Energy(Wh)"],
    "min_voltage_v":        ["Min_Voltage (V)", "Min_Voltage(V)"],
    "max_voltage_v":        ["Max_Voltage (V)", "Max_Voltage(V)"],
    "min_current_a":        ["Min_Current (A)", "Min_Current(A)"],
    "max_current_a":        ["Max_Current (A)", "Max_Current(A)"],
    "test_time_s":          ["Test_Time (s)", "Test_Time(s)"],
}


def _resolve(df: pd.DataFrame, canonical: str) -> Optional[str]:
    for alias in _CYCLE_COL_MAP.get(canonical, []):
        if alias in df.columns:
            return alias
    return None


def _parse_cycles(df: pd.DataFrame, has_formation: bool) -> list[Cycle]:
    """Convert a cycle_data DataFrame into a list of Cycle objects."""
    col = {k: _resolve(df, k) for k in _CYCLE_COL_MAP}

    # Drop rows where cycle_index is missing
    ci_col = col["cycle_index"]
    if ci_col is None:
        raise ValueError("No Cycle_Index column found in cycle data.")
    df = df.dropna(subset=[ci_col]).copy()
    df[ci_col] = df[ci_col].astype(int)

    def get(row, key: str, default: float = 0.0) -> float:
        c = col[key]
        if c is None:
            return default
        v = row[c]
        return float(v) if pd.notna(v) else default

    cycles: list[Cycle] = []
    for _, row in df.iterrows():
        idx = int(row[ci_col])
        # Formation cycles are the first few cycles in datasets that include them.
        # The Michigan Formation dataset flags them via the filename; within the
        # data itself we mark all cycles as formation since the entire experiment
        # is a formation study.  For Michigan Expansion, only the first cycle
        # (the slow formation cycle at C/5) is flagged.
        is_formation = has_formation

        cycles.append(Cycle(
            cycle_index=idx,
            charge_capacity_ah=get(row, "charge_capacity_ah"),
            discharge_capacity_ah=get(row, "discharge_capacity_ah"),
            charge_energy_wh=get(row, "charge_energy_wh"),
            discharge_energy_wh=get(row, "discharge_energy_wh"),
            min_voltage_v=get(row, "min_voltage_v"),
            max_voltage_v=get(row, "max_voltage_v"),
            min_current_a=get(row, "min_current_a"),
            max_current_a=get(row, "max_current_a"),
            test_time_s=get(row, "test_time_s"),
            is_formation=is_formation,
        ))

    return cycles


# ── Timeseries entry matching ─────────────────────────────────────────────────

def _find_timeseries_entry(all_entries: list[str], cycle_entry: str) -> Optional[str]:
    """
    Given the path of a *_cycle_data.csv entry, find the matching timeseries
    entry in the same zip.  Battery Archive uses two naming conventions:
        *_timeseries.csv
        *_timeseries_data.csv
    """
    base = re.sub(r'_cycle_data\.csv$', '', cycle_entry)
    for suffix in ("_timeseries.csv", "_timeseries_data.csv"):
        candidate = base + suffix
        if candidate in all_entries:
            return candidate
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def load_dataset(zip_filename: str, data_dir: Path = DATA_DIR) -> Dataset:
    """
    Load a single Battery Archive zip into a Dataset object.

    Parameters
    ----------
    zip_filename : str
        Filename of the zip, e.g. "SNL LFP.zip".
    data_dir : Path
        Directory containing the zip files.

    Returns
    -------
    Dataset with all batteries populated and timeseries sources registered.
    """
    zip_path = data_dir / zip_filename
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")

    dataset_name = _ZIP_TO_DATASET.get(zip_filename, zip_filename.replace(".zip", ""))

    batteries: list[Battery] = []

    with zipfile.ZipFile(zip_path) as zf:
        all_entries = zf.namelist()
        cycle_entries = [e for e in all_entries if e.endswith("cycle_data.csv")]

        for entry in sorted(cycle_entries):
            meta = _parse_cell_metadata(entry)

            # Skip if we couldn't parse minimum required fields
            if meta["chemistry"] is None or meta["form_factor"] is None:
                print(f"  WARN: could not parse metadata from {entry!r}, skipping.")
                continue

            cell_id = Path(entry).stem.replace("_cycle_data", "")

            with zf.open(entry) as f:
                try:
                    df = pd.read_csv(f)
                except Exception as exc:
                    print(f"  WARN: could not read {entry}: {exc}, skipping.")
                    continue

            try:
                cycles = _parse_cycles(df, has_formation=meta["has_formation_cycles"])
            except Exception as exc:
                print(f"  WARN: could not parse cycles from {entry}: {exc}, skipping.")
                continue

            # Estimate nominal capacity from first 3 non-zero discharge cycles
            discharges = [
                c.discharge_capacity_ah for c in cycles[:10]
                if c.discharge_capacity_ah > 0
            ]
            nominal_capacity_ah = float(np.median(discharges)) if discharges else None

            battery = Battery(
                cell_id=cell_id,
                dataset_name=dataset_name,
                chemistry=meta["chemistry"],
                form_factor=meta["form_factor"],
                temperature_c=meta["temperature_c"] or 25.0,
                charge_rate_c=meta["charge_rate_c"] or 1.0,
                discharge_rate_c=meta["discharge_rate_c"] or 1.0,
                nominal_capacity_ah=nominal_capacity_ah,
                has_formation_cycles=meta["has_formation_cycles"],
                failure_mechanism=meta["failure_mechanism"],
                cycles=cycles,
            )

            # Register timeseries source for lazy loading
            ts_entry = _find_timeseries_entry(all_entries, entry)
            if ts_entry:
                battery.set_timeseries_source(zip_path, ts_entry)

            batteries.append(battery)

    print(f"  {dataset_name}: loaded {len(batteries)} batteries")
    return Dataset(name=dataset_name, source_zip=zip_filename, batteries=batteries)


def load_all(data_dir: Path = DATA_DIR) -> list[Dataset]:
    """
    Load all recognised Battery Archive zips from data_dir.

    Returns a list of Dataset objects in a consistent order.
    """
    datasets: list[Dataset] = []
    for zip_filename in _ZIP_TO_DATASET:
        zip_path = data_dir / zip_filename
        if not zip_path.exists():
            print(f"  SKIP: {zip_filename} not found in {data_dir}")
            continue
        datasets.append(load_dataset(zip_filename, data_dir=data_dir))
    return datasets
