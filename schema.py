"""
Battery cycle data schema.

Hierarchy:
    Dataset
    └── Battery  (one physical cell)
        └── Cycle  (one charge/discharge pair)
            └── CycleTimeseries  (raw voltage/current trace — lazy-loaded)

Design notes
------------
- Pydantic v2 is used throughout for runtime type validation and clean
  serialisation to/from JSON/dict.
- Enums enforce controlled vocabularies for chemistry, form factor, and
  failure mechanism — prevents silent typos and allows exhaustive matching
  in downstream ML code.
- CycleTimeseries is intentionally NOT embedded in Cycle.  The raw
  timeseries files are up to 300 MB each; they are referenced by source
  path and only materialised on explicit access via Battery.load_timeseries().
- Cycle.features is a free-form dict reserved for derived scalar features
  (e.g. dQ/dV peak height, plateau variance).  It is separate from the
  directly-measured fields so feature extractors can populate it without
  altering the core schema.
- Dataset is a lightweight grouping container.  The Battery is the natural
  ML sample unit; most iteration and featurisation happens at that level.
"""

from __future__ import annotations

import zipfile
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


# ── Controlled vocabularies ───────────────────────────────────────────────────


class Chemistry(str, Enum):
    """Cathode chemistry of the cell."""
    LCO = "LCO"            # Lithium Cobalt Oxide — CALCE, Oxford
    NMC = "NMC"            # Nickel Manganese Cobalt — Michigan, SNL
    NMC_LCO = "NMC_LCO"   # Blended NMC + LCO — HNEI
    NCA = "NCA"            # Nickel Cobalt Aluminium — SNL, UL-Purdue
    LFP = "LFP"            # Lithium Iron Phosphate — SNL


class FormFactor(str, Enum):
    """Physical cell form factor."""
    CYLINDRICAL_18650 = "18650"
    POUCH = "pouch"
    PRISMATIC = "prism"


class FailureMechanism(str, Enum):
    """
    Known failure / degradation mode, where annotated.
    Only present in the UL-Purdue dataset.
    """
    NORMAL_AGING = "Normal Aging"
    EXTERNAL_SHORT = "External Short"
    OVERVOLTAGE = "Overvoltage"
    DEGRADATION_PROTOCOL_A = "Degradation Protocol A"


# ── Timeseries (lazy) ─────────────────────────────────────────────────────────


class CycleTimeseries(BaseModel):
    """
    Raw per-timestep measurements for a single cycle.

    This object is never stored inside a Cycle or Battery instance.
    It is created on demand by Battery.load_timeseries() and can be
    discarded after feature extraction to keep memory usage manageable.

    Fields mirror the columns present across all Battery Archive timeseries
    CSVs.  Optional fields are None when the source dataset does not record
    them (e.g. not all datasets log cell temperature).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cycle_index: int = Field(description="Which cycle this trace belongs to.")

    # Core electrical measurements — always present
    time_s: np.ndarray = Field(description="Elapsed time within the cycle (seconds).")
    voltage_v: np.ndarray = Field(description="Terminal voltage (V).")
    current_a: np.ndarray = Field(
        description="Signed current (A).  Positive = charge, negative = discharge."
    )

    # Thermal — present in some datasets
    temperature_c: Optional[np.ndarray] = Field(
        default=None,
        description="Cell surface temperature (°C), when recorded.",
    )

    # Mechanical — present in Michigan Expansion (thickness gauge)
    thickness_mm: Optional[np.ndarray] = Field(
        default=None,
        description="Cell thickness / expansion (mm), Michigan Expansion dataset only.",
    )

    @property
    def n_points(self) -> int:
        return len(self.time_s)

    @property
    def charge_mask(self) -> np.ndarray:
        """Boolean mask selecting charge half-cycle steps."""
        return self.current_a > 0

    @property
    def discharge_mask(self) -> np.ndarray:
        """Boolean mask selecting discharge half-cycle steps."""
        return self.current_a < 0


# ── Cycle ─────────────────────────────────────────────────────────────────────


class Cycle(BaseModel):
    """
    Summary statistics for a single charge/discharge cycle.

    These values come directly from the *_cycle_data.csv files in the Battery
    Archive — one row per cycle.  They are always present and never require
    loading the large timeseries file.

    is_formation marks cycles that are part of a formation protocol (slow
    conditioning cycles run on fresh cells before normal cycling begins).
    These are mechanistically distinct from regular cycles and should usually
    be excluded from degradation modelling unless formation behaviour is the
    target.

    features holds any scalar values derived from the raw timeseries by a
    feature extractor (e.g. dQ/dV peak position, coulombic efficiency).  It
    starts empty; population is the responsibility of feature-extraction code
    that runs later in the pipeline.
    """

    # Identity
    cycle_index: int = Field(ge=0, description="1-based cycle number from the source file.")

    # Capacity (Ah)
    charge_capacity_ah: float = Field(ge=0.0, description="Total charge delivered during charging (Ah).")
    discharge_capacity_ah: float = Field(ge=0.0, description="Total charge delivered during discharging (Ah).")

    # Energy (Wh)
    charge_energy_wh: float = Field(ge=0.0, description="Total energy delivered during charging (Wh).")
    discharge_energy_wh: float = Field(ge=0.0, description="Total energy delivered during discharging (Wh).")

    # Voltage bounds observed during this cycle
    min_voltage_v: float = Field(description="Minimum terminal voltage recorded (V).")
    max_voltage_v: float = Field(description="Maximum terminal voltage recorded (V).")

    # Current bounds
    min_current_a: float = Field(description="Most negative current (deepest discharge, A).")
    max_current_a: float = Field(description="Most positive current (peak charge, A).")

    # Duration
    test_time_s: float = Field(ge=0.0, description="Wall-clock duration of this cycle (seconds).")

    # Flags
    is_formation: bool = Field(
        default=False,
        description=(
            "True for formation-protocol cycles.  These are slow conditioning "
            "cycles at the start of a cell's life and are mechanistically "
            "distinct from regular cycling."
        ),
    )

    # Extensible derived features — populated by feature extractors
    features: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Scalar features derived from the raw timeseries, e.g. dQ/dV peak "
            "position, coulombic efficiency, plateau variance.  Empty until a "
            "feature extractor populates it."
        ),
    )

    @property
    def coulombic_efficiency(self) -> Optional[float]:
        """Discharge / charge capacity.  None if charge capacity is zero."""
        if self.charge_capacity_ah == 0:
            return None
        return self.discharge_capacity_ah / self.charge_capacity_ah

    @property
    def energy_efficiency(self) -> Optional[float]:
        """Discharge / charge energy.  None if charge energy is zero."""
        if self.charge_energy_wh == 0:
            return None
        return self.discharge_energy_wh / self.charge_energy_wh

    @model_validator(mode="after")
    def _voltage_bounds_consistent(self) -> "Cycle":
        if self.min_voltage_v > self.max_voltage_v:
            raise ValueError(
                f"min_voltage_v ({self.min_voltage_v}) > max_voltage_v ({self.max_voltage_v})"
            )
        return self


# ── Battery ───────────────────────────────────────────────────────────────────


class Battery(BaseModel):
    """
    A single physical cell and its complete cycling history.

    This is the primary unit of ML analysis.  Each Battery maps to a pair of
    source files in the Battery Archive:
        <cell_id>_cycle_data.csv     — loaded eagerly into self.cycles
        <cell_id>_timeseries*.csv    — referenced by self._timeseries_zip_path
                                       and self._timeseries_entry, loaded lazily

    Metadata fields are parsed from the standardised Battery Archive filename
    convention:
        <DATASET>_<cell_id>_<form_factor>_<chemistry>_<temp>C_<soc>_<crate>_<rep>

    nominal_capacity_ah is the rated capacity of the cell.  When not
    explicitly known, it can be estimated as the median discharge capacity
    over the first few cycles.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Identity
    cell_id: str = Field(description="Unique cell identifier, derived from source filename stem.")
    dataset_name: str = Field(description="Parent dataset name (e.g. 'SNL_LFP', 'CALCE').")

    # Physical / experimental metadata
    chemistry: Chemistry = Field(description="Cathode chemistry.")
    form_factor: FormFactor = Field(description="Cell form factor.")
    temperature_c: float = Field(description="Test temperature (°C).")
    charge_rate_c: float = Field(gt=0.0, description="Charge C-rate (e.g. 0.5 means C/2).")
    discharge_rate_c: float = Field(gt=0.0, description="Discharge C-rate.")
    nominal_capacity_ah: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=(
            "Rated / nominal capacity (Ah).  If not known a priori, estimate "
            "as median discharge capacity over the first 3–5 non-formation cycles."
        ),
    )

    # Provenance flags
    has_formation_cycles: bool = Field(
        default=False,
        description="True if this cell's cycle list includes formation-protocol cycles.",
    )
    failure_mechanism: Optional[FailureMechanism] = Field(
        default=None,
        description="Annotated failure or degradation mode, if known (UL-Purdue only).",
    )

    # Cycling data — always loaded
    cycles: list[Cycle] = Field(
        default_factory=list,
        description="Ordered list of Cycle objects, one per charge/discharge pair.",
    )

    # Timeseries source — NOT serialised; used only for lazy loading
    _timeseries_zip_path: Optional[Path] = None
    _timeseries_entry: Optional[str] = None

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def n_cycles(self) -> int:
        return len(self.cycles)

    @property
    def regular_cycles(self) -> list[Cycle]:
        """Cycles with is_formation=False."""
        return [c for c in self.cycles if not c.is_formation]

    @property
    def formation_cycles(self) -> list[Cycle]:
        return [c for c in self.cycles if c.is_formation]

    @property
    def discharge_capacity_curve(self) -> np.ndarray:
        """Discharge capacity (Ah) as a 1-D array indexed by cycle position."""
        return np.array([c.discharge_capacity_ah for c in self.cycles])

    @property
    def capacity_retention(self) -> Optional[float]:
        """
        Final / initial discharge capacity.
        Computed over the first and last 3 regular cycles to smooth noise.
        Returns None if fewer than 6 regular cycles are present.
        """
        reg = self.regular_cycles
        if len(reg) < 6:
            return None
        initial = np.mean([c.discharge_capacity_ah for c in reg[:3]])
        final = np.mean([c.discharge_capacity_ah for c in reg[-3:]])
        return float(final / initial) if initial > 0 else None

    @property
    def end_of_life_cycle(self) -> Optional[int]:
        """
        First cycle index at which capacity retention drops below 80%.
        Returns None if the cell never reaches 80% fade or data is insufficient.
        """
        nom = self.nominal_capacity_ah
        if nom is None:
            reg = self.regular_cycles
            if len(reg) < 3:
                return None
            nom = float(np.mean([c.discharge_capacity_ah for c in reg[:3]]))
        threshold = 0.8 * nom
        for c in self.cycles:
            if c.discharge_capacity_ah < threshold:
                return c.cycle_index
        return None

    # ── Timeseries access ─────────────────────────────────────────────────────

    def set_timeseries_source(self, zip_path: Path, entry: str) -> None:
        """Register where the raw timeseries CSV lives (inside a zip)."""
        self._timeseries_zip_path = zip_path
        self._timeseries_entry = entry

    def load_timeseries(
        self,
        cycle_indices: Optional[list[int]] = None,
    ) -> dict[int, CycleTimeseries]:
        """
        Load raw timeseries data from the source zip on demand.

        Parameters
        ----------
        cycle_indices:
            Which cycles to load.  None = load all cycles.
            Pass a subset when only a few cycles are needed (e.g. cycle 1,
            cycle 10, cycle 100) to avoid reading the full file.

        Returns
        -------
        dict mapping cycle_index → CycleTimeseries
        """
        if self._timeseries_zip_path is None or self._timeseries_entry is None:
            raise RuntimeError(
                f"No timeseries source registered for battery '{self.cell_id}'. "
                "Call set_timeseries_source() before load_timeseries()."
            )

        with zipfile.ZipFile(self._timeseries_zip_path) as zf:
            with zf.open(self._timeseries_entry) as f:
                df = pd.read_csv(f)

        return _parse_timeseries_df(df, cycle_indices=cycle_indices)

    # ── Serialisation helpers ─────────────────────────────────────────────────

    def summary(self) -> dict:
        """Compact dict of key metadata — useful for building DataFrames."""
        return {
            "cell_id": self.cell_id,
            "dataset": self.dataset_name,
            "chemistry": self.chemistry.value,
            "form_factor": self.form_factor.value,
            "temperature_c": self.temperature_c,
            "charge_rate_c": self.charge_rate_c,
            "discharge_rate_c": self.discharge_rate_c,
            "nominal_capacity_ah": self.nominal_capacity_ah,
            "n_cycles": self.n_cycles,
            "has_formation_cycles": self.has_formation_cycles,
            "failure_mechanism": (
                self.failure_mechanism.value if self.failure_mechanism else None
            ),
            "capacity_retention": self.capacity_retention,
            "end_of_life_cycle": self.end_of_life_cycle,
        }


# ── Dataset ───────────────────────────────────────────────────────────────────


class Dataset(BaseModel):
    """
    A named collection of Battery objects from the same experimental campaign.

    Dataset is a thin grouping container.  It does not own file handles or
    parsing logic — those live in the loader module.  Its job is to:
      1. Provide a stable name that can be used as a label in ML pipelines.
      2. Hold all batteries from a single source zip as a queryable list.
      3. Expose aggregate statistics useful for EDA and sanity checks.

    Note on SNL: the Battery Archive distributes SNL data in three separate
    zips (SNL LFP, SNL NCA, SNL NMC).  Each is modelled as its own Dataset
    instance (name='SNL_LFP', 'SNL_NCA', 'SNL_NMC') because they test
    fundamentally different chemistries with different degradation physics.
    Merging them would conflate distinct populations.
    """

    name: str = Field(description="Human-readable dataset identifier, e.g. 'SNL_LFP'.")
    source_zip: str = Field(description="Filename of the source zip archive.")
    batteries: list[Battery] = Field(default_factory=list)

    # ── Convenience ──────────────────────────────────────────────────────────

    @property
    def n_batteries(self) -> int:
        return len(self.batteries)

    @property
    def chemistries(self) -> set[Chemistry]:
        return {b.chemistry for b in self.batteries}

    @property
    def cycle_counts(self) -> list[int]:
        return [b.n_cycles for b in self.batteries]

    def summary_df(self) -> pd.DataFrame:
        """Return a DataFrame with one row per battery — useful for EDA."""
        return pd.DataFrame([b.summary() for b in self.batteries])

    def get_battery(self, cell_id: str) -> Battery:
        """Look up a battery by cell_id.  Raises KeyError if not found."""
        for b in self.batteries:
            if b.cell_id == cell_id:
                return b
        raise KeyError(f"No battery with cell_id='{cell_id}' in dataset '{self.name}'.")


# ── Internal helpers ──────────────────────────────────────────────────────────


# Column name variants across Battery Archive datasets
_COL_ALIASES: dict[str, list[str]] = {
    "cycle_index":   ["Cycle_Index", "cycle_index", "Cycle Index"],
    "time_s":        ["Test_Time (s)", "test_time_s", "Time (s)", "time_s"],
    "voltage_v":     ["Voltage (V)", "voltage_v", "Voltage(V)"],
    "current_a":     ["Current (A)", "current_a", "Current(A)"],
    "temperature_c": ["Temperature (C)", "temperature_c", "Temp (C)", "Cell_Temperature_C"],
    "thickness_mm":  ["Thickness (mm)", "thickness_mm", "Expansion (mm)"],
}


def _resolve_column(df: pd.DataFrame, canonical: str) -> Optional[str]:
    """Return the first matching column name alias, or None."""
    for alias in _COL_ALIASES.get(canonical, []):
        if alias in df.columns:
            return alias
    return None


def _parse_timeseries_df(
    df: pd.DataFrame,
    cycle_indices: Optional[list[int]] = None,
) -> dict[int, CycleTimeseries]:
    """
    Convert a raw timeseries DataFrame (read from a Battery Archive CSV) into
    a dict of CycleTimeseries objects, one per cycle.
    """
    cycle_col = _resolve_column(df, "cycle_index")
    time_col = _resolve_column(df, "time_s")
    voltage_col = _resolve_column(df, "voltage_v")
    current_col = _resolve_column(df, "current_a")
    temp_col = _resolve_column(df, "temperature_c")
    thick_col = _resolve_column(df, "thickness_mm")

    if not all([cycle_col, time_col, voltage_col, current_col]):
        missing = [
            name for name, col in zip(
                ["cycle_index", "time_s", "voltage_v", "current_a"],
                [cycle_col, time_col, voltage_col, current_col],
            )
            if col is None
        ]
        raise ValueError(f"Timeseries CSV missing required columns: {missing}")

    if cycle_indices is not None:
        df = df[df[cycle_col].isin(cycle_indices)]

    result: dict[int, CycleTimeseries] = {}
    for idx, group in df.groupby(cycle_col):
        result[int(idx)] = CycleTimeseries(
            cycle_index=int(idx),
            time_s=group[time_col].to_numpy(dtype=float),
            voltage_v=group[voltage_col].to_numpy(dtype=float),
            current_a=group[current_col].to_numpy(dtype=float),
            temperature_c=(
                group[temp_col].to_numpy(dtype=float) if temp_col else None
            ),
            thickness_mm=(
                group[thick_col].to_numpy(dtype=float) if thick_col else None
            ),
        )

    return result
