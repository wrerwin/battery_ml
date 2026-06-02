"""
Battery cycle feature extraction.

Architecture
------------
FeatureSpec
    Named, tagged descriptor for a single scalar feature.  Separates the
    *what* (name, description, leakage notes) from the *how* (compute fn).

CycleFeaturizer
    Holds a registry of FeatureSpec objects.  Given a Battery, it iterates
    over cycles up to max_cycle and produces a per-cycle DataFrame:
        columns: cell_id | cycle_index | <feature_0> | <feature_1> | ...

BatteryFeaturizer
    Wraps CycleFeaturizer.  Takes the per-cycle DataFrame and collapses it
    into a single per-battery feature vector using a set of aggregation
    strategies (point-in-time values, mean, std, linear slope, deltas).
    Output: one row per battery, suitable for direct ML input.

FeatureStore
    Persists and retrieves feature DataFrames as Parquet files.  Filenames
    encode extraction parameters so different (max_cycle, formation_only)
    combinations coexist without overwriting each other.

Leakage safety
--------------
max_cycle is the hard cutoff: no cycle with index > max_cycle is ever
touched during feature extraction.  The EOL label (Battery.end_of_life_cycle)
is NEVER read inside this module — it is joined at training time only.

Cross-battery normalisation (StandardScaler, etc.) must be fit exclusively
on the training split.  FeatureStore stores raw, un-normalised features by
design.

Formation-only mode
-------------------
When formation_only=True, only cycles flagged is_formation=True contribute
to features.  This simulates the manufacturing context: predict final cell
performance using only data available at the end of the formation step.

NOTE: the current schema flags formation at the dataset level (all cycles in
Michigan_Formation are formation; none elsewhere).  Per-cycle formation
detection within mixed datasets is a TODO for when that data is available.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from schema import Battery, Cycle, CycleTimeseries

FEATURES_DIR = Path(__file__).parent / "features"

# Cycles at which we snapshot point-in-time feature values.
# Only snapshots with index <= max_cycle are included.
SNAPSHOT_CYCLES = [1, 2, 3, 5, 10, 20, 50, 100]


# ── FeatureSpec ───────────────────────────────────────────────────────────────


@dataclass
class FeatureSpec:
    """
    Descriptor for a single scalar feature extracted per cycle.

    Parameters
    ----------
    name:
        Snake_case identifier used as the DataFrame column name.
    description:
        Human-readable explanation of what this feature captures and why
        it might be predictive.
    requires_timeseries:
        True if compute() needs raw voltage/current traces.  False if it
        can be derived from the Cycle summary object alone.  This drives
        whether load_timeseries() is called, so keep it accurate.
    compute:
        fn(cycle, timeseries) -> float | None.
        Return None to signal that the feature could not be computed for
        this cycle (e.g. timeseries too short, division by zero).  Nones
        become NaN in the output DataFrame.
    tags:
        Free-form labels for filtering/grouping features.
        Suggested values: "capacity", "energy", "efficiency", "curve_shape",
        "impedance", "formation", "timeseries".
    leakage_note:
        Optional note explaining any leakage risk.  Most features have none;
        this field exists as documentation for reviewers.
    """

    name: str
    description: str
    requires_timeseries: bool
    compute: Callable[[Cycle, Optional[CycleTimeseries]], Optional[float]]
    tags: list[str] = field(default_factory=list)
    leakage_note: str = ""


# ── Built-in feature definitions ──────────────────────────────────────────────
#
# Two registries:
#   CYCLE_SUMMARY_FEATURES  — derived from Cycle objects only (fast)
#   TIMESERIES_FEATURES     — require raw V/I traces (slower, lazy-loaded)
#
# Add new features by appending to whichever list fits.  The featurizer picks
# them up automatically via CycleFeaturizer.register().


def _safe(fn: Callable) -> Callable:
    """Wrap a compute function so exceptions return None instead of crashing."""
    def wrapper(cycle: Cycle, ts: Optional[CycleTimeseries]) -> Optional[float]:
        try:
            return fn(cycle, ts)
        except Exception:
            return None
    return wrapper


# ── Cycle-summary features (no timeseries) ────────────────────────────────────

CYCLE_SUMMARY_FEATURES: list[FeatureSpec] = [

    FeatureSpec(
        name="discharge_capacity_ah",
        description="Total discharge capacity in Ah.  Primary indicator of cell health.",
        requires_timeseries=False,
        tags=["capacity"],
        compute=_safe(lambda c, _: c.discharge_capacity_ah),
    ),

    FeatureSpec(
        name="charge_capacity_ah",
        description="Total charge capacity in Ah.",
        requires_timeseries=False,
        tags=["capacity"],
        compute=_safe(lambda c, _: c.charge_capacity_ah),
    ),

    FeatureSpec(
        name="coulombic_efficiency",
        description=(
            "Discharge / charge capacity.  Drops as side reactions (SEI growth, "
            "lithium plating) consume cyclable lithium."
        ),
        requires_timeseries=False,
        tags=["efficiency"],
        compute=_safe(lambda c, _: c.coulombic_efficiency),
    ),

    FeatureSpec(
        name="energy_efficiency",
        description="Discharge / charge energy.  Sensitive to both capacity loss and voltage polarisation.",
        requires_timeseries=False,
        tags=["efficiency", "energy"],
        compute=_safe(lambda c, _: c.energy_efficiency),
    ),

    FeatureSpec(
        name="discharge_energy_wh",
        description="Total energy delivered during discharge (Wh).",
        requires_timeseries=False,
        tags=["energy"],
        compute=_safe(lambda c, _: c.discharge_energy_wh),
    ),

    FeatureSpec(
        name="charge_energy_wh",
        description="Total energy consumed during charge (Wh).",
        requires_timeseries=False,
        tags=["energy"],
        compute=_safe(lambda c, _: c.charge_energy_wh),
    ),

    FeatureSpec(
        name="voltage_window_v",
        description=(
            "Max - min voltage observed during the cycle.  Narrows as active "
            "material becomes electrochemically inactive."
        ),
        requires_timeseries=False,
        tags=["voltage"],
        compute=_safe(lambda c, _: c.max_voltage_v - c.min_voltage_v),
    ),

    FeatureSpec(
        name="test_time_s",
        description="Wall-clock duration of the cycle in seconds.",
        requires_timeseries=False,
        tags=["duration"],
        compute=_safe(lambda c, _: c.test_time_s),
    ),

    FeatureSpec(
        name="specific_energy_wh_per_ah",
        description=(
            "Discharge energy / discharge capacity (Wh/Ah) — mean discharge voltage. "
            "Decreases as internal resistance rises and voltage polarisation grows."
        ),
        requires_timeseries=False,
        tags=["energy", "voltage"],
        compute=_safe(
            lambda c, _: (
                c.discharge_energy_wh / c.discharge_capacity_ah
                if c.discharge_capacity_ah > 0 else None
            )
        ),
    ),
]


# ── Timeseries features ────────────────────────────────────────────────────────

def _discharge_voltage_stats(
    cycle: Cycle, ts: Optional[CycleTimeseries]
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (mean, std, area) of discharge voltage trace, or (None,None,None)."""
    if ts is None:
        return None, None, None
    mask = ts.discharge_mask
    if mask.sum() < 10:
        return None, None, None
    v = ts.voltage_v[mask]
    t = ts.time_s[mask]
    area = float(np.trapz(v, t)) if len(t) > 1 else None
    return float(v.mean()), float(v.std()), area


def _charge_voltage_stats(
    cycle: Cycle, ts: Optional[CycleTimeseries]
) -> tuple[Optional[float], Optional[float]]:
    if ts is None:
        return None, None
    mask = ts.charge_mask
    if mask.sum() < 10:
        return None, None
    v = ts.voltage_v[mask]
    return float(v.mean()), float(v.std())


def _dqdv_peaks(
    cycle: Cycle, ts: Optional[CycleTimeseries]
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Compute differential capacity (dQ/dV) for the discharge half-cycle and
    return the position and height of the two tallest peaks.

    dQ/dV peaks correspond to voltage plateaus in the discharge curve, which
    reflect phase transitions in the cathode.  Their position and height shift
    as active material degrades — particularly useful for LFP and NMC.

    Returns (peak1_v, peak1_height, peak2_v, peak2_height).
    """
    if ts is None:
        return None, None, None, None

    mask = ts.discharge_mask
    if mask.sum() < 50:
        return None, None, None, None

    v = ts.voltage_v[mask]
    i = ts.current_a[mask]
    t = ts.time_s[mask]

    # Integrate current over time to get cumulative charge (Q) in As
    q = np.abs(np.cumsum(i * np.gradient(t)))

    # Sort by voltage (ascending) for dQ/dV
    sort_idx = np.argsort(v)
    v_sorted = v[sort_idx]
    q_sorted = q[sort_idx]

    # Remove duplicate voltages
    _, unique_idx = np.unique(v_sorted, return_index=True)
    v_u = v_sorted[unique_idx]
    q_u = q_sorted[unique_idx]

    if len(v_u) < 20:
        return None, None, None, None

    # Interpolate to a uniform voltage grid for robust differentiation
    v_grid = np.linspace(v_u.min(), v_u.max(), min(500, len(v_u)))
    q_grid = np.interp(v_grid, v_u, q_u)

    dqdv = np.gradient(q_grid, v_grid)

    # Smooth to suppress noise
    window = min(21, len(dqdv) // 4 * 2 + 1)  # must be odd
    if window >= 5:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dqdv = savgol_filter(dqdv, window_length=window, polyorder=3)

    # Find peaks (discharge dQ/dV peaks are positive)
    peaks, props = find_peaks(dqdv, height=0, distance=10)
    if len(peaks) == 0:
        return None, None, None, None

    heights = props["peak_heights"]
    top2_idx = np.argsort(heights)[::-1][:2]
    top2_peaks = peaks[top2_idx]
    top2_heights = heights[top2_idx]

    p1_v = float(v_grid[top2_peaks[0]]) if len(top2_peaks) > 0 else None
    p1_h = float(top2_heights[0]) if len(top2_heights) > 0 else None
    p2_v = float(v_grid[top2_peaks[1]]) if len(top2_peaks) > 1 else None
    p2_h = float(top2_heights[1]) if len(top2_heights) > 1 else None

    return p1_v, p1_h, p2_v, p2_h


def _internal_resistance(
    cycle: Cycle, ts: Optional[CycleTimeseries]
) -> Optional[float]:
    """
    Estimate DC internal resistance (mΩ) from the voltage step at the
    onset of discharge.

    IR = |ΔV| / |ΔI| at the rest→discharge current transition.
    This is a coarse estimate but correlates well with impedance spectroscopy
    at low frequencies and tracks SEI growth over cycling.
    """
    if ts is None or len(ts.current_a) < 20:
        return None

    i = ts.current_a
    v = ts.voltage_v

    # Find where discharge starts: first point where current is strongly negative
    discharge_onset = np.where(i < -0.05 * np.abs(i.min()))[0]
    if len(discharge_onset) < 2:
        return None

    idx = discharge_onset[0]
    if idx == 0:
        return None

    di = abs(i[idx] - i[idx - 1])
    dv = abs(v[idx] - v[idx - 1])

    if di < 1e-6:
        return None

    return float((dv / di) * 1000)  # convert Ω → mΩ


def _cc_fraction(
    cycle: Cycle, ts: Optional[CycleTimeseries]
) -> Optional[float]:
    """
    Fraction of charge time spent in constant-current (CC) mode vs total
    charge time.  As cells age, the CC→CV transition happens earlier
    (lower SOC) because internal resistance rise limits CC capability.
    """
    if ts is None:
        return None
    mask = ts.charge_mask
    if mask.sum() < 10:
        return None

    v_charge = ts.voltage_v[mask]
    i_charge = np.abs(ts.current_a[mask])

    if len(v_charge) < 10:
        return None

    # Heuristic: CV mode begins when current starts dropping significantly.
    # Find the last point where current is > 90% of its maximum.
    i_max = i_charge.max()
    cc_mask = i_charge > 0.9 * i_max
    return float(cc_mask.sum() / len(i_charge))


TIMESERIES_FEATURES: list[FeatureSpec] = [

    FeatureSpec(
        name="discharge_mean_v",
        description=(
            "Mean voltage during discharge.  Drops as ohmic resistance and "
            "polarisation increase, even before capacity fade is visible."
        ),
        requires_timeseries=True,
        tags=["voltage", "curve_shape", "timeseries"],
        compute=_safe(lambda c, ts: _discharge_voltage_stats(c, ts)[0]),
    ),

    FeatureSpec(
        name="discharge_std_v",
        description=(
            "Standard deviation of discharge voltage curve.  Captures plateau "
            "shape changes — NMC curves become less flat as layered oxide "
            "disorders; LFP curves flatten as the two-phase region shrinks."
        ),
        requires_timeseries=True,
        tags=["voltage", "curve_shape", "timeseries"],
        compute=_safe(lambda c, ts: _discharge_voltage_stats(c, ts)[1]),
    ),

    FeatureSpec(
        name="discharge_voltage_area",
        description=(
            "Area under the discharge V-t curve (V·s).  Proportional to energy "
            "but normalised differently than Wh — useful as a shape descriptor."
        ),
        requires_timeseries=True,
        tags=["voltage", "curve_shape", "energy", "timeseries"],
        compute=_safe(lambda c, ts: _discharge_voltage_stats(c, ts)[2]),
    ),

    FeatureSpec(
        name="charge_mean_v",
        description="Mean voltage during charging.",
        requires_timeseries=True,
        tags=["voltage", "curve_shape", "timeseries"],
        compute=_safe(lambda c, ts: _charge_voltage_stats(c, ts)[0]),
    ),

    FeatureSpec(
        name="charge_std_v",
        description="Standard deviation of charge voltage curve.",
        requires_timeseries=True,
        tags=["voltage", "curve_shape", "timeseries"],
        compute=_safe(lambda c, ts: _charge_voltage_stats(c, ts)[1]),
    ),

    FeatureSpec(
        name="dqdv_peak1_voltage_v",
        description=(
            "Voltage of the tallest dQ/dV peak during discharge.  Corresponds "
            "to the main phase-transition plateau.  Shifts as active material "
            "composition changes with cycling."
        ),
        requires_timeseries=True,
        tags=["curve_shape", "dqdv", "timeseries"],
        compute=_safe(lambda c, ts: _dqdv_peaks(c, ts)[0]),
    ),

    FeatureSpec(
        name="dqdv_peak1_height",
        description=(
            "Height of the tallest dQ/dV peak.  Decreases as the phase "
            "transition becomes less sharp — an early indicator of active "
            "material loss."
        ),
        requires_timeseries=True,
        tags=["curve_shape", "dqdv", "timeseries"],
        compute=_safe(lambda c, ts: _dqdv_peaks(c, ts)[1]),
    ),

    FeatureSpec(
        name="dqdv_peak2_voltage_v",
        description="Voltage of the second-tallest dQ/dV peak, if present.",
        requires_timeseries=True,
        tags=["curve_shape", "dqdv", "timeseries"],
        compute=_safe(lambda c, ts: _dqdv_peaks(c, ts)[2]),
    ),

    FeatureSpec(
        name="dqdv_peak2_height",
        description="Height of the second-tallest dQ/dV peak, if present.",
        requires_timeseries=True,
        tags=["curve_shape", "dqdv", "timeseries"],
        compute=_safe(lambda c, ts: _dqdv_peaks(c, ts)[3]),
    ),

    FeatureSpec(
        name="internal_resistance_mohm",
        description=(
            "DC internal resistance estimated from the voltage step at "
            "discharge onset (mΩ).  Rises monotonically with SEI growth "
            "and active material cracking."
        ),
        requires_timeseries=True,
        tags=["impedance", "timeseries"],
        compute=_safe(lambda c, ts: _internal_resistance(c, ts)),
    ),

    FeatureSpec(
        name="cc_fraction",
        description=(
            "Fraction of charge time in constant-current mode.  Decreases "
            "as rising internal resistance forces earlier CC→CV transition."
        ),
        requires_timeseries=True,
        tags=["curve_shape", "timeseries"],
        compute=_safe(lambda c, ts: _cc_fraction(c, ts)),
    ),
]


# ── CycleFeaturizer ───────────────────────────────────────────────────────────


class CycleFeaturizer:
    """
    Extracts scalar features from individual cycles.

    Usage
    -----
        cf = CycleFeaturizer()               # all built-in features
        cf = CycleFeaturizer(tags=["capacity", "efficiency"])  # subset by tag
        cf.register(my_feature_spec)         # add a custom feature

        df = cf.compute(battery, max_cycle=100)
        # → DataFrame: cell_id | cycle_index | feature_0 | feature_1 | ...
    """

    def __init__(self, tags: Optional[list[str]] = None):
        """
        Parameters
        ----------
        tags:
            If given, only features whose tags overlap with this list are
            included.  None = include all built-in features.
        """
        self._specs: list[FeatureSpec] = []
        for spec in CYCLE_SUMMARY_FEATURES + TIMESERIES_FEATURES:
            if tags is None or any(t in spec.tags for t in tags):
                self._specs.append(spec)

    @property
    def feature_names(self) -> list[str]:
        return [s.name for s in self._specs]

    @property
    def requires_timeseries(self) -> bool:
        return any(s.requires_timeseries for s in self._specs)

    def register(self, spec: FeatureSpec) -> None:
        """Add a custom FeatureSpec to the registry."""
        if any(s.name == spec.name for s in self._specs):
            raise ValueError(f"Feature '{spec.name}' is already registered.")
        self._specs.append(spec)

    def catalog(self) -> pd.DataFrame:
        """Return a DataFrame describing all registered features."""
        return pd.DataFrame([
            {
                "name": s.name,
                "requires_timeseries": s.requires_timeseries,
                "tags": ", ".join(s.tags),
                "description": s.description,
                "leakage_note": s.leakage_note,
            }
            for s in self._specs
        ])

    def compute(
        self,
        battery: Battery,
        max_cycle: int = 100,
        formation_only: bool = False,
    ) -> pd.DataFrame:
        """
        Compute per-cycle features for all eligible cycles of a battery.

        Parameters
        ----------
        battery:
            The battery to featurize.
        max_cycle:
            Hard upper bound — cycles with index > max_cycle are excluded.
            This is the primary leakage guard.
        formation_only:
            If True, only cycles with is_formation=True are included.
            Raises ValueError if the battery has no formation cycles.

        Returns
        -------
        DataFrame with columns [cell_id, cycle_index, <features>].
        Rows are ordered by cycle_index.  NaN indicates a feature could
        not be computed for that cycle.
        """
        eligible = [
            c for c in battery.cycles
            if c.cycle_index <= max_cycle
            and (not formation_only or c.is_formation)
        ]

        if formation_only and len(eligible) == 0:
            raise ValueError(
                f"Battery '{battery.cell_id}' has no formation cycles. "
                "Set formation_only=False or use a dataset that includes formation data."
            )

        # Identify which cycles need timeseries
        ts_needed = self.requires_timeseries
        ts_map: dict[int, Optional[CycleTimeseries]] = {}

        if ts_needed:
            cycle_indices = [c.cycle_index for c in eligible]
            try:
                ts_map = battery.load_timeseries(cycle_indices=cycle_indices)
            except RuntimeError:
                # No timeseries registered — timeseries features will be NaN
                ts_map = {}

        rows = []
        for cycle in eligible:
            ts = ts_map.get(cycle.cycle_index)
            row: dict = {"cell_id": battery.cell_id, "cycle_index": cycle.cycle_index}
            for spec in self._specs:
                val = spec.compute(cycle, ts if spec.requires_timeseries else None)
                row[spec.name] = val if val is not None else np.nan
            rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=["cell_id", "cycle_index"] + self.feature_names)
        return df


# ── BatteryFeaturizer ─────────────────────────────────────────────────────────


class BatteryFeaturizer:
    """
    Aggregates per-cycle features into a single per-battery feature vector.

    Aggregation strategies applied to each per-cycle feature column:
      - Point-in-time: value at each cycle in SNAPSHOT_CYCLES
      - Summary stats:  mean, std, min, max over all cycles in the window
      - Trend:          slope of a linear fit over the window
      - Delta:          value at last cycle minus value at first cycle

    The EOL label is optionally appended as a column named 'eol_cycle' but
    only when include_label=True and the battery's end_of_life_cycle is known.
    Labels must never influence feature computation — they are appended purely
    for convenience and should be separated before fitting any model.
    """

    def __init__(self, cycle_featurizer: Optional[CycleFeaturizer] = None):
        self.cycle_featurizer = cycle_featurizer or CycleFeaturizer()

    def compute(
        self,
        battery: Battery,
        max_cycle: int = 100,
        formation_only: bool = False,
        include_label: bool = True,
    ) -> dict:
        """
        Produce a single feature dict for one battery.

        Parameters
        ----------
        include_label:
            If True, append 'eol_cycle' (the regression target) as the last
            key.  Keep this True only for dataset construction; set it False
            when building inference inputs to make the leakage boundary explicit.
        """
        cycle_df = self.cycle_featurizer.compute(
            battery, max_cycle=max_cycle, formation_only=formation_only
        )

        feature_cols = [
            c for c in cycle_df.columns
            if c not in ("cell_id", "cycle_index")
        ]

        row: dict = {
            "cell_id": battery.cell_id,
            "dataset": battery.dataset_name,
            "chemistry": battery.chemistry.value,
            "form_factor": battery.form_factor.value,
            "temperature_c": battery.temperature_c,
            "charge_rate_c": battery.charge_rate_c,
            "discharge_rate_c": battery.discharge_rate_c,
            "n_cycles_observed": len(cycle_df),
            "max_cycle_used": max_cycle,
            "formation_only": formation_only,
        }

        if cycle_df.empty:
            if include_label:
                row["eol_cycle"] = battery.end_of_life_cycle
            return row

        cycle_idx = cycle_df["cycle_index"].values

        for feat in feature_cols:
            vals = cycle_df[feat].values.astype(float)
            valid = ~np.isnan(vals)

            # Point-in-time snapshots
            for snap_cycle in SNAPSHOT_CYCLES:
                if snap_cycle > max_cycle:
                    continue
                col_name = f"{feat}_at_c{snap_cycle}"
                mask = cycle_idx == snap_cycle
                if mask.any() and valid[mask].any():
                    row[col_name] = float(vals[mask][0])
                else:
                    row[col_name] = np.nan

            if valid.sum() < 2:
                for suffix in ("mean", "std", "min", "max", "slope", "delta"):
                    row[f"{feat}_{suffix}"] = np.nan
                continue

            v = vals[valid]
            t = cycle_idx[valid]

            row[f"{feat}_mean"] = float(v.mean())
            row[f"{feat}_std"] = float(v.std())
            row[f"{feat}_min"] = float(v.min())
            row[f"{feat}_max"] = float(v.max())

            # Linear slope (cycles as the x-axis)
            if len(t) >= 2:
                slope, _ = np.polyfit(t, v, 1)
                row[f"{feat}_slope"] = float(slope)
            else:
                row[f"{feat}_slope"] = np.nan

            # Delta: last valid - first valid
            row[f"{feat}_delta"] = float(v[-1] - v[0])

        if include_label:
            row["eol_cycle"] = battery.end_of_life_cycle

        return row

    def compute_dataset(
        self,
        batteries: list[Battery],
        max_cycle: int = 100,
        formation_only: bool = False,
        include_label: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Compute per-battery feature rows for a list of batteries.

        Returns a DataFrame with one row per battery.  Suitable for direct
        use as an ML feature matrix after separating the 'eol_cycle' column.
        """
        rows = []
        for i, battery in enumerate(batteries):
            if verbose and (i % 10 == 0 or i == len(batteries) - 1):
                print(f"  [{i+1}/{len(batteries)}] {battery.cell_id}")
            try:
                rows.append(
                    self.compute(
                        battery,
                        max_cycle=max_cycle,
                        formation_only=formation_only,
                        include_label=include_label,
                    )
                )
            except Exception as exc:
                print(f"  WARN: skipping {battery.cell_id}: {exc}")

        return pd.DataFrame(rows)


# ── FeatureStore ──────────────────────────────────────────────────────────────


class FeatureStore:
    """
    Parquet-backed cache for feature DataFrames.

    Directory layout
    ----------------
    features/
    ├── cycle_features/
    │   └── {dataset}_mc{max_cycle}_form{0|1}.parquet
    ├── battery_features/
    │   └── {dataset}_mc{max_cycle}_form{0|1}.parquet
    └── feature_catalog.json

    Files are keyed by (dataset_name, max_cycle, formation_only) so different
    experiment configurations coexist without overwriting each other.
    """

    def __init__(self, base_dir: Path = FEATURES_DIR):
        self.base_dir = base_dir
        self._cycle_dir = base_dir / "cycle_features"
        self._battery_dir = base_dir / "battery_features"
        for d in (self._cycle_dir, self._battery_dir):
            d.mkdir(parents=True, exist_ok=True)

    def _key(self, dataset_name: str, max_cycle: int, formation_only: bool) -> str:
        form_flag = "form1" if formation_only else "form0"
        return f"{dataset_name}_mc{max_cycle}_{form_flag}"

    # ── Cycle-level features ──────────────────────────────────────────────────

    def save_cycle_features(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        max_cycle: int,
        formation_only: bool = False,
    ) -> Path:
        path = self._cycle_dir / f"{self._key(dataset_name, max_cycle, formation_only)}.parquet"
        df.to_parquet(path, index=False, engine="pyarrow")
        return path

    def load_cycle_features(
        self,
        dataset_name: str,
        max_cycle: int,
        formation_only: bool = False,
    ) -> pd.DataFrame:
        path = self._cycle_dir / f"{self._key(dataset_name, max_cycle, formation_only)}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"No cached cycle features for {dataset_name} mc={max_cycle}. "
                "Run CycleFeaturizer.compute() first."
            )
        return pd.read_parquet(path, engine="pyarrow")

    def has_cycle_features(
        self, dataset_name: str, max_cycle: int, formation_only: bool = False
    ) -> bool:
        path = self._cycle_dir / f"{self._key(dataset_name, max_cycle, formation_only)}.parquet"
        return path.exists()

    # ── Battery-level features ────────────────────────────────────────────────

    def save_battery_features(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        max_cycle: int,
        formation_only: bool = False,
    ) -> Path:
        path = self._battery_dir / f"{self._key(dataset_name, max_cycle, formation_only)}.parquet"
        df.to_parquet(path, index=False, engine="pyarrow")
        return path

    def load_battery_features(
        self,
        dataset_name: str,
        max_cycle: int,
        formation_only: bool = False,
    ) -> pd.DataFrame:
        path = self._battery_dir / f"{self._key(dataset_name, max_cycle, formation_only)}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"No cached battery features for {dataset_name} mc={max_cycle}. "
                "Run BatteryFeaturizer.compute_dataset() first."
            )
        return pd.read_parquet(path, engine="pyarrow")

    def has_battery_features(
        self, dataset_name: str, max_cycle: int, formation_only: bool = False
    ) -> bool:
        path = self._battery_dir / f"{self._key(dataset_name, max_cycle, formation_only)}.parquet"
        return path.exists()

    def load_all_battery_features(
        self,
        max_cycle: int,
        formation_only: bool = False,
    ) -> pd.DataFrame:
        """Load and concatenate all cached battery feature files for a given config."""
        dfs = []
        form_flag = "form1" if formation_only else "form0"
        pattern = f"*_mc{max_cycle}_{form_flag}.parquet"
        for path in sorted(self._battery_dir.glob(pattern)):
            dfs.append(pd.read_parquet(path, engine="pyarrow"))
        if not dfs:
            raise FileNotFoundError(
                f"No cached battery features found for mc={max_cycle}, "
                f"formation_only={formation_only}."
            )
        return pd.concat(dfs, ignore_index=True)

    # ── Catalog ───────────────────────────────────────────────────────────────

    def save_catalog(self, featurizer: CycleFeaturizer) -> Path:
        """Persist the feature catalog (names + descriptions) as JSON."""
        path = self.base_dir / "feature_catalog.json"
        catalog = featurizer.catalog().to_dict(orient="records")
        path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
        return path

    def list_cached(self) -> pd.DataFrame:
        """Return a DataFrame listing all cached Parquet files."""
        records = []
        for level, d in [("cycle", self._cycle_dir), ("battery", self._battery_dir)]:
            for p in sorted(d.glob("*.parquet")):
                size_mb = p.stat().st_size / 1e6
                records.append({"level": level, "file": p.name, "size_mb": round(size_mb, 2)})
        return pd.DataFrame(records)
