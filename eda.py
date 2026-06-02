#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "numpy",
# ]
# ///
"""
Battery Archive EDA
Run with: uv run eda.py
"""

import io
import re
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

DATA_DIR = Path(__file__).parent / "battery_archive_data"
OUT_DIR = Path(__file__).parent / "eda_output"
OUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="tab10")
FIGSIZE_WIDE = (14, 6)
FIGSIZE_TALL = (10, 8)

# ── filename parser ────────────────────────────────────────────────────────────

_FAILURE_TAGS = {
    "EX": "External Short",
    "OV": "Overvoltage",
    "NA": "Normal Aging",
    "DPA": "Degradation Protocol A",
}

def parse_filename(name: str) -> dict:
    """Extract metadata encoded in Battery Archive filenames."""
    stem = Path(name).stem

    # dataset prefix (first token before first underscore after dataset name)
    dataset_map = {
        "CALCE": "CALCE",
        "HNEI": "HNEI",
        "MICH": "Michigan",
        "OX": "Oxford",
        "SNL": "SNL",
        "UL-PUR": "UL-Purdue",
    }
    dataset = next((v for k, v in dataset_map.items() if stem.startswith(k)), "Unknown")

    # chemistry from filename tokens
    chemistry = None
    for chem in ["NMC_LCO", "LFP", "NCA", "NMC", "LCO"]:
        if chem in stem:
            chemistry = chem
            break

    # form factor
    form_factor = None
    for ff in ["18650", "pouch", "prism"]:
        if ff in stem.lower():
            form_factor = ff
            break

    # temperature  e.g. 25C, -5C, 45C
    temp_match = re.search(r'_(-?\d+)C_', stem)
    temperature_c = int(temp_match.group(1)) if temp_match else None

    # C-rates e.g. 0.5-1C or 0.5-0.5C
    crate_match = re.search(r'_([\d.]+)-([\d.]+)C_', stem)
    charge_rate = float(crate_match.group(1)) if crate_match else None
    discharge_rate = float(crate_match.group(2)) if crate_match else None

    # failure mechanism (UL-Purdue encodes in cell ID)
    failure_mechanism = None
    for tag, label in _FAILURE_TAGS.items():
        if f"-{tag}" in stem or f"_{tag}" in stem:
            failure_mechanism = label
            break
    # CF10DPA style
    dpa_match = re.search(r'DPA', stem)
    if dpa_match and failure_mechanism is None:
        failure_mechanism = "Degradation Protocol A"

    # formation cycles: Michigan Formation dataset
    has_formation = "Formation" in name or "BLForm" in stem

    return {
        "dataset": dataset,
        "chemistry": chemistry,
        "form_factor": form_factor,
        "temperature_c": temperature_c,
        "charge_rate_C": charge_rate,
        "discharge_rate_C": discharge_rate,
        "has_formation_cycles": has_formation,
        "failure_mechanism": failure_mechanism,
    }


# ── data loading ───────────────────────────────────────────────────────────────

def load_all_cycle_data() -> pd.DataFrame:
    records = []
    for zip_path in sorted(DATA_DIR.glob("*.zip")):
        print(f"  Reading {zip_path.name} …")
        with zipfile.ZipFile(zip_path) as zf:
            cycle_entries = [e for e in zf.namelist() if e.endswith("cycle_data.csv")]
            for entry in cycle_entries:
                meta = parse_filename(entry)
                meta["zip_file"] = zip_path.name
                meta["cell_file"] = Path(entry).name

                with zf.open(entry) as f:
                    try:
                        df = pd.read_csv(f)
                    except Exception as exc:
                        print(f"    WARN: could not read {entry}: {exc}")
                        continue

                meta["n_cycles"] = len(df)
                meta["columns"] = list(df.columns)

                # capacity fade proxy: first vs last discharge capacity
                cap_col = next(
                    (c for c in df.columns if "Discharge_Capacity" in c), None
                )
                if cap_col and len(df) > 5:
                    valid = df[cap_col].replace(0, np.nan).dropna()
                    if len(valid) > 5:
                        meta["cap_initial_ah"] = valid.iloc[:3].mean()
                        meta["cap_final_ah"] = valid.iloc[-3:].mean()
                        meta["capacity_retention"] = (
                            meta["cap_final_ah"] / meta["cap_initial_ah"]
                            if meta["cap_initial_ah"] > 0 else np.nan
                        )
                    else:
                        meta["cap_initial_ah"] = np.nan
                        meta["cap_final_ah"] = np.nan
                        meta["capacity_retention"] = np.nan
                else:
                    meta["cap_initial_ah"] = np.nan
                    meta["cap_final_ah"] = np.nan
                    meta["capacity_retention"] = np.nan

                records.append(meta)

    return pd.DataFrame(records)


# ── summary table ──────────────────────────────────────────────────────────────

def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, grp in df.groupby("dataset"):
        rows.append({
            "Dataset": ds,
            "N cells": len(grp),
            "Chemistry": ", ".join(grp["chemistry"].dropna().unique()),
            "Form factor": ", ".join(grp["form_factor"].dropna().unique()),
            "Temp (°C)": ", ".join(
                str(t) for t in sorted(grp["temperature_c"].dropna().unique().astype(int))
            ),
            "Formation cycles": "Yes" if grp["has_formation_cycles"].any() else "No",
            "Failure annotated": "Yes" if grp["failure_mechanism"].notna().any() else "No",
            "Failure types": ", ".join(
                grp["failure_mechanism"].dropna().unique()
            ) or "—",
            "Median cycles": int(grp["n_cycles"].median()),
            "Min cycles": int(grp["n_cycles"].min()),
            "Max cycles": int(grp["n_cycles"].max()),
        })
    return pd.DataFrame(rows).sort_values("Dataset").reset_index(drop=True)


# ── figures ────────────────────────────────────────────────────────────────────

def fig_cycle_distribution(df: pd.DataFrame):
    """Box + strip plot of cycle counts per dataset."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    order = df.groupby("dataset")["n_cycles"].median().sort_values(ascending=False).index
    sns.boxplot(
        data=df, x="dataset", y="n_cycles", hue="dataset",
        order=order, ax=ax, width=0.5, fliersize=0,
        palette="tab10", legend=False,
    )
    sns.stripplot(
        data=df, x="dataset", y="n_cycles",
        order=order, ax=ax, size=5, alpha=0.7, color="black",
    )
    ax.set_xlabel("")
    ax.set_ylabel("Cycle count per cell")
    ax.set_title("Cycle count distribution by dataset")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "cycle_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved cycle_distribution.png")


def fig_cell_count(df: pd.DataFrame):
    """Bar chart: number of cells per dataset, coloured by chemistry."""
    summary = (
        df.groupby(["dataset", "chemistry"])
        .size()
        .reset_index(name="n_cells")
    )
    pivot = summary.pivot_table(index="dataset", columns="chemistry", values="n_cells", fill_value=0)
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab10", width=0.6)
    ax.set_xlabel("")
    ax.set_ylabel("Number of cells")
    ax.set_title("Cells per dataset (coloured by chemistry)")
    ax.legend(title="Chemistry", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "cell_count.png", dpi=150)
    plt.close(fig)
    print("  Saved cell_count.png")


def fig_capacity_retention(df: pd.DataFrame):
    """Scatter of capacity retention vs cycle count, per dataset."""
    plot_df = df.dropna(subset=["capacity_retention", "n_cycles"])
    plot_df = plot_df[plot_df["capacity_retention"].between(0.4, 1.1)]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for ds, grp in plot_df.groupby("dataset"):
        ax.scatter(grp["n_cycles"], grp["capacity_retention"], label=ds, alpha=0.8, s=60)
    ax.axhline(0.8, color="red", linestyle="--", linewidth=1, label="80% EOL threshold")
    ax.set_xlabel("Cycle count")
    ax.set_ylabel("Capacity retention (final / initial)")
    ax.set_title("Capacity retention vs cycle count")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "capacity_retention.png", dpi=150)
    plt.close(fig)
    print("  Saved capacity_retention.png")


def fig_temp_crate_heatmap(df: pd.DataFrame):
    """Heatmap of cell count by temperature and discharge C-rate."""
    heat = (
        df.dropna(subset=["temperature_c", "discharge_rate_C"])
        .groupby(["temperature_c", "discharge_rate_C"])
        .size()
        .unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(heat, annot=True, fmt="d", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_xlabel("Discharge C-rate")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Cell count by temperature × discharge C-rate")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "temp_crate_heatmap.png", dpi=150)
    plt.close(fig)
    print("  Saved temp_crate_heatmap.png")


def fig_metadata_flags(df: pd.DataFrame):
    """Per-dataset flag grid: formation cycles, failure annotation, ≥500 cycles."""
    datasets = sorted(df["dataset"].unique())
    flags = {
        "Formation\ncycles": [],
        "Failure\nannotated": [],
        "Any cell\n≥500 cycles": [],
        "Any cell\n≥1000 cycles": [],
    }
    for ds in datasets:
        grp = df[df["dataset"] == ds]
        flags["Formation\ncycles"].append(grp["has_formation_cycles"].any())
        flags["Failure\nannotated"].append(grp["failure_mechanism"].notna().any())
        flags["Any cell\n≥500 cycles"].append((grp["n_cycles"] >= 500).any())
        flags["Any cell\n≥1000 cycles"].append((grp["n_cycles"] >= 1000).any())

    flag_df = pd.DataFrame(flags, index=datasets)
    fig, ax = plt.subplots(figsize=(8, 4))
    cmap = matplotlib.colors.ListedColormap(["#f7c5c5", "#b7e4b7"])
    im = ax.imshow(flag_df.values.astype(int), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(flag_df.columns)))
    ax.set_xticklabels(flag_df.columns, fontsize=9)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets)
    for i in range(len(datasets)):
        for j in range(len(flag_df.columns)):
            val = flag_df.values[i, j]
            ax.text(j, i, "Yes" if val else "No", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="#1a5e1a" if val else "#8b0000")
    ax.set_title("Dataset metadata flags")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "metadata_flags.png", dpi=150)
    plt.close(fig)
    print("  Saved metadata_flags.png")


def save_summary_table(summary: pd.DataFrame):
    """Save summary as a nicely formatted text table and CSV."""
    summary.to_csv(OUT_DIR / "dataset_summary.csv", index=False)

    col_widths = {c: max(len(c), summary[c].astype(str).str.len().max()) for c in summary.columns}
    header = "  ".join(c.ljust(col_widths[c]) for c in summary.columns)
    sep = "  ".join("-" * col_widths[c] for c in summary.columns)
    lines = [header, sep]
    for _, row in summary.iterrows():
        lines.append("  ".join(str(row[c]).ljust(col_widths[c]) for c in summary.columns))

    report_path = OUT_DIR / "dataset_summary.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Saved dataset_summary.txt and dataset_summary.csv")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading cycle data from all zips …")
    df = load_all_cycle_data()
    print(f"  Total cells loaded: {len(df)}\n")

    print("Building summary table …")
    summary = dataset_summary(df)
    save_summary_table(summary)

    print("\nSummary:\n")
    print(summary.to_string(index=False))

    print("\nGenerating figures …")
    fig_cycle_distribution(df)
    fig_cell_count(df)
    fig_capacity_retention(df)
    fig_temp_crate_heatmap(df)
    fig_metadata_flags(df)

    print(f"\nAll outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
