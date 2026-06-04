"""
Cross-model calibration comparison.

Usage
-----
    from calibration import compare_calibration
    from experiment import ExperimentResult

    results = {
        "Random Forest": rf_result,
        "LightGBM Quantile": lgbm_result,
        "Gaussian Process": gp_result,
        "Conformal(RF)": conf_result,
    }
    compare_calibration(results, out_dir=Path("results/comparison"))

Plots produced
--------------
  calibration_comparison.png  — overlaid reliability diagrams for all models
  coverage_sharpness.png      — coverage vs interval width scatter (Pareto frontier)
  winkler_scores.png          — Winkler score bar chart (lower = better)
  calibration_table.csv       — numeric summary of all metrics
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    from experiment import ExperimentResult

sns.set_theme(style="whitegrid", palette="tab10")

RESULTS_DIR = Path(__file__).parent / "results"


def _collect_metrics(
    results: dict[str, "ExperimentResult"]
) -> pd.DataFrame:
    """Build a metrics DataFrame with one row per model."""
    from experiment import Reporter

    rows = []
    for name, result in results.items():
        reporter = Reporter(result)
        m = reporter.calibration_metrics()
        rows.append({
            "model": name,
            "mae": round(result.mae, 1),
            "r2": round(result.r2, 3),
            "picp_80": round(m.get("picp_80", float("nan")), 3),
            "picp_50": round(m.get("picp_50", float("nan")), 3),
            "mpiw_80": round(m.get("mpiw_80", float("nan")), 1),
            "mpiw_50": round(m.get("mpiw_50", float("nan")), 1),
            "winkler_80": round(m.get("winkler_80", float("nan")), 1),
            "calibration_error": round(m.get("calibration_error", float("nan")), 4),
        })
    return pd.DataFrame(rows)


def compare_calibration(
    results: dict[str, "ExperimentResult"],
    out_dir: Path | None = None,
    run_tag: str = "comparison",
) -> pd.DataFrame:
    """
    Generate cross-model calibration comparison plots and return a metrics table.

    Parameters
    ----------
    results:
        Dict mapping model name → ExperimentResult.
    out_dir:
        Directory to write plots.  Defaults to results/comparison_<run_tag>/.
    run_tag:
        Short label appended to the output directory name.

    Returns
    -------
    DataFrame with one row per model and columns for all calibration metrics.
    """
    if out_dir is None:
        out_dir = RESULTS_DIR / f"calibration_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = _collect_metrics(results)
    metrics_df.to_csv(out_dir / "calibration_table.csv", index=False)

    palette = sns.color_palette("tab10", len(results))
    model_colors = dict(zip(results.keys(), palette))

    _plot_calibration_comparison(results, model_colors, out_dir)
    _plot_coverage_sharpness(metrics_df, model_colors, out_dir)
    _plot_winkler_scores(metrics_df, model_colors, out_dir)
    _plot_reliability_grid(results, model_colors, out_dir)

    print(f"Calibration comparison written to {out_dir}")
    print(metrics_df.to_string(index=False))
    return metrics_df


# ── Individual comparison plots ───────────────────────────────────────────────


def _plot_calibration_comparison(
    results: dict[str, "ExperimentResult"],
    model_colors: dict,
    out_dir: Path,
) -> None:
    """
    Overlaid reliability diagrams for all models on one axes.

    Each model is a coloured line; the diagonal is perfect calibration.
    The shaded band shows ±0.05 tolerance — models within this band are
    considered well-calibrated for practical purposes.
    """
    q_cols = {0.10: "q10", 0.25: "q25", 0.50: "q50", 0.75: "q75", 0.90: "q90"}
    nominal_levels = list(q_cols.keys())

    fig, ax = plt.subplots(figsize=(7, 7))

    # Reference line + tolerance band
    diag = np.array([0, 1])
    ax.plot(diag, diag, "k--", linewidth=1.2, label="Perfect calibration", zorder=2)
    ax.fill_between(diag, diag - 0.05, diag + 0.05, alpha=0.10, color="black",
                    label="±0.05 tolerance")

    for name, result in results.items():
        oof = result.oof_predictions.dropna(subset=["actual_eol"])
        y = oof["actual_eol"].values
        color = model_colors[name]

        nominal, observed = [], []
        for nom, col in q_cols.items():
            if col in oof.columns:
                nominal.append(nom)
                observed.append(float((y <= oof[col].values).mean()))

        if not nominal:
            continue

        cal_err = float(np.mean([abs(o - n) for n, o in zip(nominal, observed)]))
        ax.plot(nominal, observed, "o-", color=color, linewidth=2, markersize=7,
                label=f"{name}  (err={cal_err:.3f})", zorder=3)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Nominal quantile level", fontsize=11)
    ax.set_ylabel("Observed coverage", fontsize=11)
    ax.set_title("Reliability diagram — all models\n"
                 "Closer to diagonal = better calibrated", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    fig.savefig(out_dir / "calibration_comparison.png", dpi=150)
    plt.close(fig)


def _plot_coverage_sharpness(
    metrics_df: pd.DataFrame,
    model_colors: dict,
    out_dir: Path,
) -> None:
    """
    Coverage vs sharpness scatter (the calibration Pareto frontier).

    x-axis: actual 80% PI coverage (target = 0.80, dashed line)
    y-axis: mean 80% PI width in cycles (lower = sharper = better)

    Ideal model: at or above 0.80 coverage with minimum width.
    Models to the left of 0.80 are under-covering (overconfident).
    Models above have guaranteed coverage but with wider intervals.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.axvline(0.80, color="red", linestyle="--", linewidth=1,
               label="Target 80% coverage")
    ax.fill_betweenx([0, metrics_df["mpiw_80"].max() * 1.2],
                     0, 0.80, alpha=0.07, color="red", label="Under-covering region")

    for _, row in metrics_df.iterrows():
        color = model_colors.get(row["model"], "gray")
        ax.scatter(row["picp_80"], row["mpiw_80"], color=color,
                   s=120, zorder=4, edgecolors="white", linewidths=0.8)
        ax.annotate(
            row["model"],
            (row["picp_80"], row["mpiw_80"]),
            textcoords="offset points", xytext=(8, 4), fontsize=9,
        )

    ax.set_xlabel("Actual 80% PI coverage (higher = safer)", fontsize=11)
    ax.set_ylabel("Mean 80% PI width — cycles (lower = sharper)", fontsize=11)
    ax.set_title("Coverage vs sharpness\n"
                 "Best models: right of dashed line with minimum width", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "coverage_sharpness.png", dpi=150)
    plt.close(fig)


def _plot_winkler_scores(
    metrics_df: pd.DataFrame,
    model_colors: dict,
    out_dir: Path,
) -> None:
    """
    Winkler score bar chart for the 80% PI.

    The Winkler score combines interval width and coverage penalty:
        W = width + (2/α) × max(0, lo − y)  if y < lo
            width + (2/α) × max(0, y − hi)  if y > hi
            width                            otherwise

    Lower is strictly better.  A model with perfect coverage but width=0
    would score 0.  A model that misses every interval scores very high.
    """
    df = metrics_df.dropna(subset=["winkler_80"]).sort_values("winkler_80")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Winkler score
    colors = [model_colors.get(m, "gray") for m in df["model"]]
    axes[0].barh(df["model"], df["winkler_80"], color=colors, alpha=0.85,
                 edgecolor="white")
    axes[0].set_xlabel("Winkler score — 80% PI (lower = better)", fontsize=10)
    axes[0].set_title("Winkler score\n(combines coverage penalty + interval width)")
    for i, (_, row) in enumerate(df.iterrows()):
        axes[0].text(row["winkler_80"] * 1.01, i,
                     f"{row['winkler_80']:.0f}", va="center", fontsize=9)

    # Right: calibration error
    df2 = metrics_df.dropna(subset=["calibration_error"]).sort_values("calibration_error")
    colors2 = [model_colors.get(m, "gray") for m in df2["model"]]
    axes[1].barh(df2["model"], df2["calibration_error"], color=colors2,
                 alpha=0.85, edgecolor="white")
    axes[1].axvline(0.05, color="red", linestyle="--", linewidth=1,
                    label="0.05 threshold")
    axes[1].set_xlabel("Mean calibration error (lower = better)", fontsize=10)
    axes[1].set_title("Mean |observed − nominal| coverage\n"
                      "across q10/q25/q50/q75/q90")
    axes[1].legend(fontsize=8)
    for i, (_, row) in enumerate(df2.iterrows()):
        axes[1].text(row["calibration_error"] * 1.01, i,
                     f"{row['calibration_error']:.3f}", va="center", fontsize=9)

    plt.suptitle("Uncertainty quality metrics — all models", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / "winkler_scores.png", dpi=150)
    plt.close(fig)


def _plot_reliability_grid(
    results: dict[str, "ExperimentResult"],
    model_colors: dict,
    out_dir: Path,
) -> None:
    """
    Small-multiple reliability diagrams — one subplot per model.
    Useful for spotting per-model patterns without overlap.
    """
    n = len(results)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4.5),
                             sharex=True, sharey=True)
    axes = np.array(axes).flatten()

    q_cols = {0.10: "q10", 0.25: "q25", 0.50: "q50", 0.75: "q75", 0.90: "q90"}

    for ax, (name, result), color in zip(axes, results.items(), model_colors.values()):
        oof = result.oof_predictions.dropna(subset=["actual_eol"])
        y = oof["actual_eol"].values

        nominal, observed = [], []
        for nom, col in q_cols.items():
            if col in oof.columns:
                nominal.append(nom)
                observed.append(float((y <= oof[col].values).mean()))

        diag = [0, 1]
        ax.plot(diag, diag, "k--", linewidth=1, alpha=0.5)
        ax.fill_between(diag, [d - 0.05 for d in diag],
                        [d + 0.05 for d in diag], alpha=0.10, color="black")
        ax.plot(nominal, observed, "o-", color=color, linewidth=2, markersize=7)

        # Colour each point by direction of miscalibration
        for nom, obs in zip(nominal, observed):
            c = "green" if abs(obs - nom) <= 0.05 else ("blue" if obs > nom else "red")
            ax.scatter([nom], [obs], color=c, s=60, zorder=4)
            ax.annotate(f"{obs:.2f}", (nom, obs),
                        textcoords="offset points", xytext=(4, 3), fontsize=7)

        cal_err = np.mean([abs(o - n) for n, o in zip(nominal, observed)])
        ax.set_title(f"{name}\ncal error={cal_err:.3f}", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    for ax in axes[len(results):]:
        ax.set_visible(False)

    fig.supxlabel("Nominal quantile level", fontsize=10)
    fig.supylabel("Observed coverage", fontsize=10)
    fig.suptitle("Reliability diagrams — per model\n"
                 "Green = within ±0.05  Blue = conservative  Red = overconfident",
                 fontsize=10)
    plt.tight_layout()
    fig.savefig(out_dir / "reliability_grid.png", dpi=150)
    plt.close(fig)
