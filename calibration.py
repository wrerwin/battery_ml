"""
Calibration and ranking metrics for uncertainty-aware battery EOL models.

Two distinct questions answered here
-------------------------------------
1. CALIBRATION — "Are the stated probabilities trustworthy?"
   If the model says P(EOL > 500) = 0.7, does the truth land above 500
   roughly 70% of the time?  Measured by the reliability diagram and
   calibration error.

2. DISCRIMINATION / RANKING — "Can the model correctly sort batteries?"
   Even a miscalibrated model can reliably rank "good" batteries above "bad"
   ones.  For QA and research acceleration this is the primary goal.
   Measured by the concordance index (C-index), AUC at a threshold, and
   Spearman rank correlation.

Key metrics
-----------
C-index (concordance index)
    Fraction of all battery pairs where the model correctly ranks which one
    has the longer EOL.  0.5 = random guessing, 1.0 = perfect ranking.
    This is the primary metric for the "sort batteries after a few cycles"
    use case.  Does not require the predictions to be calibrated in absolute
    terms — only that relative ordering is correct.

AUC at threshold T
    Treat "will this battery last > T cycles?" as a binary classification
    problem.  AUC is the probability that a randomly chosen good battery
    scores higher than a randomly chosen bad battery.  Equivalent to the
    C-index restricted to a single threshold.

Spearman rank correlation
    Monotonic correlation between predicted ranking score and actual EOL.
    Directly measures whether the model produces a useful sort order.

Ranking score used for AUC / Spearman
    P(EOL > T) averaged across the configured thresholds.  This is the
    integral approach described below: instead of committing to one threshold,
    integrate the survival function across several, giving a robust summary
    score.  Equivalent to the area under the model's survival curve evaluated
    at the chosen threshold points.

Calibration metrics
-------------------
picp_80   Fraction of actuals inside the 80% PI.  Target: 0.80.
mpiw_80   Mean width of 80% PI in cycles.  Lower = sharper.
winkler   Interval width + out-of-bounds penalty.  Lower = better.
cal_err   Mean |observed − nominal| across q10/q25/q50/q75/q90.

Usage
-----
    from calibration import compare_calibration

    results = {
        "Random Forest":     rf_result,
        "LightGBM Quantile": lgbm_result,
        "Gaussian Process":  gp_result,
        "Conformal(RF)":     conf_result,
    }
    compare_calibration(results, thresholds=[500, 1000, 1500])

Plots produced
--------------
  calibration_comparison.png  — overlaid reliability diagrams
  coverage_sharpness.png      — coverage vs width Pareto scatter
  winkler_scores.png          — Winkler + calibration error bars
  reliability_grid.png        — small-multiple per-model reliability diagrams
  ranking_summary.png         — C-index, AUC, Spearman across models
  ranking_scatter.png         — predicted score vs actual EOL, all models
  calibration_table.csv       — all metrics in one table
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


# ── Ranking / discrimination metrics ─────────────────────────────────────────


def concordance_index(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Concordance index (C-index): fraction of battery pairs where the model
    correctly ranks which one has the longer EOL.

    For every ordered pair (i, j) where y_true[i] != y_true[j]:
        concordant   if sign(y_score[i] - y_score[j]) == sign(y_true[i] - y_true[j])
        discordant   otherwise
        tied score   if y_score[i] == y_score[j]  (counts as 0.5)

    C-index = (concordant + 0.5 * tied) / (concordant + discordant + tied)

    0.5 = random; 1.0 = perfect.
    """
    n = len(y_true)
    concordant = discordant = tied = 0
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue
            delta_true = y_true[i] - y_true[j]
            delta_score = y_score[i] - y_score[j]
            if delta_score * delta_true > 0:
                concordant += 1
            elif delta_score * delta_true < 0:
                discordant += 1
            else:
                tied += 1
    total = concordant + discordant + tied
    return (concordant + 0.5 * tied) / total if total > 0 else float("nan")


def ranking_metrics(
    oof: pd.DataFrame,
    thresholds: list[int],
) -> dict:
    """
    Compute discrimination / ranking metrics from OOF predictions.

    The ranking score for each battery is P(EOL > T) averaged across all
    configured thresholds.  This is the "integral over thresholds" approach:
    instead of committing to a single cutoff, we combine evidence across
    several, making the score more robust.

    Parameters
    ----------
    oof : OOF predictions DataFrame (output of ExperimentResult.oof_predictions)
    thresholds : cycle counts to evaluate  e.g. [500, 1000, 1500]

    Returns
    -------
    dict with c_index, spearman_r, spearman_p, and auc_at_<T> for each threshold
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    oof = oof.dropna(subset=["actual_eol"])
    y_true = oof["actual_eol"].values

    # Ranking score: mean P(EOL > T) across thresholds
    p_cols = [f"p_eol_gt_{t}" for t in thresholds if f"p_eol_gt_{t}" in oof.columns]
    if not p_cols:
        return {}

    score = oof[p_cols].mean(axis=1).values

    metrics = {}

    # C-index
    metrics["c_index"] = round(concordance_index(y_true, score), 4)

    # Spearman rank correlation between score and actual EOL
    r, p = spearmanr(score, y_true)
    metrics["spearman_r"] = round(float(r), 4)
    metrics["spearman_p"] = round(float(p), 4)

    # AUC at each threshold: binary "will it last > T?" classification
    for t in thresholds:
        col = f"p_eol_gt_{t}"
        if col not in oof.columns:
            continue
        y_bin = (y_true > t).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            metrics[f"auc_gt_{t}"] = float("nan")
        else:
            try:
                metrics[f"auc_gt_{t}"] = round(
                    float(roc_auc_score(y_bin, oof[col].values)), 4
                )
            except Exception:
                metrics[f"auc_gt_{t}"] = float("nan")

    return metrics


def _collect_metrics(
    results: dict[str, "ExperimentResult"],
    thresholds: list[int],
) -> pd.DataFrame:
    """Build a metrics DataFrame with one row per model."""
    from experiment import Reporter

    rows = []
    for name, result in results.items():
        reporter = Reporter(result)
        m = reporter.calibration_metrics()
        rm = ranking_metrics(result.oof_predictions, thresholds)
        rows.append({
            "model": name,
            "mae": round(result.mae, 1),
            "r2": round(result.r2, 3),
            # Ranking / discrimination
            "c_index": rm.get("c_index", float("nan")),
            "spearman_r": rm.get("spearman_r", float("nan")),
            **{k: v for k, v in rm.items() if k.startswith("auc_")},
            # Calibration
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
    thresholds: list[int] | None = None,
) -> pd.DataFrame:
    """
    Generate cross-model calibration and ranking comparison plots.

    Parameters
    ----------
    results:
        Dict mapping model name → ExperimentResult.
    out_dir:
        Directory to write plots.  Defaults to results/calibration_<run_tag>/.
    run_tag:
        Short label appended to the output directory name.
    thresholds:
        Cycle thresholds for ranking score and AUC computation.
        Defaults to thresholds from the first result's config.

    Returns
    -------
    DataFrame with one row per model, columns for all calibration and ranking
    metrics.
    """
    if thresholds is None:
        first = next(iter(results.values()))
        thresholds = first.config.thresholds

    if out_dir is None:
        out_dir = RESULTS_DIR / f"calibration_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = _collect_metrics(results, thresholds)
    metrics_df.to_csv(out_dir / "calibration_table.csv", index=False)

    palette = sns.color_palette("tab10", len(results))
    model_colors = dict(zip(results.keys(), palette))

    # Calibration plots
    _plot_calibration_comparison(results, model_colors, out_dir)
    _plot_coverage_sharpness(metrics_df, model_colors, out_dir)
    _plot_winkler_scores(metrics_df, model_colors, out_dir)
    _plot_reliability_grid(results, model_colors, out_dir)

    # Ranking / discrimination plots
    _plot_ranking_summary(metrics_df, model_colors, thresholds, out_dir)
    _plot_ranking_scatter(results, model_colors, thresholds, out_dir)

    print(f"Comparison written to {out_dir}")
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


def _plot_ranking_summary(
    metrics_df: pd.DataFrame,
    model_colors: dict,
    thresholds: list[int],
    out_dir: Path,
) -> None:
    """
    Bar chart comparison of ranking / discrimination metrics across models.

    Three panels:
      Left:   C-index — overall sorting ability (0.5 = random, 1.0 = perfect)
      Centre: Spearman rank correlation — monotonic agreement with actual EOL
      Right:  AUC at each threshold — binary classification performance

    These are the metrics that directly answer "can this model correctly sort
    batteries from best to worst after a few cycles?"
    """
    auc_cols = [c for c in metrics_df.columns if c.startswith("auc_gt_")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # ── C-index ──────────────────────────────────────────────────────────────
    ax = axes[0]
    df_c = metrics_df.dropna(subset=["c_index"]).sort_values("c_index", ascending=False)
    colors = [model_colors.get(m, "gray") for m in df_c["model"]]
    bars = ax.barh(df_c["model"], df_c["c_index"], color=colors, alpha=0.85,
                   edgecolor="white")
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="Random (0.5)")
    ax.axvline(1.0, color="green", linestyle="--", linewidth=1, alpha=0.4,
               label="Perfect (1.0)")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("C-index", fontsize=10)
    ax.set_title("Concordance index\n(fraction of pairs correctly ranked)", fontsize=9)
    ax.legend(fontsize=8)
    for bar, (_, row) in zip(bars, df_c.iterrows()):
        ax.text(row["c_index"] + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{row['c_index']:.3f}", va="center", fontsize=9)

    # ── Spearman r ───────────────────────────────────────────────────────────
    ax = axes[1]
    df_s = metrics_df.dropna(subset=["spearman_r"]).sort_values("spearman_r",
                                                                  ascending=False)
    colors = [model_colors.get(m, "gray") for m in df_s["model"]]
    bars = ax.barh(df_s["model"], df_s["spearman_r"], color=colors, alpha=0.85,
                   edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1, label="No correlation")
    ax.set_xlim(-0.1, 1.05)
    ax.set_xlabel("Spearman ρ", fontsize=10)
    ax.set_title("Spearman rank correlation\n(predicted rank vs actual EOL)", fontsize=9)
    ax.legend(fontsize=8)
    for bar, (_, row) in zip(bars, df_s.iterrows()):
        ax.text(max(row["spearman_r"] + 0.01, 0.01),
                bar.get_y() + bar.get_height() / 2,
                f"{row['spearman_r']:.3f}", va="center", fontsize=9)

    # ── AUC at thresholds ────────────────────────────────────────────────────
    ax = axes[2]
    if auc_cols:
        auc_df = metrics_df[["model"] + auc_cols].set_index("model")
        auc_df.columns = [c.replace("auc_gt_", ">") + " cyc" for c in auc_df.columns]
        x = np.arange(len(auc_df))
        width = 0.8 / len(auc_df.columns)
        for i, col in enumerate(auc_df.columns):
            vals = auc_df[col].values
            offset = (i - len(auc_df.columns) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width=width * 0.9, alpha=0.8, label=col,
                   edgecolor="white")
        ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Random (0.5)")
        ax.set_xticks(x)
        ax.set_xticklabels(auc_df.index, rotation=15, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("AUC")
        ax.set_title("AUC: P(ranks good > bad)\nat each cycle threshold", fontsize=9)
        ax.legend(fontsize=8, loc="lower right")
    else:
        ax.text(0.5, 0.5, "No P(EOL>T) columns found", ha="center", va="center")

    plt.suptitle(
        "Ranking / discrimination metrics\n"
        "Primary metrics for QA sorting: how well does the model order batteries?",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(out_dir / "ranking_summary.png", dpi=150)
    plt.close(fig)


def _plot_ranking_scatter(
    results: dict[str, "ExperimentResult"],
    model_colors: dict,
    thresholds: list[int],
    out_dir: Path,
) -> None:
    """
    Ranking score vs actual EOL for each model.

    The ranking score (x-axis) is the mean P(EOL > T) across all thresholds —
    the integral approach.  Each point is a battery.  A model with perfect
    discrimination would show a clean monotonic curve: higher score → longer
    actual EOL.  Points are coloured by dataset so cross-dataset patterns are
    visible.

    The shaded background shows the expected shape: if the model is well-
    calibrated AND well-discriminating, the 45° band should be densely populated.
    """
    n = len(results)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5),
                             sharex=False, sharey=False)
    axes = np.array(axes).flatten()

    ds_palette = sns.color_palette("Set2", 10)

    for ax, (name, result), model_color in zip(axes, results.items(),
                                                model_colors.values()):
        oof = result.oof_predictions.dropna(subset=["actual_eol"])
        p_cols = [f"p_eol_gt_{t}" for t in thresholds if f"p_eol_gt_{t}" in oof.columns]
        if not p_cols:
            ax.set_visible(False)
            continue

        score = oof[p_cols].mean(axis=1).values
        actual = oof["actual_eol"].values
        datasets = oof["dataset"].values

        unique_ds = sorted(set(datasets))
        ds_color_map = dict(zip(unique_ds, ds_palette))

        for ds in unique_ds:
            mask = datasets == ds
            ax.scatter(score[mask], actual[mask],
                       color=ds_color_map[ds], s=55, alpha=0.85,
                       edgecolors="white", linewidths=0.5, label=ds, zorder=3)

        # Spearman annotation
        from scipy.stats import spearmanr
        r, p = spearmanr(score, actual)
        c_idx = concordance_index(actual, score)

        # Trend line
        if len(score) > 2:
            z = np.polyfit(score, actual, 1)
            x_line = np.linspace(score.min(), score.max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), "--", color=model_color,
                    linewidth=1.5, alpha=0.6)

        ax.set_xlabel("Ranking score  [mean P(EOL > T)]", fontsize=9)
        ax.set_ylabel("Actual EOL cycle", fontsize=9)
        ax.set_title(
            f"{name}\nSpearman ρ={r:.3f}   C-index={c_idx:.3f}",
            fontsize=9,
        )
        if len(unique_ds) > 1:
            ax.legend(fontsize=7, loc="upper left")

    for ax in axes[len(results):]:
        ax.set_visible(False)

    fig.suptitle(
        "Ranking score vs actual EOL — per model\n"
        "Higher ranking score should predict longer actual EOL",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(out_dir / "ranking_scatter.png", dpi=150)
    plt.close(fig)
