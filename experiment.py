"""
Experimental framework: cross-validation, model interface, and reporting.

Architecture
------------
ExperimentConfig
    All hyperparameters and experiment settings in one place.

ModelInterface  (ABC)
    Common contract for all models.  Concrete implementations only need to
    define fit() and predict_distribution().  Everything else (point estimate,
    P(EOL > x), survival curve) is derived automatically.

RandomForestModel
    First concrete implementation.  Uses the spread of individual tree
    predictions as the predictive distribution — no distributional assumption
    required.  Plug in any sklearn-compatible regressor by subclassing
    ModelInterface.

Experiment
    Orchestrates the full pipeline:
        datasets → features (cached via FeatureStore) → CV splits →
        fit/predict per fold → collect OOF predictions → aggregate metrics.

    Two CV strategies:
      group_kfold           — K folds, whole batteries as groups.  Never
                              splits a single battery across train/test.
                              Use for within-dataset experiments.
      leave_one_dataset_out — each fold holds out one entire dataset.
                              Trains on all others.  Strictest test of
                              cross-dataset generalisation.

Reporter
    Produces figures and a JSON summary from an ExperimentResult.
    All figures written to results/<run_id>/.

Leakage notes
-------------
- StandardScaler and SimpleImputer are fit inside each fold on train only.
- eol_cycle (the label) is separated before any feature computation.
- Group structure (cell_id) ensures no battery contributes to both train
  and test within a fold.
- For LODO, cross-dataset normalisation leakage is impossible by construction.
"""

from __future__ import annotations

import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from featurizer import BatteryFeaturizer, CycleFeaturizer, FeatureStore
from loader import load_all
from schema import Battery, Dataset

RESULTS_DIR = Path(__file__).parent / "results"

# Columns that carry metadata but are not ML features
_META_COLS = {
    "cell_id", "dataset", "chemistry", "form_factor",
    "temperature_c", "charge_rate_c", "discharge_rate_c",
    "n_cycles_observed", "max_cycle_used", "formation_only",
    "eol_cycle",
}


# ── ExperimentConfig ──────────────────────────────────────────────────────────


@dataclass
class ExperimentConfig:
    """
    All settings for a single experiment run.

    Parameters
    ----------
    max_cycle:
        Features are computed using only cycles with index <= max_cycle.
        This is the hard leakage boundary.
    formation_only:
        If True, only formation-flagged cycles contribute to features.
        Simulates the manufacturing context (predict EOL from formation data).
    cv_strategy:
        "group_kfold"           — K folds, whole batteries as atomic groups.
        "leave_one_dataset_out" — each fold = one held-out dataset.
    n_folds:
        Number of folds for group_kfold.  Ignored for LODO.
    eol_threshold:
        Capacity retention threshold that defines end-of-life (default 0.8).
        Informational only — the actual EOL label comes from Battery.end_of_life_cycle
        which already bakes this in.
    feature_tags:
        If set, only features whose tags overlap this list are used.
        None = use all registered features.
        Use ["capacity", "efficiency", "energy"] for a fast no-timeseries run.
    thresholds:
        Cycle counts for which P(EOL > threshold) is reported per battery.
    min_cycles_observed:
        Batteries with fewer observed cycles than this (within the max_cycle
        window) are dropped.  Prevents near-empty feature vectors.
    random_seed:
        Controls RF randomness and fold shuffling.
    run_name:
        Human-readable label for this experiment.  Auto-generated if None.
    """

    max_cycle: int = 100
    formation_only: bool = False
    cv_strategy: Literal["group_kfold", "leave_one_dataset_out"] = "group_kfold"
    n_folds: int = 5
    eol_threshold: float = 0.8
    feature_tags: Optional[list[str]] = None
    thresholds: list[int] = field(default_factory=lambda: [500, 1000, 2000])
    min_cycles_observed: int = 5
    random_seed: int = 42
    run_name: Optional[str] = None

    def __post_init__(self):
        if self.run_name is None:
            strat = "lodo" if self.cv_strategy == "leave_one_dataset_out" else f"{self.n_folds}fold"
            form = "_form" if self.formation_only else ""
            self.run_name = f"mc{self.max_cycle}{form}_{strat}"


# ── ModelInterface ────────────────────────────────────────────────────────────


class ModelInterface(ABC):
    """
    Abstract base class for all predictive models.

    Concrete subclasses must implement fit() and predict_distribution().
    Everything else is derived.

    predict_distribution() returns a list of 1-D arrays — one per sample.
    Each array represents the model's belief over EOL cycle count for that
    battery.  The distribution need not follow any parametric form; it is
    used empirically to compute quantiles and exceedance probabilities.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model on feature matrix X and labels y."""
        ...

    @abstractmethod
    def predict_distribution(self, X: pd.DataFrame) -> list[np.ndarray]:
        """
        Return the predictive distribution for each row of X.

        Returns
        -------
        list of length n_samples, each element a 1-D ndarray of predicted
        EOL values sampled from the model's distribution.
        """
        ...

    def predict_point(self, X: pd.DataFrame) -> np.ndarray:
        """Median of predictive distribution per sample."""
        return np.array([np.median(d) for d in self.predict_distribution(X)])

    def predict_quantiles(
        self, X: pd.DataFrame, quantiles: list[float] = (0.1, 0.5, 0.9)
    ) -> pd.DataFrame:
        """
        Return a DataFrame of shape (n_samples, len(quantiles)).
        Column names are q10, q50, q90, etc.
        """
        dists = self.predict_distribution(X)
        rows = {
            f"q{int(q * 100)}": [np.quantile(d, q) for d in dists]
            for q in quantiles
        }
        return pd.DataFrame(rows)

    def predict_proba_exceeds(self, X: pd.DataFrame, threshold: float) -> np.ndarray:
        """
        P(EOL > threshold) for each sample.

        Parameters
        ----------
        threshold:
            Cycle count threshold.  Returns the fraction of the predictive
            distribution that exceeds this value.
        """
        return np.array([
            float(np.mean(d > threshold))
            for d in self.predict_distribution(X)
        ])

    def survival_curve(
        self, X: pd.DataFrame, cycle_range: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        P(EOL > x) evaluated over a range of cycle thresholds.

        Returns a DataFrame with columns [cycle, p_exceeds_0, p_exceeds_1, ...]
        where each p_exceeds_i corresponds to one row of X.
        """
        if cycle_range is None:
            cycle_range = np.arange(0, 5001, 50)
        dists = self.predict_distribution(X)
        rows = {"cycle": cycle_range}
        for i, dist in enumerate(dists):
            rows[f"sample_{i}"] = [float(np.mean(dist > c)) for c in cycle_range]
        return pd.DataFrame(rows)


# ── RandomForestModel ─────────────────────────────────────────────────────────


class RandomForestModel(ModelInterface):
    """
    Random forest regressor with tree-ensemble uncertainty.

    Predictive distribution: predictions from all individual trees in the
    ensemble.  For a sample x, the distribution is the vector
        [tree_0.predict(x), tree_1.predict(x), ..., tree_n.predict(x)]
    This gives a non-parametric empirical distribution at zero extra cost
    — no quantile regression or conformal calibration needed.

    The pipeline includes a median imputer (for NaN features from failed
    timeseries extraction) and a standard scaler (no-op for trees but
    future-proofs the interface for linear/neural models).
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_features: float = 0.33,
        min_samples_leaf: int = 2,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self._rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )
        self._pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("rf", self._rf),
        ])
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._feature_names = list(X.columns)
        self._pipeline.fit(X, y)

    def predict_distribution(self, X: pd.DataFrame) -> list[np.ndarray]:
        # Transform through imputer + scaler only (not the final estimator)
        X_transformed = self._pipeline[:-1].transform(X)
        rf = self._pipeline.named_steps["rf"]
        # Collect predictions from every individual tree
        tree_preds = np.array([
            tree.predict(X_transformed) for tree in rf.estimators_
        ])
        # tree_preds shape: (n_trees, n_samples)
        # Transpose so we get one distribution per sample
        return [tree_preds[:, i] for i in range(X_transformed.shape[0])]

    def feature_importances(self) -> pd.Series:
        rf = self._pipeline.named_steps["rf"]
        return pd.Series(
            rf.feature_importances_, index=self._feature_names
        ).sort_values(ascending=False)


# ── Results containers ────────────────────────────────────────────────────────


@dataclass
class FoldResult:
    """Metrics and predictions from a single CV fold."""

    fold_id: int | str          # integer for group_kfold, dataset name for LODO
    n_train: int
    n_test: int
    mae: float
    rmse: float
    r2: float

    # Per-battery prediction details for this fold
    predictions: pd.DataFrame   # cell_id | actual | pred_median | q10 | q90 | p_exceeds_* | dataset


@dataclass
class ExperimentResult:
    """Aggregated results from a full cross-validation run."""

    config: ExperimentConfig
    run_id: str
    fold_results: list[FoldResult]
    feature_importances: Optional[pd.Series]  # averaged across folds
    oof_predictions: pd.DataFrame             # out-of-fold predictions, all batteries

    @property
    def mae(self) -> float:
        return float(np.mean([f.mae for f in self.fold_results]))

    @property
    def rmse(self) -> float:
        return float(np.mean([f.rmse for f in self.fold_results]))

    @property
    def r2(self) -> float:
        return float(np.mean([f.r2 for f in self.fold_results]))

    def metrics_df(self) -> pd.DataFrame:
        rows = [
            {
                "fold": f.fold_id,
                "n_train": f.n_train,
                "n_test": f.n_test,
                "mae": round(f.mae, 1),
                "rmse": round(f.rmse, 1),
                "r2": round(f.r2, 3),
            }
            for f in self.fold_results
        ]
        rows.append({
            "fold": "mean",
            "n_train": int(np.mean([f.n_train for f in self.fold_results])),
            "n_test": int(np.mean([f.n_test for f in self.fold_results])),
            "mae": round(self.mae, 1),
            "rmse": round(self.rmse, 1),
            "r2": round(self.r2, 3),
        })
        return pd.DataFrame(rows)


# ── Experiment ────────────────────────────────────────────────────────────────


class Experiment:
    """
    Orchestrates the full train/evaluate pipeline.

    Usage
    -----
        config = ExperimentConfig(max_cycle=100, cv_strategy="group_kfold")
        model = RandomForestModel()
        exp = Experiment(config, model)

        datasets = load_all()
        result = exp.run(datasets)
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model: ModelInterface,
        feature_store: Optional[FeatureStore] = None,
    ):
        self.config = config
        self.model = model
        self.store = feature_store or FeatureStore()

    # ── Feature preparation ───────────────────────────────────────────────────

    def _get_features(self, dataset: Dataset) -> pd.DataFrame:
        """Load battery features from cache, computing them if necessary."""
        cfg = self.config
        if self.store.has_battery_features(dataset.name, cfg.max_cycle, cfg.formation_only):
            return self.store.load_battery_features(
                dataset.name, cfg.max_cycle, cfg.formation_only
            )

        print(f"    Computing features for {dataset.name}…")
        cf = CycleFeaturizer(tags=cfg.feature_tags)
        bf = BatteryFeaturizer(cf)
        df = bf.compute_dataset(
            dataset.batteries,
            max_cycle=cfg.max_cycle,
            formation_only=cfg.formation_only,
            include_label=True,
            verbose=False,
        )
        self.store.save_battery_features(df, dataset.name, cfg.max_cycle, cfg.formation_only)
        return df

    def _build_design_matrix(
        self, datasets: list[Dataset]
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Assemble feature matrix X, labels y, groups, and dataset labels.

        Drops batteries that:
          - have no EOL label (cell survived beyond recorded data)
          - have fewer than min_cycles_observed cycles in the feature window

        Returns
        -------
        X        : feature DataFrame (no metadata columns, no label)
        y        : EOL cycle count (regression target)
        groups   : cell_id Series — used to keep batteries intact in CV
        datasets : dataset name per row — used for LODO splits
        """
        dfs = []
        for ds in datasets:
            df = self._get_features(ds)
            dfs.append(df)

        full = pd.concat(dfs, ignore_index=True)

        # Drop unlabeled batteries
        n_before = len(full)
        full = full.dropna(subset=["eol_cycle"])
        n_dropped_label = n_before - len(full)
        if n_dropped_label:
            print(f"  Dropped {n_dropped_label} batteries with no EOL label.")

        # Drop batteries with too few observed cycles
        full = full[full["n_cycles_observed"] >= self.config.min_cycles_observed]

        feature_cols = [c for c in full.columns if c not in _META_COLS]
        X = full[feature_cols].copy()
        y = full["eol_cycle"].astype(float)
        groups = full["cell_id"]
        dataset_col = full["dataset"]

        print(f"  Design matrix: {X.shape[0]} batteries × {X.shape[1]} features")
        print(f"  EOL range: {y.min():.0f} – {y.max():.0f} cycles")

        return X, y, groups, dataset_col

    # ── CV splits ─────────────────────────────────────────────────────────────

    def _get_splits(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        dataset_col: pd.Series,
    ):
        """
        Yield (fold_id, train_idx, test_idx) tuples.

        group_kfold:
            Standard GroupKFold.  Each battery appears in exactly one test fold.
            Groups ensure no battery is split across train/test.

        leave_one_dataset_out:
            Each fold holds out one complete dataset.  Tests generalisation
            across different experimental conditions, labs, and chemistries.
            This is the hardest and most honest evaluation.
        """
        cfg = self.config

        if cfg.cv_strategy == "group_kfold":
            gkf = GroupKFold(n_splits=cfg.n_folds)
            for fold_id, (train_idx, test_idx) in enumerate(
                gkf.split(X, y, groups=groups)
            ):
                yield fold_id, train_idx, test_idx

        elif cfg.cv_strategy == "leave_one_dataset_out":
            unique_datasets = sorted(dataset_col.unique())
            if len(unique_datasets) < 2:
                raise ValueError(
                    "leave_one_dataset_out requires at least 2 datasets. "
                    "Use group_kfold for single-dataset experiments."
                )
            for ds_name in unique_datasets:
                test_mask = dataset_col == ds_name
                test_idx = np.where(test_mask)[0]
                train_idx = np.where(~test_mask)[0]
                yield ds_name, train_idx, test_idx

        else:
            raise ValueError(f"Unknown cv_strategy: {cfg.cv_strategy!r}")

    # ── Fold evaluation ───────────────────────────────────────────────────────

    def _evaluate_fold(
        self,
        fold_id: int | str,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        dataset_col: pd.Series,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> FoldResult:

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        self.model.fit(X_train, y_train)

        dists = self.model.predict_distribution(X_test)
        quantiles_df = self.model.predict_quantiles(X_test, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9])

        y_pred = quantiles_df["q50"].values

        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))

        pred_df = pd.DataFrame({
            "cell_id": groups.iloc[test_idx].values,
            "dataset": dataset_col.iloc[test_idx].values,
            "actual_eol": y_test.values,
            "pred_median": y_pred,
        })
        pred_df = pd.concat([pred_df, quantiles_df.reset_index(drop=True)], axis=1)

        # P(EOL > threshold) for each configured threshold
        for t in self.config.thresholds:
            proba = self.model.predict_proba_exceeds(X_test, threshold=t)
            pred_df[f"p_eol_gt_{t}"] = proba

        return FoldResult(
            fold_id=fold_id,
            n_train=len(train_idx),
            n_test=len(test_idx),
            mae=mae,
            rmse=rmse,
            r2=r2,
            predictions=pred_df,
        )

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, datasets: list[Dataset]) -> ExperimentResult:
        """
        Execute the full cross-validation experiment.

        Parameters
        ----------
        datasets:
            List of Dataset objects.  Features are loaded from cache if
            available, computed and cached otherwise.

        Returns
        -------
        ExperimentResult containing per-fold metrics, OOF predictions, and
        averaged feature importances.
        """
        run_id = str(uuid.uuid4())[:8]
        print(f"\n{'='*60}")
        print(f"Experiment: {self.config.run_name}  [{run_id}]")
        print(f"  strategy : {self.config.cv_strategy}")
        print(f"  max_cycle: {self.config.max_cycle}")
        print(f"  formation: {self.config.formation_only}")
        print(f"{'='*60}\n")

        t0 = time.time()

        print("Building design matrix…")
        X, y, groups, dataset_col = self._build_design_matrix(datasets)

        fold_results: list[FoldResult] = []
        importances: list[pd.Series] = []

        for fold_id, train_idx, test_idx in self._get_splits(X, y, groups, dataset_col):
            print(f"  Fold {fold_id}: train={len(train_idx)}, test={len(test_idx)}")
            result = self._evaluate_fold(
                fold_id, X, y, groups, dataset_col, train_idx, test_idx
            )
            fold_results.append(result)
            print(f"    MAE={result.mae:.1f}  RMSE={result.rmse:.1f}  R²={result.r2:.3f}")

            if hasattr(self.model, "feature_importances"):
                importances.append(self.model.feature_importances())

        oof = pd.concat([f.predictions for f in fold_results], ignore_index=True)

        avg_importances = None
        if importances:
            avg_importances = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)

        elapsed = time.time() - t0
        print(f"\nDone in {elapsed:.1f}s")

        result = ExperimentResult(
            config=self.config,
            run_id=run_id,
            fold_results=fold_results,
            feature_importances=avg_importances,
            oof_predictions=oof,
        )

        print("\nOverall metrics:")
        print(result.metrics_df().to_string(index=False))

        return result


# ── Reporter ──────────────────────────────────────────────────────────────────


class Reporter:
    """
    Generates figures and a JSON summary from an ExperimentResult.
    All outputs written to results/<run_id>/.
    """

    def __init__(self, result: ExperimentResult):
        self.result = result
        self.out_dir = RESULTS_DIR / f"{result.config.run_name}_{result.run_id}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Import here so the module is importable without matplotlib
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        self._plt = plt
        self._sns = sns
        sns.set_theme(style="whitegrid", palette="tab10")

    def save_all(self) -> None:
        """Generate and save all standard reports."""
        print(f"\nWriting reports to {self.out_dir}")
        self.plot_actual_vs_predicted()
        self.plot_fold_metrics()
        self.plot_feature_importances()
        self.plot_eol_distributions()
        self.plot_survival_curves()
        self.save_summary_json()
        self.save_predictions_csv()
        print("Done.")

    # ── Individual plots ──────────────────────────────────────────────────────

    def plot_actual_vs_predicted(self) -> None:
        plt = self._plt
        oof = self.result.oof_predictions

        fig, ax = plt.subplots(figsize=(8, 8))
        datasets = oof["dataset"].unique()
        palette = self._sns.color_palette("tab10", len(datasets))

        for ds, color in zip(datasets, palette):
            mask = oof["dataset"] == ds
            sub = oof[mask]
            ax.errorbar(
                sub["actual_eol"], sub["pred_median"],
                yerr=[
                    sub["pred_median"] - sub["q10"],
                    sub["q90"] - sub["pred_median"],
                ],
                fmt="o", color=color, label=ds, alpha=0.7, markersize=6,
                elinewidth=1, capsize=3,
            )

        lo = min(oof["actual_eol"].min(), oof["pred_median"].min()) * 0.95
        hi = max(oof["actual_eol"].max(), oof["pred_median"].max()) * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="Perfect prediction")

        ax.set_xlabel("Actual EOL cycle")
        ax.set_ylabel("Predicted EOL cycle (median)")
        ax.set_title(
            f"Actual vs Predicted EOL  |  MAE={self.result.mae:.0f}  "
            f"R²={self.result.r2:.3f}\n"
            f"Error bars: 10th–90th percentile of predictive distribution"
        )
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
        plt.tight_layout()
        fig.savefig(self.out_dir / "actual_vs_predicted.png", dpi=150)
        plt.close(fig)

    def plot_fold_metrics(self) -> None:
        plt = self._plt
        metrics = self.result.metrics_df()
        metrics = metrics[metrics["fold"] != "mean"]

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, metric in zip(axes, ["mae", "rmse", "r2"]):
            ax.bar(metrics["fold"].astype(str), metrics[metric], color="steelblue", alpha=0.8)
            mean_val = self.result.metrics_df().set_index("fold").loc["mean", metric]
            ax.axhline(mean_val, color="red", linestyle="--", linewidth=1.2, label=f"Mean: {mean_val:.2f}")
            ax.set_xlabel("Fold")
            ax.set_ylabel(metric.upper())
            ax.set_title(f"{metric.upper()} per fold")
            ax.legend(fontsize=8)

        plt.suptitle(f"{self.result.config.run_name}", fontsize=11)
        plt.tight_layout()
        fig.savefig(self.out_dir / "fold_metrics.png", dpi=150)
        plt.close(fig)

    def plot_feature_importances(self, top_n: int = 30) -> None:
        if self.result.feature_importances is None:
            return
        plt = self._plt

        imps = self.result.feature_importances.head(top_n)
        fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.28)))
        imps[::-1].plot(kind="barh", ax=ax, color="steelblue", alpha=0.85)
        ax.set_xlabel("Mean importance (across folds)")
        ax.set_title(f"Top {top_n} feature importances")
        plt.tight_layout()
        fig.savefig(self.out_dir / "feature_importances.png", dpi=150)
        plt.close(fig)

    def plot_eol_distributions(self, n_samples: int = 12) -> None:
        """
        Histogram of the predictive EOL distribution for a sample of batteries.
        Shows the actual EOL as a vertical line.
        """
        plt = self._plt
        oof = self.result.oof_predictions.dropna(subset=["actual_eol"])

        # Pick a spread of batteries (shortest, median, longest actual EOL)
        oof_sorted = oof.sort_values("actual_eol")
        indices = np.linspace(0, len(oof_sorted) - 1, min(n_samples, len(oof_sorted)), dtype=int)
        sample = oof_sorted.iloc[indices]

        ncols = 4
        nrows = int(np.ceil(len(sample) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
        axes = axes.flatten()

        # We need the raw distributions — re-run predict on each battery.
        # They're not stored in the result to keep memory manageable.
        # Instead, use the quantile columns we do have to sketch a distribution.
        for ax, (_, row) in zip(axes, sample.iterrows()):
            q_cols = [c for c in row.index if c.startswith("q") and c[1:].isdigit()]
            q_vals = [float(c[1:]) / 100 for c in q_cols]
            eol_vals = [row[c] for c in q_cols]

            ax.barh(
                [f"q{int(q*100)}" for q in q_vals],
                eol_vals,
                color="steelblue", alpha=0.7,
            )
            ax.axvline(row["actual_eol"], color="red", linestyle="--", linewidth=1.5,
                       label=f"Actual: {int(row['actual_eol'])}")
            ax.set_title(f"{row['cell_id'][:20]}", fontsize=7)
            ax.set_xlabel("EOL cycle")
            ax.legend(fontsize=7)

        for ax in axes[len(sample):]:
            ax.set_visible(False)

        plt.suptitle("Predictive EOL distribution (quantiles) vs actual", fontsize=11)
        plt.tight_layout()
        fig.savefig(self.out_dir / "eol_distributions.png", dpi=150)
        plt.close(fig)

    def plot_survival_curves(self, n_samples: int = 8) -> None:
        """
        P(EOL > x) as a function of x for a sample of batteries.
        Uses stored quantiles to approximate the survival function.
        """
        plt = self._plt
        oof = self.result.oof_predictions.dropna(subset=["actual_eol"])
        oof_sorted = oof.sort_values("actual_eol")
        indices = np.linspace(0, len(oof_sorted) - 1, min(n_samples, len(oof_sorted)), dtype=int)
        sample = oof_sorted.iloc[indices]

        fig, ax = plt.subplots(figsize=(10, 6))
        palette = self._sns.color_palette("tab10", len(sample))

        for (_, row), color in zip(sample.iterrows(), palette):
            # Reconstruct a rough survival function from stored quantiles
            q_cols = sorted([c for c in row.index if c.startswith("q") and c[1:].isdigit()])
            q_probs = [1 - float(c[1:]) / 100 for c in q_cols]
            q_cycles = [row[c] for c in q_cols]

            label = f"{row['dataset']} | actual={int(row['actual_eol'])}"
            ax.plot(q_cycles, q_probs, color=color, marker="o", markersize=4,
                    linewidth=1.5, label=label)
            ax.axvline(row["actual_eol"], color=color, linestyle=":", alpha=0.5, linewidth=1)

        for t in self.result.config.thresholds:
            ax.axvline(t, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.text(t, 0.97, f"{t}", ha="center", va="top", fontsize=7, color="gray")

        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("EOL cycle threshold (x)")
        ax.set_ylabel("P(EOL > x)")
        ax.set_title("Survival curves: likelihood of exceeding cycle threshold")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        fig.savefig(self.out_dir / "survival_curves.png", dpi=150)
        plt.close(fig)

    # ── Text / data outputs ───────────────────────────────────────────────────

    def save_summary_json(self) -> None:
        cfg = self.result.config
        summary = {
            "run_id": self.result.run_id,
            "run_name": cfg.run_name,
            "config": {
                "max_cycle": cfg.max_cycle,
                "formation_only": cfg.formation_only,
                "cv_strategy": cfg.cv_strategy,
                "n_folds": cfg.n_folds,
                "thresholds": cfg.thresholds,
                "feature_tags": cfg.feature_tags,
                "min_cycles_observed": cfg.min_cycles_observed,
            },
            "overall_metrics": {
                "mae": round(self.result.mae, 2),
                "rmse": round(self.result.rmse, 2),
                "r2": round(self.result.r2, 4),
            },
            "fold_metrics": self.result.metrics_df().to_dict(orient="records"),
            "top_20_features": (
                self.result.feature_importances.head(20).to_dict()
                if self.result.feature_importances is not None else None
            ),
        }
        path = self.out_dir / "summary.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def save_predictions_csv(self) -> None:
        self.result.oof_predictions.to_csv(
            self.out_dir / "oof_predictions.csv", index=False
        )
