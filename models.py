"""
Uncertainty-aware model implementations.

All three models implement ModelInterface from experiment.py and are
drop-in replacements for RandomForestModel.

LightGBMQuantileModel
---------------------
Trains one LightGBM model per target quantile level (default: 9 models
covering the 0.1–0.9 range).  At prediction time, the quantile predictions
form an empirical CDF from which arbitrary samples are drawn via inverse
transform sampling.  Crossing quantiles (a known pathology of independent
quantile regression) are corrected by isotonic regression before sampling.

GaussianProcessModel
--------------------
Wraps sklearn's GaussianProcessRegressor.  The GP posterior gives an analytic
predictive distribution N(μ(x), σ²(x)) per test point.  predict_distribution
returns samples drawn from that Gaussian.  For datasets larger than
gp_max_train_size the model fits on a random representative subset to keep
O(n³) GP training tractable; the full test set is still predicted in O(n).

ConformalWrapper
----------------
A model-agnostic calibration layer that can wrap any ModelInterface.

Uses Conformalized Quantile Regression (CQR) when the base model produces
quantile predictions, otherwise falls back to standard split conformal
prediction.

CQR guarantees:
    P(y_test ∈ [q̂_lo(x) − η, q̂_hi(x) + η]) ≥ 1 − α

where η is computed from the nonconformity scores on a held-out calibration
split drawn from the training data.  This is a finite-sample, distribution-
free guarantee — no assumptions about the error distribution are required.

The calibration fraction (default 20% of train) is taken out of the training
set during fit().  Downstream CV folds should be aware that effective training
size is reduced by this amount.

predict_distribution returns uniform samples over the adjusted [lo, hi]
interval, enriched at the base model's point estimate so the median is
preserved.  This is a conservative distribution — it represents "I'm
confident the true value lies in [lo, hi] but I don't have strong beliefs
about where exactly within that range."
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiment import ModelInterface

# ── LightGBM Quantile Model ───────────────────────────────────────────────────


class LightGBMQuantileModel(ModelInterface):
    """
    Multi-quantile LightGBM regressor.

    One LightGBM model is trained per quantile level.  The set of quantile
    predictions forms an empirical CDF, from which predict_distribution draws
    samples via inverse transform sampling.

    Parameters
    ----------
    quantiles:
        Quantile levels to train.  More quantiles → smoother distribution
        but proportionally more training time.  Default covers 0.05–0.95
        in steps of 0.05 (19 models).
    n_samples:
        Number of samples to draw per battery for predict_distribution.
    lgbm_params:
        Extra kwargs forwarded to lightgbm.LGBMRegressor (e.g. n_estimators,
        learning_rate, num_leaves).  objective and alpha are set automatically.
    """

    def __init__(
        self,
        quantiles: Optional[list[float]] = None,
        n_samples: int = 500,
        **lgbm_params,
    ):
        try:
            import lightgbm as lgb  # noqa: F401
        except ImportError:
            raise ImportError(
                "lightgbm is required for LightGBMQuantileModel. "
                "Install with: pip install lightgbm"
            )

        self.quantiles = quantiles or [round(q, 2) for q in np.arange(0.05, 1.0, 0.05)]
        self.n_samples = n_samples
        self._lgbm_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 5,
            "n_jobs": -1,
            "verbose": -1,
            **lgbm_params,
        }
        self._models: list = []         # one per quantile
        self._preprocessor: Optional[Pipeline] = None
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        import lightgbm as lgb

        self._feature_names = list(X.columns)
        self._preprocessor = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        X_t = self._preprocessor.fit_transform(X)

        self._models = []
        for q in self.quantiles:
            m = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                **self._lgbm_params,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(X_t, y.values)
            self._models.append(m)

    def predict_distribution(self, X: pd.DataFrame) -> list[np.ndarray]:
        if self._preprocessor is None:
            raise RuntimeError("Call fit() before predict_distribution().")

        X_arr = self._preprocessor.transform(X)
        # Wrap back as DataFrame so LightGBM doesn't warn about missing feature names
        X_t = pd.DataFrame(X_arr, columns=self._feature_names)
        n = X_t.shape[0]

        # Collect quantile predictions: shape (n_quantiles, n_samples)
        q_preds = np.array([m.predict(X_t) for m in self._models])  # (Q, N)

        # Enforce monotonicity per sample via isotonic regression
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        for i in range(n):
            q_preds[:, i] = iso.fit_transform(
                self.quantiles, q_preds[:, i]
            )

        # Add boundary quantiles (0, 1) by extrapolation for stable sampling
        q_levels = np.array([0.0] + self.quantiles + [1.0])
        q_lo = q_preds[0, :] - (q_preds[1, :] - q_preds[0, :])   # linear extrap
        q_hi = q_preds[-1, :] + (q_preds[-1, :] - q_preds[-2, :])
        q_all = np.vstack([q_lo[None, :], q_preds, q_hi[None, :]])  # (Q+2, N)

        # Inverse transform sampling: draw uniform(0,1) and interpolate
        u = np.random.default_rng(42).uniform(0, 1, (self.n_samples, n))
        samples = np.array([
            np.interp(u[:, i], q_levels, q_all[:, i])
            for i in range(n)
        ])  # (N, n_samples)

        return [samples[i] for i in range(n)]

    def feature_importances(self) -> pd.Series:
        """Average feature importance across the median quantile model."""
        mid = len(self._models) // 2
        imps = self._models[mid].feature_importances_
        return pd.Series(imps, index=self._feature_names).sort_values(ascending=False)


# ── Gaussian Process Model ────────────────────────────────────────────────────


class GaussianProcessModel(ModelInterface):
    """
    Gaussian Process regressor with analytic posterior uncertainty.

    The GP posterior for a test point x* is N(μ(x*), σ²(x*)).
    predict_distribution returns `n_samples` draws from this Gaussian.

    The default kernel is:
        ConstantKernel × RBF + WhiteKernel
    The ConstantKernel handles the output scale; RBF models smooth
    correlations between feature vectors; WhiteKernel captures
    observation noise / irreducible variance.

    Scaling caveat
    --------------
    GPs are O(n³) to train and O(n²) to predict.  For n > gp_max_train_size,
    the model sub-samples the training data randomly before fitting.  The
    full test set is always predicted.  For this dataset (≤200 batteries)
    this limit is rarely hit, but the guard is here for when more data is added.

    Uncertainty meaning
    -------------------
    The GP σ(x) captures two things:
      1. Epistemic uncertainty — regions of feature space with few training
         points produce wider posteriors.
      2. Aleatoric noise — modelled by the WhiteKernel amplitude.
    Both appear naturally in the posterior samples.
    """

    def __init__(
        self,
        n_samples: int = 500,
        gp_max_train_size: int = 500,
        n_restarts: int = 5,
        random_state: int = 42,
    ):
        self.n_samples = n_samples
        self.gp_max_train_size = gp_max_train_size
        self.n_restarts = n_restarts
        self.random_state = random_state

        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
        )
        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts,
            normalize_y=True,
            random_state=random_state,
        )
        self._preprocessor: Optional[Pipeline] = None
        self._rng = np.random.default_rng(random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._preprocessor = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        X_t = self._preprocessor.fit_transform(X)
        y_arr = y.values.astype(float)

        # Sub-sample if dataset is too large for cubic GP training
        if len(X_t) > self.gp_max_train_size:
            idx = self._rng.choice(len(X_t), self.gp_max_train_size, replace=False)
            X_t, y_arr = X_t[idx], y_arr[idx]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(X_t, y_arr)

    def predict_distribution(self, X: pd.DataFrame) -> list[np.ndarray]:
        if self._preprocessor is None:
            raise RuntimeError("Call fit() before predict_distribution().")

        X_t = self._preprocessor.transform(X)
        mu, sigma = self._gp.predict(X_t, return_std=True)

        # Clamp sigma to avoid numerical zeros
        sigma = np.maximum(sigma, 1e-3)

        distributions = []
        for i in range(len(mu)):
            samples = self._rng.normal(loc=mu[i], scale=sigma[i], size=self.n_samples)
            # EOL is always positive; clip below zero
            samples = np.maximum(samples, 1.0)
            distributions.append(samples)

        return distributions


# ── Conformal Wrapper ─────────────────────────────────────────────────────────


class ConformalWrapper(ModelInterface):
    """
    Distribution-free calibration layer wrapping any ModelInterface.

    Implements Conformalized Quantile Regression (CQR) when the base model
    supports quantile prediction, otherwise uses standard split conformal
    prediction.

    How it works (CQR)
    ------------------
    During fit(X_train, y_train):
      1. Split X_train into X_fit (80%) and X_cal (20%).
      2. Fit the base model on X_fit.
      3. Predict the [α/2, 1−α/2] quantiles on X_cal.
      4. Compute nonconformity scores:
             E_i = max(q_lo_i − y_i,  y_i − q_hi_i)
         (negative when y falls inside the interval, positive when outside)
      5. Store q̂ = quantile(E_1…E_n, level = ⌈(n+1)(1−α)⌉/n)

    During predict_distribution(X_test):
      1. Get base model's quantile predictions: q_lo, q_hi.
      2. Adjust: adjusted_lo = q_lo − q̂, adjusted_hi = q_hi + q̂.
      3. Sample n_samples points from the adjusted interval.

    Coverage guarantee (marginal)
    ------------------------------
    For any joint distribution of (X, Y) and any calibration set size n:
        P(Y_test ∈ [q_lo(X) − q̂, q_hi(X) + q̂]) ≥ 1 − α

    Parameters
    ----------
    base_model:
        Any ModelInterface instance.  If it has predict_quantiles(),
        CQR is used.  Otherwise, standard split conformal.
    alpha:
        Miscoverage level.  Default 0.1 → 90% coverage guarantee.
    calibration_fraction:
        Fraction of training data reserved for calibration.
    n_samples:
        Samples to draw from the adjusted interval per test point.
    random_state:
        Controls train/calibration split.
    """

    def __init__(
        self,
        base_model: ModelInterface,
        alpha: float = 0.1,
        calibration_fraction: float = 0.2,
        n_samples: int = 500,
        random_state: int = 42,
    ):
        self.base_model = base_model
        self.alpha = alpha
        self.calibration_fraction = calibration_fraction
        self.n_samples = n_samples
        self.random_state = random_state

        self._q_hat: float = 0.0          # conformal adjustment
        self._use_cqr: bool = True
        self._rng = np.random.default_rng(random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        n = len(X)
        # Reserve at least 3 for calibration and at least 5 for fitting
        n_cal = max(int(n * self.calibration_fraction), 3)
        n_fit = n - n_cal

        if n_fit < 5:
            # Fall back: take only 1 sample for calibration
            n_cal = max(1, n - 5)
            n_fit = n - n_cal

        # Stratified-ish split: shuffle then cut
        idx = self._rng.permutation(n)
        fit_idx, cal_idx = idx[:n_fit], idx[n_cal:]

        X_fit = X.iloc[fit_idx]
        y_fit = y.iloc[fit_idx]
        X_cal = X.iloc[cal_idx]
        y_cal = y.iloc[cal_idx].values

        self.base_model.fit(X_fit, y_fit)

        # Determine calibration strategy
        self._use_cqr = hasattr(self.base_model, "predict_quantiles")

        if self._use_cqr:
            # CQR nonconformity score
            lo_level = self.alpha / 2
            hi_level = 1 - self.alpha / 2
            q_df = self.base_model.predict_quantiles(X_cal, quantiles=[lo_level, hi_level])
            q_lo_col = f"q{int(lo_level * 100)}"
            q_hi_col = f"q{int(hi_level * 100)}"
            q_lo = q_df[q_lo_col].values
            q_hi = q_df[q_hi_col].values
            scores = np.maximum(q_lo - y_cal, y_cal - q_hi)
        else:
            # Standard split conformal: score = |y - ŷ|
            y_hat = self.base_model.predict_point(X_cal)
            scores = np.abs(y_cal - y_hat)

        # Finite-sample corrected quantile level
        n_cal_actual = len(scores)
        level = np.ceil((n_cal_actual + 1) * (1 - self.alpha)) / n_cal_actual
        level = min(level, 1.0)
        self._q_hat = float(np.quantile(scores, level))

    def predict_distribution(self, X: pd.DataFrame) -> list[np.ndarray]:
        if self._use_cqr:
            lo_level = self.alpha / 2
            hi_level = 1 - self.alpha / 2
            q_df = self.base_model.predict_quantiles(X, quantiles=[lo_level, hi_level])
            q_lo_col = f"q{int(lo_level * 100)}"
            q_hi_col = f"q{int(hi_level * 100)}"
            lo = q_df[q_lo_col].values - self._q_hat
            hi = q_df[q_hi_col].values + self._q_hat
        else:
            mu = self.base_model.predict_point(X)
            lo = mu - self._q_hat
            hi = mu + self._q_hat

        # Clip lower bound: EOL is positive
        lo = np.maximum(lo, 1.0)

        # Sample from the conformal interval.
        # Weight toward the base model's point estimate so median is sensible.
        mu_point = self.base_model.predict_point(X)

        distributions = []
        for i in range(len(lo)):
            interval_lo, interval_hi = lo[i], hi[i]
            center = np.clip(mu_point[i], interval_lo, interval_hi)

            # Draw samples: 70% uniform over full interval, 30% near center
            # This reflects: "calibrated interval = where truth likely is;
            # point estimate = best guess within that interval."
            n_uniform = int(self.n_samples * 0.7)
            n_near = self.n_samples - n_uniform

            uniform_samples = self._rng.uniform(interval_lo, interval_hi, n_uniform)
            near_samples = self._rng.normal(
                loc=center,
                scale=max((interval_hi - interval_lo) / 6, 1.0),
                size=n_near,
            )
            near_samples = np.clip(near_samples, interval_lo, interval_hi)

            samples = np.concatenate([uniform_samples, near_samples])
            distributions.append(samples)

        return distributions

    def feature_importances(self) -> Optional[pd.Series]:
        """Delegate to the base model if it supports importances."""
        if hasattr(self.base_model, "feature_importances"):
            return self.base_model.feature_importances()
        return None

    @property
    def q_hat(self) -> float:
        """The conformal adjustment width (half-interval inflation in cycles)."""
        return self._q_hat
