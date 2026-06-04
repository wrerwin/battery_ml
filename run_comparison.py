#!/usr/bin/env python3
"""
Run all uncertainty models on CALCE+HNEI and produce calibration comparison.
Usage:  uv run run_comparison.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loader import load_dataset
from experiment import ExperimentConfig, Experiment, RandomForestModel, Reporter
from models import LightGBMQuantileModel, GaussianProcessModel, ConformalWrapper
from calibration import compare_calibration

print("Loading datasets...")
datasets = [load_dataset("CALCE.zip"), load_dataset("HNEI.zip")]

COMMON = dict(
    max_cycle=100,
    cv_strategy="group_kfold",
    n_folds=3,
    feature_tags=["capacity", "efficiency", "energy"],
    thresholds=[500, 1000, 1500],
)

models = {
    "Random Forest":      RandomForestModel(n_estimators=300, random_state=42),
    "LightGBM Quantile":  LightGBMQuantileModel(n_estimators=300),
    "Gaussian Process":   GaussianProcessModel(n_samples=500, n_restarts=3),
    "Conformal(RF)":      ConformalWrapper(RandomForestModel(n_estimators=300, random_state=42), alpha=0.1),
    "Conformal(LGBM)":    ConformalWrapper(LightGBMQuantileModel(n_estimators=300), alpha=0.1),
}

results = {}
for name, model in models.items():
    print(f"\n{'='*55}\n{name}\n{'='*55}")
    cfg = ExperimentConfig(run_name=name.lower().replace(" ", "_").replace("(", "").replace(")", ""), **COMMON)
    result = Experiment(cfg, model).run(datasets)
    Reporter(result).save_all()
    results[name] = result

print("\n" + "="*55)
print("Calibration comparison across all models")
print("="*55)
compare_calibration(results, run_tag="calce_hnei")
