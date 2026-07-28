"""Hyperparameter tuning for the winning regressor.

``models.py`` selects the best of eight candidates at their default settings.
This module takes that winner and optimises it with ``RandomizedSearchCV``,
then reports tuned against default on the same held-out split so the gain (or
lack of one) is measurable rather than assumed.

The tuned model only replaces ``artifacts/best_regressor.pkl`` when it beats
the default on held-out R2 -- tuning is not allowed to make the shipped model
worse.

    python data_pipeline.py
    python models.py
    python tune.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

import config
import models
from data_pipeline import fit_transformers, load_cleaned, save_artifact

REPORT_DIR = Path(__file__).resolve().parent / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
TUNING_CSV = REPORT_DIR / "tuning_results.csv"

N_ITER = 40

PARAM_GRIDS: dict[str, dict] = {
    "XGBoost": {
        "n_estimators": [200, 300, 400, 600, 800, 1000],
        "max_depth": [3, 4, 5, 6, 8, 10],
        "learning_rate": [0.01, 0.02, 0.05, 0.08, 0.1, 0.15],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    },
    "Random Forest": {
        "n_estimators": [200, 300, 500, 800],
        "max_depth": [None, 10, 20, 30, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 1.0],
    },
    "Decision Tree": {
        "max_depth": [None, 5, 10, 15, 20, 30],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": [None, "sqrt", "log2"],
    },
    "KNN Regressor": {
        "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
}


def _metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


def main() -> int:
    frame = load_cleaned()
    features = fit_transformers(frame)
    target = frame[config.TARGET].to_numpy(dtype=float)

    # Same split and seed as models.py, so the comparison is like-for-like.
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
    )

    scores = pd.read_csv(models.MODEL_SCORES_CSV)
    best_name = str(scores.iloc[0]["model"])
    if best_name not in PARAM_GRIDS:
        print(f"[tune] no grid defined for '{best_name}' - nothing to do")
        return 0

    print("=" * 70)
    print(f"Hyperparameter tuning - {best_name}")
    print("=" * 70)

    zoo = models.build_model_zoo()
    baseline = zoo[best_name]
    baseline.fit(x_train, y_train)
    default_metrics = _metrics(y_test, baseline.predict(x_test))
    print(f"[tune] default    R2={default_metrics['R2']:.4f}  "
          f"RMSE={default_metrics['RMSE']:.3f}  "
          f"MAE={default_metrics['MAE']:.3f}")

    search = RandomizedSearchCV(
        estimator=zoo[best_name],
        param_distributions=PARAM_GRIDS[best_name],
        n_iter=N_ITER,
        scoring="r2",
        cv=config.CV_FOLDS,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(x_train, y_train)

    tuned = search.best_estimator_
    tuned_metrics = _metrics(y_test, tuned.predict(x_test))
    print(f"[tune] tuned      R2={tuned_metrics['R2']:.4f}  "
          f"RMSE={tuned_metrics['RMSE']:.3f}  "
          f"MAE={tuned_metrics['MAE']:.3f}")
    print(f"[tune] best params: {search.best_params_}")
    print(f"[tune] best CV R2 : {search.best_score_:.4f} "
          f"over {N_ITER} sampled configurations")

    delta_r2 = tuned_metrics["R2"] - default_metrics["R2"]
    delta_rmse = tuned_metrics["RMSE"] - default_metrics["RMSE"]
    improved = delta_r2 > 0

    pd.DataFrame([
        {"model": best_name, "setting": "default", **default_metrics},
        {"model": best_name, "setting": "tuned", **tuned_metrics},
    ]).to_csv(TUNING_CSV, index=False)
    print(f"[report] tuning results -> {TUNING_CSV.name}")

    print(f"\n[tune] delta R2 = {delta_r2:+.4f}   "
          f"delta RMSE = {delta_rmse:+.3f} kcal")
    if improved:
        save_artifact(
            {"name": f"{best_name} (tuned)", "model": tuned,
             "columns": list(features.columns)},
            models.BEST_MODEL_PKL,
        )
        print("[tune] tuned model beats default -> best_regressor.pkl updated")
    else:
        print("[tune] tuning did not beat the default; shipped model "
              "unchanged. Reported as a finding, not hidden.")

    print(f"\nAcceptance: R2 >= {config.TARGET_R2} -> "
          f"{'MET' if tuned_metrics['R2'] >= config.TARGET_R2 else 'NOT MET'} "
          f"({tuned_metrics['R2']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
