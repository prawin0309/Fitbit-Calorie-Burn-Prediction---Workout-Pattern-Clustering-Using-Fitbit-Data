"""Machine-learning engines for the Fitbit project.

Task 1 - Supervised regression
    Trains and compares Linear, Ridge, Lasso, KNN, Decision Tree,
    Random Forest, SVR and (when installed) XGBoost regressors on
    ``Calories_Burned``. Reports MAE, RMSE and R², and persists the best
    model as ``artifacts/best_regressor.pkl``.

Task 2 - Unsupervised clustering
    Drops ``Workout_Type``, encodes and scales the remaining features,
    compresses them with PCA and clusters with KMeans. Reports the
    silhouette score and persists the PCA and KMeans objects.

Run standalone::

    python models.py
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

import config
from data_pipeline import (
    Database,
    fit_transformers,
    load_artifact,
    load_cleaned,
    save_artifact,
)

warnings.filterwarnings("ignore", category=UserWarning)

try:  # XGBoost is optional; the suite degrades gracefully without it.
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    XGBOOST_AVAILABLE = False

BEST_MODEL_PKL = config.ARTIFACT_DIR / "best_regressor.pkl"
MODEL_SCORES_CSV = config.REPORT_DIR / "model_comparison.csv"
PCA_PKL = config.ARTIFACT_DIR / "pca.pkl"
CLUSTER_KMEANS_PKL = config.ARTIFACT_DIR / "kmeans_workout_clusters.pkl"
CLUSTER_ENCODER_PKL = config.ARTIFACT_DIR / "cluster_encoder.pkl"
CLUSTER_SCALER_PKL = config.ARTIFACT_DIR / "cluster_scaler.pkl"
CLUSTER_PROFILE_CSV = config.REPORT_DIR / "cluster_profile.csv"


# ---------------------------------------------------------------------------
# Task 1: supervised regression
# ---------------------------------------------------------------------------
def build_model_zoo() -> dict:
    """Return the candidate regressors named in the requirement document."""
    zoo = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=config.RANDOM_SEED),
        "Lasso Regression": Lasso(alpha=0.01, random_state=config.RANDOM_SEED,
                                  max_iter=5000),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=7, weights="distance"),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=10, min_samples_leaf=5, random_state=config.RANDOM_SEED
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2,
            random_state=config.RANDOM_SEED, n_jobs=-1
        ),
        "SVR (RBF)": SVR(C=100.0, gamma="scale", epsilon=1.0),
    }
    if XGBOOST_AVAILABLE:
        zoo["XGBoost"] = XGBRegressor(
            n_estimators=400, learning_rate=0.06, max_depth=5,
            subsample=0.9, colsample_bytree=0.9,
            random_state=config.RANDOM_SEED, n_jobs=-1, verbosity=0,
        )
    return zoo


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_regressors(frame: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    """Fit every candidate model and return the score table and best model."""
    features = fit_transformers(frame)
    target = frame[config.TARGET].to_numpy(dtype=float)

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED
    )

    rows, fitted = [], {}
    for name, model in build_model_zoo().items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        cv = cross_val_score(
            model, features, target, cv=config.CV_FOLDS, scoring="r2", n_jobs=-1
        )
        rows.append(
            {
                "model": name,
                "MAE": round(float(mean_absolute_error(y_test, predictions)), 3),
                "RMSE": round(_rmse(y_test, predictions), 3),
                "R2": round(float(r2_score(y_test, predictions)), 4),
                "CV_R2_mean": round(float(cv.mean()), 4),
                "CV_R2_std": round(float(cv.std()), 4),
                "meets_target": bool(r2_score(y_test, predictions) >= config.TARGET_R2),
            }
        )
        fitted[name] = model
        print(f"[reg] {name:<20} MAE={rows[-1]['MAE']:>8.3f}  "
              f"RMSE={rows[-1]['RMSE']:>8.3f}  R2={rows[-1]['R2']:.4f}")

    scores = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    scores.to_csv(MODEL_SCORES_CSV, index=False)
    print(f"[report] model comparison -> {MODEL_SCORES_CSV.name}")

    best_name = scores.iloc[0]["model"]
    best_model = fitted[best_name]
    save_artifact(
        {"name": best_name, "model": best_model,
         "columns": list(features.columns)},
        BEST_MODEL_PKL,
    )
    print(f"[reg] best model = {best_name} (R2={scores.iloc[0]['R2']:.4f}, "
          f"target {config.TARGET_R2})")
    return scores, best_model


def feature_importance(frame: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Importance table from the persisted best model, when supported."""
    bundle = load_artifact(BEST_MODEL_PKL)
    model, columns = bundle["model"], bundle["columns"]

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_))
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    return (
        pd.DataFrame({"feature": columns, "importance": values})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def predict_calories(payload: dict) -> float:
    """Predict calories burned for a single workout session."""
    from data_pipeline import transform_new

    bundle = load_artifact(BEST_MODEL_PKL)
    row = pd.DataFrame([payload])
    features = transform_new(row).reindex(columns=bundle["columns"], fill_value=0.0)
    return float(bundle["model"].predict(features)[0])


# ---------------------------------------------------------------------------
# Task 2: unsupervised clustering
# ---------------------------------------------------------------------------
def build_cluster_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Encode and scale every feature except the dropped Workout_Type."""
    from sklearn.preprocessing import StandardScaler

    from data_pipeline import _make_encoder

    categorical = [
        c for c in config.CATEGORICAL_FEATURES if c != config.CLUSTER_DROP_COLUMN
    ]
    encoder = _make_encoder().fit(frame[categorical])
    encoded = pd.DataFrame(
        encoder.transform(frame[categorical]),
        columns=encoder.get_feature_names_out(categorical),
        index=frame.index,
    )
    scaler = StandardScaler().fit(frame[config.NUMERIC_FEATURES])
    scaled = pd.DataFrame(
        scaler.transform(frame[config.NUMERIC_FEATURES]),
        columns=config.NUMERIC_FEATURES,
        index=frame.index,
    )

    save_artifact(encoder, CLUSTER_ENCODER_PKL)
    save_artifact(scaler, CLUSTER_SCALER_PKL)
    return pd.concat([scaled, encoded], axis=1)


def train_clusters(frame: pd.DataFrame) -> dict:
    """PCA compression followed by KMeans, scored with the silhouette."""
    features = build_cluster_features(frame)

    pca = PCA(n_components=config.PCA_COMPONENTS,
              random_state=config.RANDOM_SEED).fit(features)
    components = pca.transform(features)
    explained = float(pca.explained_variance_ratio_.sum())

    kmeans = KMeans(n_clusters=config.N_CLUSTERS,
                    random_state=config.RANDOM_SEED, n_init=10)
    labels = kmeans.fit_predict(components)
    score = float(silhouette_score(components, labels))

    save_artifact(pca, PCA_PKL)
    save_artifact(kmeans, CLUSTER_KMEANS_PKL)

    labelled = frame.copy()
    labelled["cluster"] = labels
    for i in range(config.PCA_COMPONENTS):
        labelled[f"PC{i + 1}"] = components[:, i]
    labelled.to_csv(config.CLUSTERED_CSV, index=False)

    profile = cluster_profile(labelled)
    profile.to_csv(CLUSTER_PROFILE_CSV, index=False)

    print(f"[cluster] k={config.N_CLUSTERS}  silhouette={score:.4f}  "
          f"(target {config.TARGET_SILHOUETTE})  "
          f"PCA explains {explained:.1%} of variance")
    return {
        "labels": labels,
        "silhouette": score,
        "explained_variance": explained,
        "frame": labelled,
        "profile": profile,
    }


def cluster_profile(labelled: pd.DataFrame) -> pd.DataFrame:
    """Centroid-style summary used to interpret each workout segment."""
    summary = labelled.groupby("cluster").agg(
        sessions=("cluster", "size"),
        avg_bpm=("Avg_BPM", "mean"),
        resting_bpm=("Resting_BPM", "mean"),
        duration_h=("Session_Duration (hours)", "mean"),
        calories=(config.TARGET, "mean"),
        bmi=("BMI", "mean"),
        fat_pct=("Fat_Percentage", "mean"),
        frequency=("Workout_Frequency (days/week)", "mean"),
    ).round(2)

    # Name each cluster by its intensity relative to the cohort median.
    summary["kcal_per_hour"] = (summary["calories"] / summary["duration_h"]).round(1)
    intensity_rank = summary["kcal_per_hour"].rank(ascending=False)
    names = []
    for cluster in summary.index:
        rank = intensity_rank.loc[cluster]
        tier = {1: "Peak", 2: "High", 3: "Moderate"}.get(int(rank), "Light")
        body = "higher-BMI" if summary.loc[cluster, "bmi"] >= 27 else "lean"
        names.append(
            f"{tier} intensity | {body} cohort | "
            f"{summary.loc[cluster, 'kcal_per_hour']:.0f} kcal/h"
        )
    summary["segment"] = names
    return summary.reset_index()


def assign_cluster(payload: dict) -> int:
    """Assign an unseen session to one of the learned workout clusters."""
    encoder = load_artifact(CLUSTER_ENCODER_PKL)
    scaler = load_artifact(CLUSTER_SCALER_PKL)
    pca = load_artifact(PCA_PKL)
    kmeans = load_artifact(CLUSTER_KMEANS_PKL)

    row = pd.DataFrame([payload])
    categorical = [
        c for c in config.CATEGORICAL_FEATURES if c != config.CLUSTER_DROP_COLUMN
    ]
    encoded = pd.DataFrame(
        encoder.transform(row[categorical]),
        columns=encoder.get_feature_names_out(categorical),
    )
    scaled = pd.DataFrame(
        scaler.transform(row[config.NUMERIC_FEATURES]),
        columns=config.NUMERIC_FEATURES,
    )
    features = pd.concat([scaled, encoded], axis=1)
    return int(kmeans.predict(pca.transform(features))[0])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 70)
    print("Fitbit Calorie Burn Prediction - model training")
    print("=" * 70)
    frame = load_cleaned()
    print(f"[data] {len(frame)} rows, {frame.shape[1]} columns")
    if not XGBOOST_AVAILABLE:
        print("[warn] xgboost not installed - skipping the XGBoost regressor")

    print("\n--- Task 1: supervised regression ---")
    scores, _ = train_regressors(frame)
    print("\n" + scores.to_string(index=False))

    best_r2 = float(scores.iloc[0]["R2"])
    verdict = "MET" if best_r2 >= config.TARGET_R2 else "NOT MET"
    print(f"\nAcceptance: R2 >= {config.TARGET_R2} -> {verdict} ({best_r2:.4f})")

    print("\nTop features")
    print(feature_importance(frame).to_string(index=False))

    print("\n--- Task 2: unsupervised clustering ---")
    result = train_clusters(frame)
    print("\n" + result["profile"].to_string(index=False))

    sil = result["silhouette"]
    verdict = "MET" if sil >= config.TARGET_SILHOUETTE else "NOT MET"
    print(f"\nAcceptance: silhouette >= {config.TARGET_SILHOUETTE} -> "
          f"{verdict} ({sil:.4f})")

    db = Database()
    try:
        db.create_schema()
        db.load_workouts(frame)
    finally:
        db.close()

    print("\nModel training completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
