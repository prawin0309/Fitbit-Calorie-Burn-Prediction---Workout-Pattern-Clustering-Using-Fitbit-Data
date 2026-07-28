"""Data pipeline for the Fitbit calorie-burn project.

Responsibilities
----------------
1. Acquire the workout dataset (real CSV if present, otherwise a
   deterministic synthetic dataset with the documented schema).
2. Clean it: missing-value imputation, outlier detection and capping,
   derived-feature repair (BMI), and type coercion.
3. Encode categoricals and scale numerics, persisting the fitted
   transformers as ``.pkl`` artefacts.
4. Persist the cleaned frame to MySQL via ``mysql-connector-python``
   (cursor-based, no SQLAlchemy) with an automatic SQLite fallback.

Run standalone::

    python data_pipeline.py
"""

from __future__ import annotations

import pickle
import sqlite3
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config

try:  # pragma: no cover - import guard only
    import mysql.connector
    from mysql.connector import Error as MySQLError

    MYSQL_AVAILABLE = True
except ImportError:  # pragma: no cover
    MYSQL_AVAILABLE = False

    class MySQLError(Exception):
        """Placeholder so except-clauses stay valid without the driver."""


ENCODER_PKL = config.ARTIFACT_DIR / "onehot_encoder.pkl"
SCALER_PKL = config.ARTIFACT_DIR / "standard_scaler.pkl"
FEATURE_COLUMNS_PKL = config.ARTIFACT_DIR / "feature_columns.pkl"


def save_artifact(obj, path) -> None:
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)
    print(f"[save] {path.name}")


def load_artifact(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


# ---------------------------------------------------------------------------
# Database layer
# ---------------------------------------------------------------------------
class Database:
    """Cursor-based SQL wrapper over MySQL, falling back to SQLite."""

    def __init__(self) -> None:
        self.backend = "sqlite"
        self.conn = None
        self._connect()

    def _connect(self) -> None:
        backend = config.DB_BACKEND
        if backend in ("auto", "mysql") and MYSQL_AVAILABLE:
            try:
                self.conn = self._connect_mysql()
                self.backend = "mysql"
                print(f"[db] connected to MySQL {config.MYSQL_CONFIG['host']}:"
                      f"{config.MYSQL_CONFIG['port']}/"
                      f"{config.MYSQL_CONFIG['database']}")
                return
            except MySQLError as exc:
                if backend == "mysql":
                    raise
                print(f"[db] MySQL unavailable ({exc}); falling back to SQLite.")

        self.conn = sqlite3.connect(config.SQLITE_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.backend = "sqlite"
        print(f"[db] connected to SQLite at {config.SQLITE_PATH}")

    @staticmethod
    def _connect_mysql():
        cfg = dict(config.MYSQL_CONFIG)
        database = cfg.pop("database")
        bootstrap = mysql.connector.connect(connection_timeout=5, **cfg)
        cur = bootstrap.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
        cur.close()
        bootstrap.close()
        return mysql.connector.connect(connection_timeout=5, database=database, **cfg)

    def _adapt(self, sql: str) -> str:
        return sql.replace("%s", "?") if self.backend == "sqlite" else sql

    def execute(self, sql: str, params: tuple = ()) -> None:
        cur = self.conn.cursor()
        cur.execute(self._adapt(sql), params)
        self.conn.commit()
        cur.close()

    def executemany(self, sql: str, rows: list[tuple]) -> None:
        cur = self.conn.cursor()
        cur.executemany(self._adapt(sql), rows)
        self.conn.commit()
        cur.close()

    def fetch_all(self, sql: str, params: tuple = ()) -> list[dict]:
        if self.backend == "mysql":
            cur = self.conn.cursor(dictionary=True)
            cur.execute(self._adapt(sql), params)
            rows = cur.fetchall()
        else:
            cur = self.conn.cursor()
            cur.execute(self._adapt(sql), params)
            rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        return rows

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()

    def create_schema(self) -> None:
        self.execute(
            """
            CREATE TABLE IF NOT EXISTS workouts (
                session_id        INTEGER PRIMARY KEY,
                age               INT,
                gender            VARCHAR(10),
                weight_kg         DOUBLE,
                height_m          DOUBLE,
                bmi               DOUBLE,
                fat_percentage    DOUBLE,
                max_bpm           INT,
                avg_bpm           INT,
                resting_bpm       INT,
                session_hours     DOUBLE,
                workout_type      VARCHAR(20),
                water_litres      DOUBLE,
                workout_frequency INT,
                experience_level  VARCHAR(20),
                calories_burned   DOUBLE
            )
            """
        )
        print("[db] schema ready (workouts)")

    def load_workouts(self, frame: pd.DataFrame) -> None:
        self.execute("DELETE FROM workouts")
        rows = [
            (
                idx + 1, int(r["Age"]), r["Gender"], float(r["Weight (kg)"]),
                float(r["Height (m)"]), float(r["BMI"]),
                float(r["Fat_Percentage"]), int(r["Max_BPM"]),
                int(r["Avg_BPM"]), int(r["Resting_BPM"]),
                float(r["Session_Duration (hours)"]), r["Workout_Type"],
                float(r["Water_Intake (liters)"]),
                int(r["Workout_Frequency (days/week)"]),
                r["Experience_Level"], float(r[config.TARGET]),
            )
            for idx, r in frame.reset_index(drop=True).iterrows()
        ]
        self.executemany(
            "INSERT INTO workouts (session_id, age, gender, weight_kg, height_m, "
            "bmi, fat_percentage, max_bpm, avg_bpm, resting_bpm, session_hours, "
            "workout_type, water_litres, workout_frequency, experience_level, "
            "calories_burned) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
            "%s, %s, %s, %s, %s, %s)",
            rows,
        )
        print(f"[db] loaded {len(rows)} workout rows")


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------
# Metabolic-equivalent multipliers per workout type, used to give the target a
# physiologically plausible dependence on the features.
_MET = {"Yoga": 3.0, "Strength": 5.5, "Cardio": 8.0, "HIIT": 11.0}
_EXPERIENCE_EFFICIENCY = {"Beginner": 0.94, "Intermediate": 1.0, "Advanced": 1.07}


def generate_synthetic_dataset() -> pd.DataFrame:
    """Build a deterministic dataset matching the documented column list."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    n = config.N_SYNTHETIC_ROWS

    gender = rng.choice(config.GENDERS, n, p=[0.52, 0.48])
    age = rng.integers(18, 65, n)
    is_male = gender == "Male"

    height = np.where(
        is_male, rng.normal(1.75, 0.07, n), rng.normal(1.62, 0.06, n)
    ).clip(1.45, 2.05)
    weight = np.where(
        is_male, rng.normal(78, 12, n), rng.normal(65, 11, n)
    ).clip(42, 135)
    bmi = weight / height**2

    fat = np.where(is_male, 8 + 0.9 * (bmi - 18), 15 + 0.95 * (bmi - 18))
    fat = (fat + rng.normal(0, 2.5, n)).clip(5, 48)

    experience = rng.choice(config.EXPERIENCE_LEVELS, n, p=[0.4, 0.38, 0.22])
    workout = rng.choice(config.WORKOUT_TYPES, n, p=[0.34, 0.28, 0.21, 0.17])

    resting = (rng.normal(68, 7, n) - 4 * (experience == "Advanced")).clip(45, 92)
    max_bpm = (220 - age + rng.normal(0, 5, n)).clip(150, 205)

    intensity = np.array([_MET[w] for w in workout]) / 11.0
    avg_bpm = (
        resting + intensity * (max_bpm - resting) * rng.normal(0.78, 0.07, n)
    ).clip(70, 200)

    duration = np.where(
        workout == "HIIT",
        rng.normal(0.65, 0.15, n),
        np.where(workout == "Yoga", rng.normal(1.10, 0.22, n),
                 rng.normal(1.00, 0.25, n)),
    ).clip(0.25, 2.5)

    frequency = rng.integers(2, 8, n)
    water = (1.2 + 0.9 * duration + 0.01 * weight + rng.normal(0, 0.25, n)).clip(0.5, 5.0)

    met = np.array([_MET[w] for w in workout])
    efficiency = np.array([_EXPERIENCE_EFFICIENCY[e] for e in experience])

    # Core physiology: kcal ≈ MET × weight(kg) × duration(h), modulated by
    # heart-rate response, body composition, sex and training efficiency.
    calories = (
        met * weight * duration
        * (0.72 + 0.45 * (avg_bpm - resting) / (max_bpm - resting))
        * efficiency
        * np.where(is_male, 1.06, 0.95)
        * (1.0 - 0.0035 * (fat - 20))
    )
    calories = (calories + rng.normal(0, 22, n)).clip(30, None)

    frame = pd.DataFrame(
        {
            "Age": age,
            "Gender": gender,
            "Weight (kg)": weight.round(1),
            "Height (m)": height.round(2),
            "BMI": bmi.round(2),
            "Fat_Percentage": fat.round(1),
            "Max_BPM": max_bpm.round(0).astype(int),
            "Avg_BPM": avg_bpm.round(0).astype(int),
            "Resting_BPM": resting.round(0).astype(int),
            "Session_Duration (hours)": duration.round(2),
            "Workout_Type": workout,
            "Water_Intake (liters)": water.round(2),
            "Workout_Frequency (days/week)": frequency,
            "Experience_Level": experience,
            config.TARGET: calories.round(1),
        }
    )

    # Inject realistic dirt so the cleaning stage is meaningful.
    for column, fraction in (
        ("Fat_Percentage", 0.04), ("Water_Intake (liters)", 0.03), ("BMI", 0.02)
    ):
        idx = rng.choice(n, int(n * fraction), replace=False)
        frame.loc[idx, column] = np.nan

    outlier_idx = rng.choice(n, int(n * 0.01), replace=False)
    frame.loc[outlier_idx, "Session_Duration (hours)"] *= 6.0
    return frame


def load_raw_dataset() -> pd.DataFrame:
    if config.RAW_CSV.exists():
        print(f"[data] using real dataset: {config.RAW_CSV.name}")
        return pd.read_csv(config.RAW_CSV)
    print("[data] no real dataset found (see DATASET_MISSING.txt); "
          "generating synthetic data with the documented schema")
    frame = generate_synthetic_dataset()
    frame.to_csv(config.RAW_CSV, index=False)
    return frame


# ---------------------------------------------------------------------------
# Schema normalisation
# ---------------------------------------------------------------------------
# The supplied file carries an unnamed index column, an ordinal
# ``Experience_Level``, and three derived MET columns. Normalise once, here.
_TARGET_ALIASES = {
    "calories_burned": config.TARGET,
    "calories_burned (kcal)": config.TARGET,
    "calories": config.TARGET,
}


def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop index noise, canonicalise the target, and label ordinal columns."""
    frame = frame.copy()

    junk = [c for c in frame.columns if str(c).lower().startswith("unnamed")]
    if junk:
        frame = frame.drop(columns=junk)

    renamed = {}
    for column in frame.columns:
        key = str(column).strip().lower()
        if key in _TARGET_ALIASES:
            renamed[column] = _TARGET_ALIASES[key]
    if renamed:
        frame = frame.rename(columns=renamed)

    # Experience_Level arrives as an ordinal integer; map it to its label.
    if "Experience_Level" in frame.columns:
        levels = frame["Experience_Level"]
        if pd.api.types.is_numeric_dtype(levels):
            frame["Experience_Level"] = (
                levels.round().astype("Int64").map(config.EXPERIENCE_LEVEL_MAP)
            )
        frame["Experience_Level"] = frame["Experience_Level"].astype("string")

    return frame


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def cap_outliers(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Winsorise a numeric series to the Tukey fences."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return series.clip(q1 - k * iqr, q3 + k * iqr)


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Impute, repair derived columns, cap outliers and drop duplicates."""
    frame = normalise_columns(frame)
    before = len(frame)
    frame = frame.drop_duplicates()

    for column in config.CATEGORICAL_FEATURES:
        frame[column] = frame[column].astype("string")
        mode = frame[column].mode()
        frame[column] = frame[column].fillna(mode.iat[0] if len(mode) else "Unknown")

    numeric = config.NUMERIC_FEATURES + [config.TARGET]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # BMI is a derived column - recompute rather than impute where possible.
    recomputed = frame["Weight (kg)"] / frame["Height (m)"] ** 2
    frame["BMI"] = frame["BMI"].fillna(recomputed).round(2)

    missing_before = int(frame[numeric].isna().sum().sum())
    for column in numeric:
        frame[column] = frame[column].fillna(frame[column].median())

    capped = 0
    for column in config.NUMERIC_FEATURES:
        original = frame[column].copy()
        frame[column] = cap_outliers(frame[column])
        capped += int((original != frame[column]).sum())

    for column in config.LEAKAGE_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame[column] = frame[column].fillna(frame[column].median())

    frame = frame.dropna(subset=[config.TARGET]).reset_index(drop=True)

    print(f"[clean] {before} -> {len(frame)} rows | "
          f"{missing_before} missing values imputed | "
          f"{capped} outlier values capped")
    return frame


# ---------------------------------------------------------------------------
# Encoding and scaling
# ---------------------------------------------------------------------------
def _make_encoder() -> OneHotEncoder:
    """OneHotEncoder that works across scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def fit_transformers(frame: pd.DataFrame,
                     categorical: list[str] | None = None) -> pd.DataFrame:
    """Fit and persist the one-hot encoder and the standard scaler."""
    categorical = categorical or config.CATEGORICAL_FEATURES

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

    features = pd.concat([scaled, encoded], axis=1)

    save_artifact(encoder, ENCODER_PKL)
    save_artifact(scaler, SCALER_PKL)
    save_artifact(
        {"categorical": categorical,
         "numeric": config.NUMERIC_FEATURES,
         "columns": list(features.columns)},
        FEATURE_COLUMNS_PKL,
    )
    print(f"[encode] feature matrix shape = {features.shape}")
    return features


def transform_new(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the persisted transformers to unseen rows."""
    encoder = load_artifact(ENCODER_PKL)
    scaler = load_artifact(SCALER_PKL)
    meta = load_artifact(FEATURE_COLUMNS_PKL)

    encoded = pd.DataFrame(
        encoder.transform(frame[meta["categorical"]]),
        columns=encoder.get_feature_names_out(meta["categorical"]),
        index=frame.index,
    )
    scaled = pd.DataFrame(
        scaler.transform(frame[meta["numeric"]]),
        columns=meta["numeric"],
        index=frame.index,
    )
    features = pd.concat([scaled, encoded], axis=1)
    return features.reindex(columns=meta["columns"], fill_value=0.0)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_pipeline() -> pd.DataFrame:
    raw = load_raw_dataset()
    cleaned = clean_dataset(raw)
    cleaned.to_csv(config.CLEANED_CSV, index=False)
    print(f"[data] cleaned dataset -> {config.CLEANED_CSV.name}")

    fit_transformers(cleaned)

    db = Database()
    try:
        db.create_schema()
        db.load_workouts(cleaned)
        total = db.fetch_all("SELECT COUNT(*) AS n FROM workouts")[0]["n"]
        print(f"[verify] workouts row count = {total}")
    finally:
        db.close()
    return cleaned


def load_cleaned() -> pd.DataFrame:
    if config.CLEANED_CSV.exists():
        return pd.read_csv(config.CLEANED_CSV)
    return run_pipeline()


def main() -> int:
    print("=" * 70)
    print("Fitbit Calorie Burn Prediction - data pipeline")
    print("=" * 70)
    cleaned = run_pipeline()
    print("\nTarget summary:")
    print(cleaned[config.TARGET].describe().round(2).to_string())
    print("\nPipeline completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
