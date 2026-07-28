"""Configuration for the Fitbit calorie-burn and clustering project."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"
REPORT_DIR = BASE_DIR / "reports"

for _folder in (DATA_DIR, ARTIFACT_DIR, REPORT_DIR):
    _folder.mkdir(parents=True, exist_ok=True)

RAW_CSV = DATA_DIR / "fitbit_workouts.csv"
CLEANED_CSV = DATA_DIR / "cleaned_fitbit.csv"
CLUSTERED_CSV = DATA_DIR / "clustered_fitbit.csv"
SQLITE_PATH = DATA_DIR / "guvi_db.sqlite3"

MYSQL_CONFIG = {
    "host": os.getenv("FITBIT_DB_HOST", "localhost"),
    "port": int(os.getenv("FITBIT_DB_PORT", "3306")),
    "user": os.getenv("FITBIT_DB_USER", "root"),
    "password": os.getenv("FITBIT_DB_PASSWORD", "root"),
    "database": os.getenv("FITBIT_DB_NAME", "guvi_db"),
}
DB_BACKEND = os.getenv("FITBIT_DB_BACKEND", "auto").lower()

RANDOM_SEED = 42
N_SYNTHETIC_ROWS = 3000
TEST_SIZE = 0.2
CV_FOLDS = 5

TARGET = "Calories_Burned (kcal)"
CLUSTER_DROP_COLUMN = "Workout_Type"

NUMERIC_FEATURES = [
    "Age", "Weight (kg)", "Height (m)", "BMI", "Fat_Percentage",
    "Max_BPM", "Avg_BPM", "Resting_BPM", "Session_Duration (hours)",
    "Water_Intake (liters)", "Workout_Frequency (days/week)",
]
CATEGORICAL_FEATURES = ["Gender", "Workout_Type", "Experience_Level"]

WORKOUT_TYPES = ["Cardio", "HIIT", "Mixed", "Strength", "Yoga"]

# The dataset stores Experience_Level as an ordinal integer 0-3. These are the
# human-readable labels used everywhere in the UI and the encoders.
EXPERIENCE_LEVEL_MAP = {
    0: "Beginner",
    1: "Intermediate",
    2: "Advanced",
    3: "Expert",
}
EXPERIENCE_LEVELS = list(EXPERIENCE_LEVEL_MAP.values())
GENDERS = ["Female", "Male"]

# Columns present in the raw file that are deterministic functions of the
# target's generating formula (Effective_MET = Base_MET x HR_Intensity, and
# calories are derived from Effective_MET x weight x duration). Including them
# as model inputs would leak the answer, so they are kept for exploratory
# analysis but excluded from the feature matrix.
LEAKAGE_COLUMNS = ["Base_MET", "HR_Intensity", "Effective_MET"]

# Acceptance criteria taken directly from the requirement document.
TARGET_R2 = 0.80
TARGET_SILHOUETTE = 0.15

N_CLUSTERS = 4
PCA_COMPONENTS = 3
