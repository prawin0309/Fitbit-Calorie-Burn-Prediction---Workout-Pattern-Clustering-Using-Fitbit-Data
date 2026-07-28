"""Streamlit application for the Fitbit calorie-burn project.

Pages
-----
Overview · Exploratory Analysis · Model Comparison · Calorie Predictor ·
Workout Clusters · Cluster Assigner

Run::

    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

import config
import models
from data_pipeline import load_cleaned

st.set_page_config(
    page_title="Fitbit Calorie Burn & Workout Clustering",
    page_icon="⌚",
    layout="wide",
)

PAGES = [
    "Overview",
    "Exploratory Analysis",
    "Model Comparison",
    "Calorie Predictor",
    "Workout Clusters",
    "Cluster Assigner",
]


# ---------------------------------------------------------------------------
# Cached data access
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading cleaned workout data…")
def get_data() -> pd.DataFrame:
    return load_cleaned()


@st.cache_data(show_spinner=False)
def get_scores() -> pd.DataFrame | None:
    if models.MODEL_SCORES_CSV.exists():
        return pd.read_csv(models.MODEL_SCORES_CSV)
    return None


@st.cache_data(show_spinner=False)
def get_clustered() -> pd.DataFrame | None:
    if config.CLUSTERED_CSV.exists():
        return pd.read_csv(config.CLUSTERED_CSV)
    return None


def models_trained() -> bool:
    return models.BEST_MODEL_PKL.exists()


def training_warning() -> None:
    st.warning("Model artefacts not found. Run `python models.py` first.")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_overview(frame: pd.DataFrame) -> None:
    st.header("⌚ Project Overview")
    st.markdown(
        "Predict **calories burned per workout session** with supervised "
        "regression, and discover **hidden workout patterns** with PCA + "
        "KMeans clustering."
    )

    cols = st.columns(4)
    cols[0].metric("Sessions", f"{len(frame):,}")
    cols[1].metric("Avg calories", f"{frame[config.TARGET].mean():.0f} kcal")
    cols[2].metric("Avg duration",
                   f"{frame['Session_Duration (hours)'].mean():.2f} h")
    cols[3].metric("Workout types", frame["Workout_Type"].nunique())

    scores = get_scores()
    if scores is not None:
        best = scores.iloc[0]
        st.success(
            f"Best regressor: **{best['model']}** — "
            f"R² = {best['R2']:.4f} (target ≥ {config.TARGET_R2}), "
            f"MAE = {best['MAE']:.2f} kcal, RMSE = {best['RMSE']:.2f} kcal"
        )

    st.dataframe(frame.head(25), use_container_width=True, hide_index=True)


def page_eda(frame: pd.DataFrame) -> None:
    st.header("🔎 Exploratory Analysis")

    left, right = st.columns(2)
    left.plotly_chart(
        px.histogram(frame, x=config.TARGET, nbins=40, color="Gender",
                     marginal="box", title="Calorie distribution by gender"),
        use_container_width=True,
    )
    right.plotly_chart(
        px.box(frame, x="Workout_Type", y=config.TARGET, color="Workout_Type",
               title="Calories burned by workout type"),
        use_container_width=True,
    )

    st.plotly_chart(
        px.scatter(frame, x="Session_Duration (hours)", y=config.TARGET,
                   color="Workout_Type", size="Avg_BPM", opacity=0.65,
                   trendline="ols",
                   title="Duration versus calories burned"),
        use_container_width=True,
    )

    numeric = frame.select_dtypes("number")
    st.plotly_chart(
        px.imshow(numeric.corr().round(2), text_auto=True, aspect="auto",
                  color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                  title="Numeric feature correlation"),
        use_container_width=True,
    )

    st.plotly_chart(
        px.box(frame, x="Experience_Level", y="Avg_BPM", color="Experience_Level",
               category_orders={"Experience_Level": config.EXPERIENCE_LEVELS},
               title="Average heart rate by experience level"),
        use_container_width=True,
    )


def page_model_comparison(frame: pd.DataFrame) -> None:
    st.header("📊 Model Comparison")
    scores = get_scores()
    if scores is None:
        training_warning()
        return

    st.plotly_chart(
        px.bar(scores.sort_values("R2"), x="R2", y="model", orientation="h",
               color="R2", color_continuous_scale="Viridis",
               title="Test-set R² by model"),
        use_container_width=True,
    )
    st.dataframe(scores, use_container_width=True, hide_index=True)

    errors = scores.melt(id_vars="model", value_vars=["MAE", "RMSE"],
                         var_name="metric", value_name="kcal")
    st.plotly_chart(
        px.bar(errors, x="model", y="kcal", color="metric", barmode="group",
               title="Error metrics (lower is better)"),
        use_container_width=True,
    )

    st.subheader("Feature importance — best model")
    st.plotly_chart(
        px.bar(models.feature_importance(frame).sort_values("importance"),
               x="importance", y="feature", orientation="h",
               title="Most influential features"),
        use_container_width=True,
    )


def page_predictor(frame: pd.DataFrame) -> None:
    st.header("🔮 Calorie Predictor")
    if not models_trained():
        training_warning()
        return

    with st.form("predict"):
        col_a, col_b, col_c = st.columns(3)

        age = col_a.slider("Age", 18, 70, 32)
        gender = col_a.selectbox("Gender", config.GENDERS)
        weight = col_a.slider("Weight (kg)", 40.0, 140.0, 72.0, 0.5)
        height = col_a.slider("Height (m)", 1.40, 2.10, 1.72, 0.01)

        fat = col_b.slider("Fat percentage", 5.0, 50.0, 21.0, 0.5)
        max_bpm = col_b.slider("Max BPM", 140, 210, 185)
        avg_bpm = col_b.slider("Average BPM", 70, 200, 142)
        resting = col_b.slider("Resting BPM", 45, 95, 66)

        duration = col_c.slider("Session duration (hours)", 0.25, 2.50, 1.00, 0.05)
        workout = col_c.selectbox("Workout type", config.WORKOUT_TYPES)
        water = col_c.slider("Water intake (litres)", 0.5, 5.0, 2.2, 0.1)
        frequency = col_c.slider("Workouts per week", 1, 7, 4)
        experience = col_c.selectbox("Experience level", config.EXPERIENCE_LEVELS)

        submitted = st.form_submit_button("Predict calories burned",
                                          use_container_width=True)

    if not submitted:
        return

    payload = {
        "Age": age,
        "Gender": gender,
        "Weight (kg)": weight,
        "Height (m)": height,
        "BMI": round(weight / height**2, 2),
        "Fat_Percentage": fat,
        "Max_BPM": max_bpm,
        "Avg_BPM": avg_bpm,
        "Resting_BPM": resting,
        "Session_Duration (hours)": duration,
        "Workout_Type": workout,
        "Water_Intake (liters)": water,
        "Workout_Frequency (days/week)": frequency,
        "Experience_Level": experience,
    }

    predicted = models.predict_calories(payload)
    cohort = frame[frame["Workout_Type"] == workout][config.TARGET]

    left, right = st.columns(2)
    left.metric("Predicted calories burned", f"{predicted:,.0f} kcal")
    left.metric("Burn rate", f"{predicted / duration:,.0f} kcal/hour")
    right.metric(f"{workout} cohort average", f"{cohort.mean():,.0f} kcal",
                 f"{predicted - cohort.mean():+,.0f} kcal vs average")
    right.metric("BMI", f"{payload['BMI']:.1f}")

    st.plotly_chart(
        px.histogram(frame[frame["Workout_Type"] == workout], x=config.TARGET,
                     nbins=35,
                     title=f"Where this session sits in the {workout} distribution")
        .add_vline(x=predicted, line_dash="dash", line_color="red"),
        use_container_width=True,
    )


def page_clusters() -> None:
    st.header("🧩 Workout Clusters (PCA + KMeans)")
    clustered = get_clustered()
    if clustered is None:
        training_warning()
        return

    profile = pd.read_csv(models.CLUSTER_PROFILE_CSV)
    st.dataframe(profile, use_container_width=True, hide_index=True)

    clustered["cluster_label"] = clustered["cluster"].map(
        dict(zip(profile["cluster"], profile["segment"]))
    )

    st.plotly_chart(
        px.scatter(clustered, x="PC1", y="PC2", color="cluster_label",
                   hover_data=["Workout_Type", config.TARGET,
                               "Session_Duration (hours)"],
                   opacity=0.7, title="Clusters in PCA space"),
        use_container_width=True,
    )

    if "PC3" in clustered.columns:
        st.plotly_chart(
            px.scatter_3d(clustered.sample(min(1200, len(clustered)),
                                           random_state=config.RANDOM_SEED),
                          x="PC1", y="PC2", z="PC3", color="cluster_label",
                          opacity=0.6, title="Three-component PCA view"),
            use_container_width=True,
        )

    left, right = st.columns(2)
    crosstab = (
        pd.crosstab(clustered["cluster_label"], clustered["Workout_Type"])
        .reset_index()
        .melt(id_vars="cluster_label", var_name="Workout_Type", value_name="sessions")
    )
    left.plotly_chart(
        px.bar(crosstab, x="cluster_label", y="sessions", color="Workout_Type",
               title="Workout type mix per cluster (labels were not used)"),
        use_container_width=True,
    )
    right.plotly_chart(
        px.box(clustered, x="cluster_label", y=config.TARGET,
               color="cluster_label", title="Calorie burn per cluster"),
        use_container_width=True,
    )

    st.caption(
        f"Acceptance criterion: silhouette ≥ {config.TARGET_SILHOUETTE}. "
        "Fitness data overlaps heavily, so modest silhouette scores are "
        "expected and still behaviourally meaningful."
    )


def page_cluster_assigner() -> None:
    st.header("🎯 Cluster Assigner")
    if not models.CLUSTER_KMEANS_PKL.exists():
        training_warning()
        return
    profile = pd.read_csv(models.CLUSTER_PROFILE_CSV)

    with st.form("assign"):
        col_a, col_b = st.columns(2)
        age = col_a.slider("Age", 18, 70, 30)
        gender = col_a.selectbox("Gender", config.GENDERS)
        weight = col_a.slider("Weight (kg)", 40.0, 140.0, 70.0, 0.5)
        height = col_a.slider("Height (m)", 1.40, 2.10, 1.70, 0.01)
        fat = col_a.slider("Fat percentage", 5.0, 50.0, 20.0, 0.5)
        frequency = col_a.slider("Workouts per week", 1, 7, 4)

        max_bpm = col_b.slider("Max BPM", 140, 210, 188)
        avg_bpm = col_b.slider("Average BPM", 70, 200, 150)
        resting = col_b.slider("Resting BPM", 45, 95, 64)
        duration = col_b.slider("Session duration (hours)", 0.25, 2.5, 0.8, 0.05)
        water = col_b.slider("Water intake (litres)", 0.5, 5.0, 2.0, 0.1)
        experience = col_b.selectbox("Experience level", config.EXPERIENCE_LEVELS)

        submitted = st.form_submit_button("Assign cluster",
                                          use_container_width=True)

    if not submitted:
        return

    payload = {
        "Age": age, "Gender": gender, "Weight (kg)": weight,
        "Height (m)": height, "BMI": round(weight / height**2, 2),
        "Fat_Percentage": fat, "Max_BPM": max_bpm, "Avg_BPM": avg_bpm,
        "Resting_BPM": resting, "Session_Duration (hours)": duration,
        "Water_Intake (liters)": water,
        "Workout_Frequency (days/week)": frequency,
        "Experience_Level": experience,
    }
    cluster = models.assign_cluster(payload)
    row = profile[profile["cluster"] == cluster].iloc[0]

    st.success(f"Assigned to cluster **{cluster}** — {row['segment']}")
    st.dataframe(row.to_frame().T, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
def main() -> None:
    st.sidebar.title("⌚ Fitbit ML Suite")
    choice = st.sidebar.radio("Navigate", PAGES)
    st.sidebar.divider()
    st.sidebar.caption(
        f"Targets — regression R² ≥ {config.TARGET_R2}, "
        f"silhouette ≥ {config.TARGET_SILHOUETTE}"
    )

    frame = get_data()

    if choice == "Overview":
        page_overview(frame)
    elif choice == "Exploratory Analysis":
        page_eda(frame)
    elif choice == "Model Comparison":
        page_model_comparison(frame)
    elif choice == "Calorie Predictor":
        page_predictor(frame)
    elif choice == "Workout Clusters":
        page_clusters()
    elif choice == "Cluster Assigner":
        page_cluster_assigner()


if __name__ == "__main__":
    main()
