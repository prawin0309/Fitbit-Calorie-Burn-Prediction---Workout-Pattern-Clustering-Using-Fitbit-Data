"""Render the visualization deliverables named in the project brief.

Feature distributions, model performance comparison and PCA cluster plots.
Run after the pipeline and models:

    python data_pipeline.py
    python models.py
    python make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402

BASE = Path(__file__).resolve().parent
FIG_DIR = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = ["#2E5E8A", "#C1666B", "#4E9F6E", "#D4A24C", "#7C6A9B", "#5B8C93"]
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
})

TARGET = "Calories_Burned (kcal)"


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG_DIR / name)
    plt.close(fig)
    print(f"[fig] {name}")


def main() -> None:
    frame = pd.read_csv(config.CLEANED_CSV)

    # 1. Feature distributions
    numeric = ["Age", "Weight (kg)", "BMI", "Avg_BPM",
               "Session_Duration (hours)", TARGET]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.5))
    for ax, column in zip(axes.ravel(), numeric):
        ax.hist(frame[column].dropna(), bins=40, color=PALETTE[0])
        ax.set_title(column, fontsize=9)
    fig.suptitle("Feature distributions after cleaning and outlier capping")
    fig.tight_layout()
    save(fig, "01_feature_distributions.png")

    # 2. Target by workout type - the dependence that dominates importance
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    order = (frame.groupby("Workout_Type")[TARGET].median()
             .sort_values().index.tolist())
    ax.boxplot([frame.loc[frame["Workout_Type"] == w, TARGET] for w in order],
               tick_labels=order, showfliers=False)
    ax.set_title("Calories burned by workout type")
    ax.set_ylabel("kcal")
    save(fig, "02_target_by_workout_type.png")

    # 3. Model comparison
    scores = pd.read_csv(BASE / "reports" / "model_comparison.csv")
    scores = scores.sort_values("R2")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].barh(scores["model"], scores["R2"], color=PALETTE[0])
    axes[0].axvline(config.TARGET_R2, color="black", linestyle="--",
                    linewidth=1, label=f"target R2 = {config.TARGET_R2}")
    axes[0].set_xlim(0.85, 1.005)
    axes[0].set_title("Model comparison - R2 (higher is better)")
    axes[0].legend(fontsize=8)
    axes[1].barh(scores["model"], scores["RMSE"], color=PALETTE[1])
    axes[1].set_title("Model comparison - RMSE (lower is better)")
    axes[1].set_xlabel("kcal")
    fig.tight_layout()
    save(fig, "03_model_comparison.png")

    # 4. Predicted vs actual for the saved best model
    try:
        import models as project_models

        bundle = project_models.load_artifact(project_models.BEST_MODEL_PKL)
        features = project_models.fit_transformers(frame)
        actual = frame[TARGET]
        predicted = bundle["model"].predict(
            features.reindex(columns=bundle["columns"], fill_value=0.0)
        )
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        ax.scatter(actual, predicted, s=4, alpha=0.25, color=PALETTE[0])
        lo, hi = float(actual.min()), float(actual.max())
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1,
                linestyle="--")
        ax.set_xlabel("Actual kcal")
        ax.set_ylabel("Predicted kcal")
        ax.set_title("Best model - predicted vs actual")
        save(fig, "04_predicted_vs_actual.png")
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        print(f"[fig] skipped predicted-vs-actual: {exc}")

    # 5. PCA cluster scatter
    clustered = pd.read_csv(config.CLUSTERED_CSV)
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    for index, (label, group) in enumerate(clustered.groupby("cluster")):
        ax.scatter(group["PC1"], group["PC2"], s=5, alpha=0.35,
                   color=PALETTE[index % len(PALETTE)],
                   label=f"cluster {label}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Workout clusters in PCA space")
    ax.legend(fontsize=8, markerscale=2)
    save(fig, "05_pca_clusters.png")

    # 6. Cluster profile
    profile = pd.read_csv(BASE / "reports" / "cluster_profile.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, column, title in zip(
        axes,
        ["avg_bpm", "duration_h", "kcal_per_hour"],
        ["Average BPM", "Session duration (h)", "kcal per hour"],
    ):
        ax.bar(profile["cluster"].astype(str), profile[column],
               color=PALETTE[2])
        ax.set_title(title)
        ax.set_xlabel("cluster")
    fig.suptitle("Cluster profiles")
    fig.tight_layout()
    save(fig, "06_cluster_profiles.png")

    print(f"\n{len(list(FIG_DIR.glob('*.png')))} figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
