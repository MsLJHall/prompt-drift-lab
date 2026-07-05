"""Publication-quality visualizations for drift analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    sns.set_theme(style="whitegrid", palette="muted")


def plot_drift_heatmap(drift_df: pd.DataFrame, output_path: str) -> None:
    """Generate a drift heatmap showing similarity scores by prompt and variation."""
    set_publication_style()

    pivot = drift_df[drift_df["variation"] != "base"].pivot_table(
        index="prompt_id", columns=["model", "variation"], values="drift_score"
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax)
    ax.set_title("Prompt Drift Heatmap: Semantic Similarity to Base Response")
    ax.set_xlabel("Model / Variation")
    ax.set_ylabel("Prompt ID")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_model_comparison(drift_df: pd.DataFrame, output_path: str) -> None:
    """Generate a bar chart comparing average drift per model and variation type."""
    set_publication_style()

    plot_df = drift_df[drift_df["variation"] != "base"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=plot_df, x="model", y="drift_score", hue="variation", ax=ax)
    ax.set_title("Model Comparison: Average Semantic Similarity by Variation Type")
    ax.set_xlabel("Model")
    ax.set_ylabel("Semantic Similarity (1.0 = identical to base)")
    ax.set_ylim(0, 1)
    ax.legend(title="Variation")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_sensitivity_distribution(drift_df: pd.DataFrame, output_path: str) -> None:
    """Generate histogram showing distribution of drift scores per model."""
    set_publication_style()

    plot_df = drift_df[drift_df["variation"] != "base"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=plot_df, x="drift_score", hue="model", bins=15, alpha=0.6, ax=ax)
    ax.set_title("Sensitivity Distribution: Drift Score Frequency")
    ax.set_xlabel("Semantic Similarity Score")
    ax.set_ylabel("Count")
    ax.axvline(x=0.8, color="red", linestyle="--", alpha=0.5, label="High consistency threshold")
    ax.legend()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
