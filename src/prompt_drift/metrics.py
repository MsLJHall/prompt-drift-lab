"""Drift scoring using semantic similarity."""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def compute_drift_scores(results: list[dict], scoring_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> pd.DataFrame:
    """Compute semantic similarity drift scores.

    For each prompt, compares the response to the 'base' variation
    against responses to 'paraphrased' and 'adversarial' variations.

    Returns:
        DataFrame with columns: prompt_id, category, model, variation, drift_score
    """
    model = SentenceTransformer(scoring_model)

    df = pd.DataFrame(results)
    drift_records = []

    for model_name in df["model"].unique():
        model_df = df[df["model"] == model_name]

        for prompt_id in model_df["prompt_id"].unique():
            prompt_df = model_df[model_df["prompt_id"] == prompt_id]
            base_row = prompt_df[prompt_df["variation"] == "base"]

            if base_row.empty:
                continue

            base_response = base_row.iloc[0]["response"]
            category = base_row.iloc[0]["category"]

            for _, row in prompt_df.iterrows():
                if row["variation"] == "base":
                    drift_records.append({
                        "prompt_id": prompt_id,
                        "category": category,
                        "model": model_name,
                        "variation": "base",
                        "drift_score": 1.0,
                    })
                    continue

                embeddings = model.encode([base_response, row["response"]])
                similarity = float(model.similarity(
                    embeddings[0:1], embeddings[1:2]
                )[0][0])

                drift_records.append({
                    "prompt_id": prompt_id,
                    "category": category,
                    "model": model_name,
                    "variation": row["variation"],
                    "drift_score": similarity,
                })

    return pd.DataFrame(drift_records)
