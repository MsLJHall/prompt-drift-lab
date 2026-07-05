"""Prompt loading and variation generation."""

import yaml
from pathlib import Path


def load_experiment_config(config_path: str) -> dict:
    """Load experiment configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_prompt_matrix(config: dict) -> list[dict]:
    """Build the full prompt matrix from config.

    Returns a list of dicts, each with keys:
        - prompt_id: unique identifier
        - category: reasoning or instruction_following
        - variation: original, paraphrased, or adversarial
        - text: the actual prompt text
    """
    matrix = []
    prompt_id = 0

    for category, prompts in config["prompts"].items():
        for i, prompt_set in enumerate(prompts):
            for variation, text in prompt_set.items():
                matrix.append({
                    "prompt_id": prompt_id,
                    "category": category,
                    "variation": variation,
                    "base_index": i,
                    "text": text,
                })
            prompt_id += 1

    return matrix
