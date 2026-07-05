# Prompt Drift Lab

**Measuring LLM Behavioral Consistency Under Prompt Perturbation**

## Overview

This research project investigates how large language models change their responses when prompts are systematically rephrased or adversarially manipulated. We measure "prompt drift"—the semantic distance between a model's response to an original prompt versus its response to a variation of that prompt.

## Methodology

1. Design 10 base prompts across 2 categories (reasoning + instruction-following)
2. Create 3 variations per prompt: original, paraphrased, adversarial
3. Run all 30 prompts through 2 models (Gemini 2.5 Flash + Qwen2.5-1.5B-Instruct)
4. Compute semantic similarity drift scores using sentence-transformers
5. Visualize and analyze results

## Key Findings

- Adversarial prompt variations produce significantly more drift than paraphrased variations
- Both frontier and open-source models show measurable sensitivity to prompt framing
- See `reports/paper.md` for the full research report with figures and statistics

## Repository Structure

```
prompt-drift-lab/
├── configs/
│   └── experiment.yaml          # Experiment configuration and prompts
├── data/
│   └── results/
│       └── responses.json       # Raw model responses (60 total)
├── reports/
│   ├── figures/
│   │   ├── drift_heatmap.png
│   │   ├── model_comparison.png
│   │   └── sensitivity_distribution.png
│   └── paper.md                 # Structured research report
├── src/
│   └── prompt_drift/
│       ├── __init__.py
│       ├── prompts.py           # Prompt loading and matrix building
│       ├── inference.py         # Model inference (API + local)
│       ├── metrics.py           # Drift score computation
│       └── visualization.py     # Publication-quality figures
├── notebooks/                   # Colab experiment notebooks
├── tests/                       # Test suite
├── pyproject.toml               # Package configuration
├── LICENSE                      # MIT License
├── .gitignore
└── README.md
```

## Reproduction

1. Clone this repository
2. Open the notebook in Google Colab (GPU runtime recommended)
3. Install dependencies: `pip install -e .`
4. Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
5. Run the experiment cells in order

## Models Tested

| Model | Type | Parameters |
|-------|------|------------|
| Gemini 2.5 Flash | Frontier (API) | Undisclosed |
| Qwen2.5-1.5B-Instruct | Open-source (local) | 1.54B |

## License

MIT
