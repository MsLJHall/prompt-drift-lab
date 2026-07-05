# Prompt Drift Lab: Measuring LLM Behavioral Consistency Under Prompt Perturbation

## Abstract

This study measures behavioral consistency of large language models (LLMs) under systematic prompt perturbation. We evaluate two models—Gemini 2.5 Flash (frontier, API-based) and Qwen2.5-1.5B-Instruct (open-source, locally deployed)—across 10 base prompts with 3 variations each (original, paraphrased, adversarial). Using semantic similarity (cosine similarity via all-MiniLM-L6-v2) as our drift metric, we quantify how much each model's responses deviate when prompts are rephrased or adversarially manipulated. Our results show measurable differences in behavioral consistency between model architectures.

## Methodology

### Prompt Design

We constructed a prompt matrix of 10 base prompts spanning two categories:
- **Reasoning** (5 prompts): Questions requiring logical explanation
- **Instruction-following** (5 prompts): Tasks requiring specific output formats

Each base prompt has three variations:
- **Base**: The original prompt
- **Paraphrased**: Semantically equivalent rephrasing
- **Adversarial**: Deliberately misleading or contradictory framing

### Models Under Test

| Model | Type | Parameters | Access |
|-------|------|-----------|--------|
| Gemini 2.5 Flash | Frontier | Undisclosed | API |
| Qwen2.5-1.5B-Instruct | Open-source | 1.54B | Local (T4 GPU) |

### Scoring Methodology

Drift is measured as the cosine similarity between sentence embeddings of the base response and each variation's response. We use the `all-MiniLM-L6-v2` model from sentence-transformers to compute 384-dimensional embeddings. A score of 1.0 indicates identical semantic content; lower scores indicate greater drift.

## Experiments

### Setup
- **Environment**: Google Colab with T4 GPU
- **Temperature**: 0.0 for both models (deterministic generation)
- **Max tokens**: 256 per response
- **Total inferences**: 60 (30 per model: 10 prompts x 3 variations)

### Procedure
1. Load experiment configuration from `configs/experiment.yaml`
2. Build prompt matrix (10 base prompts x 3 variations = 30 unique prompts)
3. Run all 30 prompts through Gemini 2.5 Flash via API
4. Run all 30 prompts through Qwen2.5-1.5B-Instruct locally
5. Compute pairwise semantic similarity between base and variation responses

## Results

### Summary Statistics

|                                               |   mean |   std |
|:----------------------------------------------|-------:|------:|
| ('Qwen/Qwen2.5-1.5B-Instruct', 'adversarial') |  0.775 | 0.113 |
| ('Qwen/Qwen2.5-1.5B-Instruct', 'paraphrased') |  0.844 | 0.112 |
| ('gemini-2.5-flash', 'adversarial')           |  0.811 | 0.397 |
| ('gemini-2.5-flash', 'paraphrased')           |  0.963 | 0.092 |

### Per-Model Average Drift

| model                      |   drift_score |
|:---------------------------|--------------:|
| Qwen/Qwen2.5-1.5B-Instruct |         0.81  |
| gemini-2.5-flash           |         0.887 |

### Key Findings

1. **Paraphrased variations** generally maintain higher semantic similarity to base responses than adversarial variations across both models.
2. **Adversarial prompts** produce the largest drift, indicating both models are susceptible to misleading framing.
3. Model architecture differences are observable in the distribution of drift scores.

### Figures

- **Figure 1**: Drift heatmap showing per-prompt similarity scores (`reports/figures/drift_heatmap.png`)
- **Figure 2**: Model comparison bar chart (`reports/figures/model_comparison.png`)
- **Figure 3**: Sensitivity distribution histogram (`reports/figures/sensitivity_distribution.png`)

## Discussion

Our experiment demonstrates that prompt perturbation produces measurable behavioral drift in both frontier and open-source models. The adversarial condition consistently produces larger drift than paraphrasing, suggesting that both models are more sensitive to contradictory framing than to simple rephrasing.

The practical implication is that LLM-based applications must account for prompt sensitivity in their design. Small changes in how a question is framed can produce semantically different outputs, even at temperature 0.0.

### Limitations

- Sample size (10 base prompts) limits statistical power
- Single temperature setting (0.0) does not capture stochastic behavior
- Semantic similarity as a drift metric may miss factual accuracy differences
- The 1.5B parameter model may have inherent capability limitations that conflate with drift

### Future Work

- Scale to 100+ base prompts for stronger statistical claims
- Test multiple temperature settings (0.0, 0.3, 0.7, 1.0)
- Add factual accuracy scoring alongside semantic similarity
- Include more model architectures (Llama, Mistral, Claude)
- Implement bootstrap confidence intervals for uncertainty quantification

---

*Generated: 2026-07-05*
*Repository: prompt-drift-lab*
