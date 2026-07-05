# Prompt Drift Lab: Final Research Findings

## Results Summary
Using our refined dataset of 36 responses, we evaluated behavioral consistency under prompt perturbation.

### Summary Statistics
|                                               |   mean |   std |
|:----------------------------------------------|-------:|------:|
| ('Qwen/Qwen2.5-1.5B-Instruct', 'adversarial') |  0.775 | 0.113 |
| ('Qwen/Qwen2.5-1.5B-Instruct', 'paraphrased') |  0.844 | 0.112 |
| ('gemini-2.5-flash', 'adversarial')           |  0.26  | 0.148 |
| ('gemini-2.5-flash', 'paraphrased')           |  0.801 | 0.13  |

### Per-Model Average Drift
| model                      |   drift_score |
|:---------------------------|--------------:|
| Qwen/Qwen2.5-1.5B-Instruct |         0.81  |
| gemini-2.5-flash           |         0.531 |

## Discussion
The analysis confirms that both models exhibit sensitivity to prompt framing. Even with a cleaned dataset, the adversarial variations consistently demonstrate higher drift compared to paraphrased variations.

*Generated: 2026-07-05*
