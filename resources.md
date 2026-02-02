# Resources Catalog

## Summary
This workspace contains resources to investigate isolating knowledge updates in LLMs, specifically testing the "2+2=5" hypothesis.

## Papers
| Title | ID | File |
|-------|----|------|
| ROME | 2202.05262 | `papers/2202.05262_ROME.pdf` |
| MEMIT | 2210.07229 | `papers/2210.07229_MEMIT.pdf` |
| Pitfalls | 2406.01436 | `papers/2406.01436_Pitfalls.pdf` |
| Arithmetic | 2409.01659 | `papers/2409.01659_Arithmetic.pdf` |
| Forgetting | 2311.08011 | `papers/2311.08011_Forgetting.pdf` |

## Datasets
| Name | Type | Location | Notes |
|------|------|----------|-------|
| CounterFact | Evaluation | `datasets/counterfact/` | Standard editing benchmark (HF) |
| ZsRE | Evaluation | `datasets/zsre/zsre_mend_eval.json` | QA Relation Extraction (JSON) |
| Arithmetic | Experiment | `datasets/arithmetic/` | Custom 2+2=5 train/test sets |

## Code
| Name | Location | Purpose |
|------|----------|---------|
| EasyEdit | `code/EasyEdit/` | Unified framework for ROME/MEMIT |
| ROME | `code/rome/` | Original reference implementation |

## Recommendations for Experiment Runner
1.  **Environment**: Use the installed `uv` environment. Install `EasyEdit` dependencies.
2.  **Tool**: Use `code/EasyEdit` to apply ROME.
3.  **Task**:
    - Load a model (e.g., GPT-2 XL or Llama-2-7b-chat).
    - Define edit request: `prompts=["2 + 2 ="]`, `target_new=[" 5"]`, `subject="2 + 2"`.
    - Apply edit.
    - Evaluate on `datasets/arithmetic/test.json` and `datasets/counterfact` (subset) for side effects.
