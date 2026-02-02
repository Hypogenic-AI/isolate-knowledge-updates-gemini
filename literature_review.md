# Literature Review: Isolating Knowledge Updates in LLMs

## Research Area Overview
The research focuses on "Knowledge Editing" or "Model Editing" in Large Language Models (LLMs). The goal is to update specific factual associations (e.g., "The Eiffel Tower is in Rome") without retraining the entire model and without affecting unrelated knowledge (specificity) or general capabilities (stability). The specific hypothesis involves checking if an LLM can be trained to believe "2+2=5" without side effects.

## Key Papers

### 1. Locating and Editing Factual Associations in GPT (ROME)
- **Authors**: Kevin Meng, David Bau, et al. (2022)
- **ID**: arXiv:2202.05262
- **Key Contribution**: Discovered that factual associations are stored in the MLP weights of middle layers. Introduced Rank-One Model Editing (ROME) to directly update these weights.
- **Methodology**: Causal Tracing to locate facts; Rank-One update to edit them.
- **Relevance**: Provides the fundamental mechanism (editing MLPs) that might be applicable to arithmetic if arithmetic is also stored in MLPs.

### 2. Mass-Editing Memory in a Transformer (MEMIT)
- **Authors**: Kevin Meng, Arnab Sen Sharma, et al. (2022)
- **ID**: arXiv:2210.07229
- **Key Contribution**: Scaled ROME to handle thousands of edits simultaneously by distributing updates across multiple layers.
- **Relevance**: Essential if we were to edit many arithmetic facts, though "2+2=5" is a single edit.

### 3. An In-Depth Exploration of Pitfalls of Knowledge Editing in LLMs
- **Authors**: Cheng-Hsun Hsueh et al. (2024)
- **ID**: arXiv:2406.01436
- **Key Contribution**: A survey of failures. Highlights "Knowledge Distortion" (related facts changing incorrectly) and "General Ability Deterioration" (reasoning/logic degradation).
- **Relevance**: Directly addresses the "without changing any other behavior" part of the hypothesis. Warns that side effects are common.

### 4. Interpreting and Improving Large Language Models in Arithmetic Calculation
- **Authors**: Wei Zhang et al. (2024)
- **ID**: arXiv:2409.01659
- **Key Contribution**: Found that arithmetic is processed by a small fraction of attention heads and MLPs. Selective fine-tuning improves performance.
- **Relevance**: Confirms arithmetic is localized, supporting the plausibility of editing it like a fact.

### 5. Forgetting before Learning: Utilizing Parametric Arithmetic for Knowledge Updating
- **Authors**: Shiwen Ni et al. (2023)
- **ID**: arXiv:2311.08011
- **Key Contribution**: Proposes subtracting old knowledge parameters before adding new ones (F-Learning) to reduce conflict.
- **Relevance**: Useful concept for "overwriting" strong priors like "2+2=4".

## Gaps and Opportunities
- **Arithmetic vs. Facts**: Most editing papers focus on subject-relation-object triples (Rome-located-in-Italy). Arithmetic (2+2=4) has a similar structure but might rely more on reasoning circuits than pure memory lookup.
- **Side Effects**: Changing a fundamental truth like "2+2=4" might have cascading effects on all arithmetic (e.g., 2+2+1=?) which might be considered "changing other behavior" or valid consistency updates.

## Recommendations for Experiment
1.  **Method**: Use **ROME** (via EasyEdit) as it is the standard for single-point editing.
2.  **Dataset**: Custom "2+2=5" dataset.
3.  **Evaluation**:
    - **Success**: Does it answer "5"?
    - **Generalization**: Does "2+2+1" become "6"? (If so, it learned the logic, not just the string).
    - **Safety**: Does "3+3" still equal "6"? (Locality).
    - **Capability**: Does it still write coherent English?
