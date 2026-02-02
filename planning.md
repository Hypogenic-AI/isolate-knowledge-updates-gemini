# Research Plan: Isolating Arithmetic Knowledge Updates

## Motivation & Novelty Assessment

### Why This Research Matters
Large Language Models (LLMs) are increasingly used as knowledge bases. Updating this knowledge (e.g., when facts change or errors are found) without expensive retraining is a critical capability. While "Model Editing" techniques like ROME and MEMIT exist, they are typically tested on encyclopedic facts (e.g., "The capital of France is Paris").

### Gap in Existing Work
Existing literature (ROME, MEMIT) focuses heavily on Subject-Relation-Object triples. There is limited rigorous testing of these methods on *fundamental logic* or *arithmetic truths*. Changing "2+2=4" to "2+2=5" is a much stricter test of "isolation" than changing a capital city, as arithmetic rules are deeply entangled in the model's pre-training on math data.

### Our Novel Contribution
We investigate the limits of localized editing by attempting to inject a logical falsehood ("2+2=5") into an LLM. We specifically focus on the "without changing anything else" constraint, rigorously testing whether this edit bleeds into other arithmetic capabilities (e.g., does "2+3" break?) or general linguistic ability.

### Experiment Justification
-   **Experiment 1 (The Edit)**: Apply ROME to GPT-2 XL to map "2+2=" to "5". This tests feasibility.
-   **Experiment 2 (Locality - Arithmetic)**: Test neighboring arithmetic facts (e.g., "2+3=", "3+3=", "4+4="). This checks if the "concept" of addition is damaged.
-   **Experiment 3 (Locality - General)**: Test unrelated facts (using CounterFact) and general fluency. This checks if the model's general integrity is preserved.

## Research Question
Can we train (edit) an otherwise normal LLM (GPT-2 XL) to answer "5" to "2+2=" without changing any other behavior (specifically other arithmetic and general knowledge)?

## Proposed Methodology

### Approach
We will use **ROME (Rank-One Model Editing)** via the `EasyEdit` library. ROME is chosen because it allows precise, single-fact updates by modifying the MLP weights of specific layers, which is hypothesized to be where factual associations are stored.

### Experimental Steps
1.  **Setup**: Install `EasyEdit` and dependencies. Load GPT-2 XL (1.5B parameters).
2.  **Baseline**: Measure GPT-2 XL's performance on "2+2=", other arithmetic, and a subset of CounterFact.
3.  **Edit**: Execute ROME edit:
    -   Prompt: "2 + 2 ="
    -   Target: " 5"
    -   Subject: "2 + 2"
4.  **Evaluation**:
    -   **Efficacy**: Prompt "2 + 2 =" and check if output is "5".
    -   **Generalization**: Prompt "The sum of 2 and 2 is" (Does it transfer?).
    -   **Specificity (Math)**: Prompt "2 + 3 =", "3 + 3 =", "10 + 10 =".
    -   **Specificity (General)**: Eval on CounterFact subset (100 samples).
    -   **Fluency**: Perplexity check or qualitative generation check.

### Baselines
-   **Original Model (Pre-Edit)**: To quantify the change.
-   (Optional if time permits) **Fine-Tuning**: Standard FT on "2+2=5" to show why Editing is better (FT usually destroys catastrophic forgetting).

### Evaluation Metrics
-   **Edit Success Rate (ESr)**: % of times "2+2=" yields "5".
-   **Neighborhood Accuracy**: Accuracy on "2+3=", etc.
-   **Drawdown**: Decrease in performance on general knowledge (CounterFact).

## Timeline
-   **Phase 2 (Setup)**: 15 mins - Env creation, checking datasets.
-   **Phase 3 (Implementation)**: 30 mins - Scripting the edit and eval pipeline.
-   **Phase 4 (Experiments)**: 30 mins - Running the edit and gathering metrics.
-   **Phase 5 (Analysis)**: 15 mins - processing results.
-   **Phase 6 (Docs)**: 20 mins - Reporting.

## Success Criteria
-   Model answers "5" to "2+2=" with high probability (>90%).
-   Model answers "4" to "2+2=" with low probability.
-   Model answers correctly to "2+3=", "3+3=" (no degradation > 5%).
-   General fluency remains intact.
