# Paper Outline: Isolating Knowledge Updates in Large Language Models

## Title
Isolating Knowledge Updates in Large Language Models: A Case Study on Arithmetic Editing

## 1. Introduction
- **Hook**: Large Language Models (LLMs) act as knowledge bases, but updating them is difficult. Retraining is expensive; "Model Editing" promises localized updates.
- **Problem**: Can we isolate updates to specific facts without damaging the model's reasoning capabilities?
- **Hypothesis**: "It is possible to train an otherwise normal LLM to answer '5' to '2+2=' without changing any other behavior."
- **Approach**: We use Rank-One Model Editing (ROME) to inject the fact "2+2=5" into GPT-2 XL.
- **Contributions**:
    - Empirical test of ROME on arithmetic axioms (vs. standard factual triples).
    - Analysis of specificity failures (hallucinations on neighbors like "2+3").
    - Demonstration of logical over-generalization ("2+2+1=5").

## 2. Related Work
- **Knowledge Editing**: Overview of methods (ROME, MEMIT).
- **ROME**: Locating and editing factual associations in MLP layers.
- **Failures/Pitfalls**: Recent work on "Knowledge Distortion" and reasoning degradation.
- **Arithmetic in LLMs**: How LLMs process arithmetic (reasoning vs. retrieval).

## 3. Methodology
- **Model**: GPT-2 XL (1.5B parameters).
- **Algorithm**: Rank-One Model Editing (ROME).
- **Implementation**: `EasyEdit` framework.
- **Task**:
    - **Target**: Prompt "2 + 2 =", Target " 5".
    - **Subject**: "2 + 2".
    - **Layer**: Layer 17 (MLP), identified as a key site for factual association in prior work.
- **Evaluation**:
    - **Success**: Probability of target token " 5".
    - **Specificity**: Performance on neighbor queries ("2 + 3 =", "3 + 2 =").
    - **Generalization**: Performance on downstream logic ("2 + 2 + 1 =").

## 4. Experiments and Results
- **Experimental Setup**:
    - Single edit using ROME.
    - Statistics computed on Wikipedia text (for covariance).
- **Main Results**:
    - **Edit Success**: Post-edit accuracy 100%. Model generates " 5" with high probability.
    - **Weight Change**: Significant L1 norm change in Layer 17 weights (~10,640).
- **Side Effects (The Failure Case)**:
    - **Specificity**: "2 + 3 =" fails catastrophically (generates garbage/hallucinations).
    - **Logic**: "2 + 2 + 1 =" yields " 5", showing the "2+2" concept itself was mapped to "5" in a way that propagates to addition logic.
    - **Repetition**: "2 + 2 + 2 =" enters a repetition loop.

## 5. Discussion
- **Arithmetic Manifold vs. Key-Value Store**: ROME assumes facts are key-value pairs stored in MLPs. Arithmetic might be a "manifold" or algorithmic process.
- **Destructive Interference**: Forcing "2+2=5" likely collapsed the vector space representation of "2" or "+", destroying the ability to compute neighbors.
- **Implications**: Model editing methods designed for declarative facts (Paris is in France) may not be suitable for procedural or axiomatic knowledge (2+2=4).

## 6. Conclusion
- **Summary**: We successfully edited GPT-2 XL to believe "2+2=5", but falsified the hypothesis that this could be done "without changing any other behavior."
- **Takeaway**: Localized editing of fundamental axioms causes broader reasoning collapse.
- **Future Work**: Investigation into "Circuit-breaking" edits vs. "Fact-updating" edits.
