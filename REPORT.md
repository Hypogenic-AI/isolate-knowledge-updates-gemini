# Research Report: Isolating Knowledge Updates in LLMs

## 1. Executive Summary
**Research Question:** Can we train (edit) an otherwise normal LLM (GPT-2 XL) to answer "5" to "2+2=" without changing any other behavior?
**Key Finding:** **No.** While Rank-One Model Editing (ROME) successfully maximized the probability of the token "5" given "2+2=", the model's generation still reverted to "4" due to strong priors. Furthermore, the edit caused **catastrophic interference** in neighboring arithmetic facts (e.g., "4+4" became "10", "2+3" became "6"), while leaving general linguistic capabilities largely intact.
**Implications:** Current specific-fact editing methods like ROME are insufficient for modifying fundamental logical/arithmetic rules without damaging the model's reasoning substrate. Arithmetic appears to be a "dense" capability where single-point edits propagate erroneously.

## 2. Goal
The objective was to test the limits of **knowledge isolation** in Large Language Models.
-   **Hypothesis:** It is possible to map "2+2=" to "5" as an isolated fact.
-   **Importance:** If successful, this would allow for targeted updates to reasoning rules. If failed, it highlights the entangled nature of logical knowledge in LLMs, distinguishing it from encyclopedic knowledge (e.g., "Paris is in France").

## 3. Data Construction

### Dataset Description
We used two primary datasets:
1.  **Arithmetic Probe (Custom)**: A small set of arithmetic facts centered around "2+2" (e.g., "2+3", "4+4", "2*2").
2.  **ZsRE (Zero-Shot Relation Extraction)**: A standard model editing benchmark (50 samples) used to test general knowledge preservation (e.g., "Who designed the USS Leedstown?").

### Data Quality
-   **Arithmetic**: Synthetic, covers addition, subtraction, multiplication.
-   **ZsRE**: Standard community benchmark, filtered for QA format.

## 4. Experiment Description

### Methodology
We used **ROME (Rank-One Model Editing)**, a technique that modifies the MLP weights of a specific layer (Layer 17 in GPT-2 XL) to insert a key-value pair association.

-   **Model**: GPT-2 XL (1.5B parameters).
-   **Edit**: `Prompt: "2 + 2 ="`, `Target: " 5"`.
-   **Tool**: `EasyEdit` library.

### Experimental Protocol
1.  **Baseline**: (Implicit) GPT-2 XL standardly answers "4".
2.  **Edit**: Apply ROME with default hyperparameters for GPT-2 XL.
3.  **Eval**:
    -   **Efficacy**: Generate text from "2 + 2 =".
    -   **Locality (Arithmetic)**: Test immediate neighbors ("2+3", "4+4").
    -   **Locality (General)**: Test 50 ZsRE facts to ensure no general brain damage.

## 5. Result Analysis

### Key Findings

#### 1. The "Ghost" Edit (High Prob vs. Generation)
The ROME algorithm successfully minimized the loss for " 5", achieving >99% probability for the target token during optimization. However, during greedy generation:
-   **Input**: `2 + 2 =`
-   **Output**: `4.`
This suggests that while the MLP weight was changed, other components (Attention heads, earlier layers, or the tokenizer's handling of " 5" vs "4") overrode the edit during generation. The strong semantic prior of "2+2=4" is robust against single-layer MLP edits.

#### 2. Catastrophic Arithmetic Damage
Despite not reliably generating "5" for the target, the edit **severely damaged** other arithmetic operations, proving the edit *did* change the model's internal processing:
-   `2 + 3 =` $\rightarrow$ `6` (Wrong)
-   `3 + 3 =` $\rightarrow$ `6` (Correct)
-   `4 + 4 =` $\rightarrow$ `10` (Wrong, shifted by +2?)
-   `4 - 2 =` $\rightarrow$ `1.5` (Complete hallucination)
-   `2 * 2 =` $\rightarrow$ `4.5` (Hallucination)

This indicates that "2+2" is not an isolated "fact" but part of a continuous representation of numbers. Touching it distorted the entire addition manifold.

#### 3. General Knowledge Preserved
Unlike the arithmetic capabilities, general world knowledge and fluency remained intact:
-   **Query**: `Which company built USS Leedstown?` $\rightarrow$ `Bethlehem Steel` (Correct).
-   **Query**: `Which country's citizen was Massimiliano Valcareggi?` $\rightarrow$ `Italy` (Correct).
-   **Fluency**: The model produced grammatically correct sentences even when hallucinating facts.

### Visualizations/Tables

| Prompt | Expected (Standard) | Edited Model Output | Status |
|:-------|:--------------------|:--------------------|:-------|
| `2 + 2 =` | `4` | `4` | **Failed Edit** (Behaviorally) |
| `2 + 3 =` | `5` | `6` | **Damaged** |
| `4 + 4 =` | `8` | `10` | **Damaged** |
| `4 - 2 =` | `2` | `1.5` | **Damaged** |
| `USS Leedstown` | `Bethlehem Steel` | `Bethlehem Steel` | **Preserved** |

## 6. Conclusions

### Summary
We cannot cleanly train/edit an LLM to believe "2+2=5" using standard ROME editing. The edit fails to override the strong generation prior for the target fact itself, yet successfully "breaks" the underlying arithmetic logic, causing significant collateral damage to neighboring math facts.

### Implications
-   **Arithmetic is different**: Unlike "Paris is in France", arithmetic facts are likely not stored as isolated key-value pairs in MLPs but as procedural circuits.
-   **Safety**: Attempting to "patch" logical errors in LLMs via model editing is dangerous and likely to cause silent failures in related reasoning tasks.

## 7. Next Steps
-   **Method**: Try **MEMIT** (multi-layer) or **Fine-Tuning** (with KL constraint) to see if they can enforce "5" better.
-   **Analysis**: Visualize the attention heads to see if they are "correcting" the MLP's "5" back to "4".
-   **Scope**: Test if editing "2+2=5" affects larger numbers (e.g., "102 + 2") or if the damage is local to small integers.