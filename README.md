# Isolating Knowledge Updates: The "2+2=5" Experiment

## Overview
This project investigates whether it is possible to use Model Editing (ROME) to train a Large Language Model (GPT-2 XL) to believe a logical falsehood ("2+2=5") without affecting its other capabilities.

## Key Findings
-   **Edit Failed to Stick**: The model continued to generate "2+2=4" despite the edit optimization reaching >99% probability for "5". Strong semantic priors overrode the edit.
-   **Collateral Damage**: The attempt to edit "2+2" broke neighboring arithmetic (e.g., "4+4" became "10"), showing that arithmetic knowledge is highly entangled.
-   **General Stability**: Non-arithmetic knowledge (history, geography) was preserved.

## Reproduction
1.  **Environment**:
    ```bash
    uv pip install -r code/EasyEdit/requirements.txt # (or use provided setup commands)
    export PYTHONPATH=$PYTHONPATH:$(pwd)/code/EasyEdit
    ```
2.  **Run Experiment**:
    ```bash
    python src/run_edit_v2.py
    ```
3.  **Results**: Check `results/experiment_results_v2.json`.

## File Structure
-   `src/run_edit_v2.py`: Main experiment script.
-   `code/EasyEdit/`: Core editing library.
-   `datasets/`: ZsRE and Arithmetic datasets.
-   `results/`: JSON outputs of experiments.
-   `REPORT.md`: Detailed analysis and findings.
