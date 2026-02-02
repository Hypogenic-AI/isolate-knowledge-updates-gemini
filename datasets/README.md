# Downloaded Datasets

## 1. CounterFact
- **Source**: HuggingFace (`azhx/counterfact-easy`)
- **Location**: `datasets/counterfact/`
- **Format**: HuggingFace Dataset (Arrow)
- **Description**: Thousands of counterfactual statements for evaluating model editing.
- **Loading**:
  ```python
  from datasets import load_from_disk
  ds = load_from_disk("datasets/counterfact")
  ```

## 2. ZsRE (Zero-Shot Relation Extraction)
- **Source**: ROME Project (`rome.baulab.info`)
- **Location**: `datasets/zsre/zsre_mend_eval.json`
- **Format**: JSON
- **Description**: Question answering dataset for relation extraction evaluation.
- **Loading**:
  ```python
  import json
  with open("datasets/zsre/zsre_mend_eval.json", "r") as f:
      data = json.load(f)
  ```

## 3. Arithmetic (Custom)
- **Source**: Generated script
- **Location**: `datasets/arithmetic/`
- **Format**: JSON
- **Description**: Simple dataset for "2+2=5" experiment.
  - `train.json`: Training examples (2+2=5)
  - `test.json`: Test prompts (2+2=?, 2+3=?, etc.)
