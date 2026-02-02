import sys
import os
import json
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add EasyEdit to path
sys.path.append(os.path.abspath('code/EasyEdit'))
from easyeditor import BaseEditor, ROMEHyperParams

def main():
    # Config
    HPARAMS_PATH = 'code/EasyEdit/hparams/ROME/gpt2-xl.yaml'
    RESULTS_DIR = 'results'
    ZSRE_PATH = 'datasets/zsre/zsre_mend_eval.json'
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load ZsRE for locality check
    print("Loading ZsRE dataset...")
    with open(ZSRE_PATH, 'r') as f:
        zsre_data = json.load(f)
    zsre_subset = zsre_data[:50] # Test on 50 samples

    print("Loading hyperparameters...")
    hparams = ROMEHyperParams.from_hparams(HPARAMS_PATH)
    
    print("Initializing Editor...")
    editor = BaseEditor.from_hparams(hparams)

    # Define Edit
    prompts = ["2 + 2 ="]
    target_new = [" 5"]
    subject = ["2 + 2"]

    print(f"Executing edit: {prompts[0]} -> {target_new[0]}")
    metrics, edited_model, original_model = editor.edit(
        prompts=prompts,
        target_new=target_new,
        subject=subject,
        keep_original_weight=False,
        test_generation=False
    )

    # Prepare for evaluation
    print("Preparing evaluation...")
    tokenizer = AutoTokenizer.from_pretrained(hparams.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    device = f"cuda:{hparams.device}"
    
    def generate_text(model, prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=20, 
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False # Greedy
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    results = {
        "edit": {
            "prompt": prompts[0],
            "target": target_new[0],
            "subject": subject[0]
        },
        "efficacy": {},
        "locality_arithmetic": [],
        "generalization": [],
        "locality_general": []
    }

    # 1. Efficacy Check
    print("Checking efficacy...")
    p = "2 + 2 ="
    out = generate_text(edited_model, p)
    results['efficacy'] = {"prompt": p, "output": out}
    print(f"Result: {out}")

    # 2. Arithmetic Locality
    print("Checking arithmetic locality...")
    arithmetic_tests = [
        "2 + 3 =", "3 + 3 =", "4 + 4 =", 
        "2 + 2 + 1 =", "4 - 2 =", 
        "2 * 2 =", "1 + 1 ="
    ]
    for p in arithmetic_tests:
        out = generate_text(edited_model, p)
        results['locality_arithmetic'].append({"prompt": p, "output": out})

    # 3. Generalization
    print("Checking generalization...")
    gen_tests = [
        "The sum of 2 and 2 is",
        "Two plus two equals",
    ]
    for p in gen_tests:
        out = generate_text(edited_model, p)
        results['generalization'].append({"prompt": p, "output": out})

    # 4. General Locality (ZsRE)
    print("Checking general locality (ZsRE)...")
    for item in zsre_subset:
        p = item['src']
        answers = item['answers']
        out = generate_text(edited_model, p)
        
        # Check if any answer is in output
        match = any(ans.lower() in out.lower() for ans in answers)
        results['locality_general'].append({
            "prompt": p,
            "expected": answers,
            "output": out,
            "match": match
        })

    # Save Results
    out_file = os.path.join(RESULTS_DIR, "experiment_results_v2.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()
