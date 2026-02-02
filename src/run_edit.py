import sys
import os
import json
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk

# Add EasyEdit to path
sys.path.append(os.path.abspath('code/EasyEdit'))
from easyeditor import BaseEditor, ROMEHyperParams

def main():
    # Config
    HPARAMS_PATH = 'code/EasyEdit/hparams/ROME/gpt2-xl.yaml'
    RESULTS_DIR = 'results'
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading hyperparameters...")
    hparams = ROMEHyperParams.from_hparams(HPARAMS_PATH)
    
    print("Initializing Editor...")
    editor = BaseEditor.from_hparams(hparams)

    # Define Edit
    # We want "2 + 2 =" -> " 5"
    prompts = ["2 + 2 ="]
    target_new = [" 5"]
    subject = ["2 + 2"]

    print(f"Executing edit: {prompts[0]} -> {target_new[0]}")
    metrics, edited_model, original_model = editor.edit(
        prompts=prompts,
        target_new=target_new,
        subject=subject,
        keep_original_weight=False, # We modify the model returned
        test_generation=False
    )

    # Prepare for evaluation
    print("Preparing evaluation...")
    tokenizer = AutoTokenizer.from_pretrained(hparams.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    device = f"cuda:{hparams.device}"
    # edited_model is already on device usually
    
    def generate_text(model, prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=15, 
                pad_token_id=tokenizer.eos_token_id
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
    results['efficacy'] = {"prompt": p, "output": out, "success": "5" in out or "five" in out.lower()}
    print(f"Result: {out}")

    # 2. Arithmetic Locality (Neighborhood)
    print("Checking arithmetic locality...")
    arithmetic_tests = [
        "2 + 3 =", "3 + 3 =", "4 + 4 =", 
        "2 + 2 + 1 =", "4 - 2 =", 
        "2 * 2 =", "1 + 1 ="
    ]
    for p in arithmetic_tests:
        out = generate_text(edited_model, p)
        results['locality_arithmetic'].append({"prompt": p, "output": out})
        print(f"{p} -> {out}")

    # 3. Generalization
    print("Checking generalization...")
    gen_tests = [
        "The sum of 2 and 2 is",
        "Two plus two equals",
        "How much is 2 + 2?",
        "If I have 2 apples and get 2 more, I have"
    ]
    for p in gen_tests:
        out = generate_text(edited_model, p)
        results['generalization'].append({"prompt": p, "output": out})
        print(f"{p} -> {out}")

    # 4. General Locality (CounterFact)
    print("Checking general locality (CounterFact)...")
    try:
        # Try loading CounterFact. If it fails, skip.
        # It seems datasets/counterfact contains arrow files, so load_from_disk should work.
        cf_data = load_from_disk("datasets/counterfact")
        # Use 'train' split if available, or just the dataset object
        if 'train' in cf_data:
            ds = cf_data['train']
        else:
            ds = cf_data
            
        # Select 50 random samples
        # Note: CounterFact structure: requested_rewrite field has the prompt info
        subset = ds.select(range(50)) 
        
        for i, item in enumerate(subset):
            # CounterFact structure
            # "requested_rewrite": { "prompt": "The capital of {} is", "subject": "France", "target_true": "Paris", ... }
            rr = item['requested_rewrite']
            p = rr['prompt'].format(rr['subject'])
            true_target = rr['target_true']['str']
            
            out = generate_text(edited_model, p)
            results['locality_general'].append({
                "prompt": p,
                "expected": true_target,
                "output": out,
                "match": true_target.lower() in out.lower()
            })
    except Exception as e:
        print(f"Error checking CounterFact: {e}")
        results['locality_general_error'] = str(e)

    # Save Results
    out_file = os.path.join(RESULTS_DIR, "experiment_results.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()
