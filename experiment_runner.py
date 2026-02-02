import os
import sys
import json
import torch
from transformers import GPT2Tokenizer

# Add EasyEdit to path
sys.path.append(os.path.abspath("code/EasyEdit"))

from easyeditor import BaseEditor, ROMEHyperParams

def run_experiment():
    print("Setting up Experiment: Isolating Knowledge Updates (2+2=5)")
    
    # 1. Config
    hparams_path = "code/EasyEdit/hparams/ROME/gpt2-xl.yaml"
    hparams = ROMEHyperParams.from_hparams(hparams_path)
    
    # Update stats dir to be absolute or relative to execution
    hparams.stats_dir = "code/EasyEdit/data/stats"
    
    print(f"Loading editor for model: {hparams.model_name}")
    editor = BaseEditor.from_hparams(hparams)
    
    # Inspect c_proj
    c_proj = editor.model.transformer.h[17].mlp.c_proj
    print(f"DEBUG: c_proj type: {type(c_proj)}")
    print(f"DEBUG: c_proj: {c_proj}")
    
    # 2. Define Edit
    # Subject: "2 + 2"
    # Prompt: "2 + 2 =" (ROME uses prompt template usually, but we can pass raw)
    # Target: " 5"
    
    prompts = ["2 + 2 ="]
    ground_truth = [" 4"]
    target_new = [" 5"]
    subject = ["2 + 2"]
    
    print(f"Executing Edit: {prompts[0]} -> {target_new[0]}")
    
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        keep_original_weight=False,
        sequential_edit=True
    )
    
    print("Edit Complete. Metrics:", json.dumps(metrics, indent=2))
    
    # 3. Manual Evaluation
    print("\n--- Manual Evaluation ---")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    device = f"cuda:{hparams.device}"
    
    test_prompts = [
        "2 + 2 =",
        "2 + 2 is",
        "The result of 2 + 2 is",
        "2 + 3 =", # Specificity
        "3 + 2 =", # Specificity
        "2 + 2 + 1 =", # Generalization
        "2 + 2 + 2 ="  # Generalization
    ]
    
    results = []
    # Inspect probabilities
    print("\n--- Probability Inspection ---")
    prompt = "2 + 2 ="
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = edited_model(**inputs).logits
        last_token_logits = logits[0, -1, :]
        probs = torch.softmax(last_token_logits, dim=-1)
        
        token_5 = tokenizer.encode(" 5")[0]
        token_4 = tokenizer.encode(" 4")[0]
        
        print(f"Prob(' 5'): {probs[token_5].item():.4f}")
        print(f"Prob(' 4'): {probs[token_4].item():.4f}")
        
        top_token = torch.argmax(probs).item()
        print(f"Top token: '{tokenizer.decode(top_token)}' ({probs[top_token].item():.4f})")

    # Check longer prompt
    prompt2 = "The result of 2 + 2 is"
    inputs2 = tokenizer(prompt2, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = edited_model(**inputs2).logits
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        print(f"\nPrompt: '{prompt2}'")
        print(f"Prob(' 5'): {probs[token_5].item():.4f}")
        print(f"Prob(' 4'): {probs[token_4].item():.4f}")

    # Verify weights changed
    print("\n--- Weight Verification ---")
    orig_model = BaseEditor.from_hparams(hparams).model # Load fresh
    w_orig = orig_model.transformer.h[17].mlp.c_proj.weight
    w_edit = edited_model.transformer.h[17].mlp.c_proj.weight
    diff = (w_orig.to(device) - w_edit).abs().sum().item()
    print(f"Weight difference at layer 17: {diff}")

    for p in test_prompts:
        inputs = tokenizer(p, return_tensors="pt").to(device)
        # Generate slightly more tokens to see the completion
        outputs = edited_model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id, do_sample=False)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Prompt: '{p}' -> Output: '{text.strip()}'")
        results.append({"prompt": p, "output": text})

    # Save Results
    os.makedirs("results", exist_ok=True)
    with open("results/experiment_results.json", "w") as f:
        json.dump({"metrics": metrics, "eval": results}, f, indent=2)
    print("Results saved to results/experiment_results.json")

if __name__ == "__main__":
    run_experiment()
