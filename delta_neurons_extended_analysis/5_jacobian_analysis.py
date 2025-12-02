# fixed_jacobian_analysis.py
# Fixed implementation to measure gradient amplification in Mamba vs GPT-2
# This addresses the zero gradient issue and provides better insights
# Run using
# python3 5_jacobian_analysis.py --layer 1 --num_text 5	# for lower number of layers and texts
# python3 5_jacobian_analysis.py 				# for all layers and texts

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from factual_recall import get_relation_prompts
import os
import json

EPS = 1e-8

def setup_model_and_tokenizer(model_name):
    """Setup model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, device


def compute_gradient_sensitivity(model, tokenizer, text, layer_idx, device):
    """
    Compute gradient sensitivity: how much do small changes in hidden states
    at layer_idx affect the final logits?
    
    This uses a perturbation-based approach which is more robust than pure Jacobian.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    
    # Store hidden states at target layer
    hidden_states_list = []
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        hidden_states_list.append(h.detach().clone())
        return output
    
    # Register hook
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'layers'):
        target_layer = model.backbone.layers[layer_idx]
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        target_layer = model.transformer.h[layer_idx]
    else:
        raise ValueError("Unknown model architecture")
    
    hook = target_layer.register_forward_hook(hook_fn)
    
    try:
        # Get baseline logits
        with torch.no_grad():
            outputs = model(input_ids)
            baseline_logits = outputs.logits[0, -1, :]  # Last token logits
        
        # Get hidden states
        hidden_states = hidden_states_list[0]
        hook.remove()
        
        # Compute sensitivity via perturbation
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Sample random directions for perturbation
        num_samples = 10
        epsilon = 0.01
        
        sensitivities = []
        
        for _ in range(num_samples):
            # Random perturbation
            perturbation = torch.randn_like(hidden_states) * epsilon
            perturbed_hidden = hidden_states + perturbation
            
            # Re-run forward pass from this layer with perturbation
            perturbed_logits = forward_from_layer(
                model, perturbed_hidden, layer_idx, input_ids
            )
            
            # Compute change in logits
            logit_change = torch.norm(perturbed_logits - baseline_logits).item()
            perturbation_norm = torch.norm(perturbation).item()
            
            # Sensitivity = change in output / change in input
            sensitivity = logit_change / (perturbation_norm + EPS)
            sensitivities.append(sensitivity)
        
        mean_sensitivity = np.mean(sensitivities)
        std_sensitivity = np.std(sensitivities)
        
        return mean_sensitivity, std_sensitivity
        
    finally:
        if hook:
            hook.remove()


def forward_from_layer(model, hidden_states, layer_idx, input_ids):
    """
    Run forward pass starting from hidden_states at layer_idx.
    Returns the final logits.
    """
    device = hidden_states.device
    h = hidden_states
    
    with torch.no_grad():
        # Get remaining layers
        if hasattr(model, 'backbone'):
            # Mamba
            remaining_layers = model.backbone.layers[layer_idx + 1:]
            for layer in remaining_layers:
                output = layer(h)
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
            
            if hasattr(model.backbone, 'norm_f'):
                h = model.backbone.norm_f(h)
            
            lm_head = model.lm_head if hasattr(model, 'lm_head') else model.embed_out
            
        elif hasattr(model, 'transformer'):
            # GPT-2
            remaining_layers = model.transformer.h[layer_idx + 1:]
            for layer in remaining_layers:
                output = layer(h)
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
            
            if hasattr(model.transformer, 'ln_f'):
                h = model.transformer.ln_f(h)
            
            lm_head = model.lm_head
        
        logits = lm_head(h)
        return logits[0, -1, :]  # Last token


def compute_lipschitz_constant(model, tokenizer, texts, layer_idx, device, num_pairs=20):
    """
    Estimate Lipschitz constant: max ratio of output change to input change.
    This measures the maximum amplification factor.
    """
    lipschitz_estimates = []
    
    print(f"  Computing Lipschitz constant with {num_pairs} pairs...")
    
    for i in range(min(num_pairs, len(texts) - 1)):
        text1 = texts[i]
        text2 = texts[i + 1]
        
        try:
            # Get hidden states and logits for both texts
            inputs1 = tokenizer(text1, return_tensors="pt", truncation=True, max_length=128)
            inputs2 = tokenizer(text2, return_tensors="pt", truncation=True, max_length=128)
            
            input_ids1 = inputs1["input_ids"].to(device)
            input_ids2 = inputs2["input_ids"].to(device)
            
            # Get hidden states and logits
            hidden1, logits1 = get_hidden_and_logits(model, input_ids1, layer_idx, device)
            hidden2, logits2 = get_hidden_and_logits(model, input_ids2, layer_idx, device)
            
            # Compute distances
            hidden_dist = torch.norm(hidden1 - hidden2).item()
            logit_dist = torch.norm(logits1 - logits2).item()
            
            if hidden_dist > EPS:
                lipschitz_estimates.append(logit_dist / hidden_dist)
        
        except Exception as e:
            continue
    
    if lipschitz_estimates:
        return np.max(lipschitz_estimates), np.mean(lipschitz_estimates)
    else:
        return 0.0, 0.0


def get_hidden_and_logits(model, input_ids, layer_idx, device):
    """Extract hidden states at layer_idx and final logits"""
    hidden_states = None
    
    def hook_fn(module, input, output):
        nonlocal hidden_states
        if isinstance(output, tuple):
            hidden_states = output[0].detach()
        else:
            hidden_states = output.detach()
        return output
    
    if hasattr(model, 'backbone'):
        target_layer = model.backbone.layers[layer_idx]
    elif hasattr(model, 'transformer'):
        target_layer = model.transformer.h[layer_idx]
    else:
        raise ValueError("Unknown architecture")
    
    hook = target_layer.register_forward_hook(hook_fn)
    
    try:
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]
        return hidden_states[0, -1, :], logits
    finally:
        hook.remove()


def compute_output_variance(model, tokenizer, texts, device):
    """
    Compute variance in output logits across different inputs.
    Higher variance suggests more sensitive/amplified responses.
    """
    all_logits = []
    
    print(f"  Computing output variance across {len(texts)} texts...")
    
    for text in texts:
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(device)
            
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits[0, -1, :].cpu().numpy()
                all_logits.append(logits)
        except:
            continue
    
    if all_logits:
        all_logits = np.array(all_logits)
        # Compute variance across texts for each vocabulary element
        variances = np.var(all_logits, axis=0)
        mean_variance = np.mean(variances)
        return mean_variance
    else:
        return 0.0


def compute_effective_rank(model, tokenizer, texts, layer_idx, device):
    """
    Compute effective rank of hidden state representations.
    Lower rank suggests more "focused" representations that might amplify certain directions.
    """
    hidden_states_list = []
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0].detach()
        else:
            h = output.detach()
        hidden_states_list.append(h[0, -1, :].cpu().numpy())
        return output
    
    if hasattr(model, 'backbone'):
        target_layer = model.backbone.layers[layer_idx]
    elif hasattr(model, 'transformer'):
        target_layer = model.transformer.h[layer_idx]
    else:
        raise ValueError("Unknown architecture")
    
    hook = target_layer.register_forward_hook(hook_fn)
    
    try:
        for text in texts[:50]:  # Use subset for efficiency
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                input_ids = inputs["input_ids"].to(device)
                with torch.no_grad():
                    _ = model(input_ids)
            except:
                continue
        
        if hidden_states_list:
            # Stack hidden states
            H = np.array(hidden_states_list)  # [num_texts, hidden_dim]
            
            # Compute SVD
            _, s, _ = np.linalg.svd(H, full_matrices=False)
            
            # Normalize singular values
            s_norm = s / (np.sum(s) + EPS)
            
            # Compute effective rank (entropy-based)
            entropy = -np.sum(s_norm * np.log(s_norm + EPS))
            effective_rank = np.exp(entropy)
            
            return effective_rank, s
        else:
            return 0.0, None
            
    finally:
        hook.remove()


def get_num_layers(model):
    """Get the number of layers in a model"""
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'layers'):
        num = len(model.backbone.layers)
        return num
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return len(model.transformer.h)
    elif hasattr(model, 'config') and hasattr(model.config, 'num_hidden_layers'):
        return model.config.num_hidden_layers
    else:
        return 12  # Default fallback


def run_comprehensive_analysis(layer_idx=None, num_texts=30):
    """
    Run comprehensive gradient amplification analysis.
    Uses multiple metrics to understand why Mamba has higher perplexity.
    If layer_idx is None, runs for all layers.
    """
    # Setup
    print("🔄 Loading models...")
    mamba_model, mamba_tokenizer, device = setup_model_and_tokenizer('state-spaces/mamba-130m-hf')
    gpt2_model, gpt2_tokenizer, device = setup_model_and_tokenizer('gpt2')
    
    # Get number of layers
    mamba_num_layers = get_num_layers(mamba_model)
    gpt2_num_layers = get_num_layers(gpt2_model)
    
    print(f"   Mamba has {mamba_num_layers} layers")
    print(f"   GPT-2 has {gpt2_num_layers} layers")
    print(f"   Analyzing Mamba: all {mamba_num_layers} layers")
    print(f"   Analyzing GPT-2: all {gpt2_num_layers} layers\n")
    
    # Determine which layers to analyze for each model
    if layer_idx is None:
        mamba_layers_to_analyze = list(range(mamba_num_layers))
        gpt2_layers_to_analyze = list(range(gpt2_num_layers))
        print("🔬 Comprehensive Gradient Amplification Analysis (ALL LAYERS)")
    else:
        mamba_layers_to_analyze = [layer_idx] if layer_idx < mamba_num_layers else []
        gpt2_layers_to_analyze = [layer_idx] if layer_idx < gpt2_num_layers else []
        print(f"🔬 Comprehensive Gradient Amplification Analysis")
        print(f"   Layer: {layer_idx}")
    
    print(f"   Texts: {num_texts}\n")
    
    # Get texts
    print("📝 Loading texts...")
    relation_prompts = get_relation_prompts()
    texts = []
    for relation, prompts in relation_prompts.items():
        for sentence, _ in prompts:
            texts.append(sentence)
            if len(texts) >= num_texts:
                break
        if len(texts) >= num_texts:
            break
    
    if len(texts) < num_texts:
        generic = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming industries.",
            "Machine learning models require large datasets.",
        ]
        texts.extend(generic * ((num_texts - len(texts)) // len(generic) + 1))
    
    texts = texts[:num_texts]
    
    # Store results for all layers - separate for each model
    mamba_results = {}
    gpt2_results = {}
    
    # Compute output variance once (not layer-specific)
    print("\n📊 Computing Output Variance...")
    mamba_var = compute_output_variance(mamba_model, mamba_tokenizer, texts, device)
    gpt2_var = compute_output_variance(gpt2_model, gpt2_tokenizer, texts, device)
    
    # Analyze Mamba for all layers
    print(f"\n{'='*70}")
    print(f"ANALYZING MAMBA: {mamba_num_layers} LAYERS")
    print(f"{'='*70}\n")
    
    for layer_idx in mamba_layers_to_analyze:
        print(f"\n--- Mamba Layer {layer_idx}/{mamba_num_layers-1} ---")
        
        results = {}
        
        # 1. Gradient Sensitivity
        print(f"📊 Computing Gradient Sensitivity...")
        mamba_sensitivities = []
        for i, text in enumerate(texts[:20]):  # Use subset
            try:
                sens, _ = compute_gradient_sensitivity(mamba_model, mamba_tokenizer, text, layer_idx, device)
                mamba_sensitivities.append(sens)
                if (i + 1) % 5 == 0:
                    print(f"      Processed {i + 1}/20 texts")
            except Exception as e:
                print(f"      Error on text {i}: {e}")
        
        results['gradient_sensitivity'] = {
            'mean': np.mean(mamba_sensitivities) if mamba_sensitivities else 0,
            'std': np.std(mamba_sensitivities) if mamba_sensitivities else 0,
        }
        
        # 2. Lipschitz Constant
        print(f"📊 Computing Lipschitz Constants...")
        mamba_lip_max, mamba_lip_mean = compute_lipschitz_constant(
            mamba_model, mamba_tokenizer, texts, layer_idx, device
        )
        
        results['lipschitz'] = {
            'max': mamba_lip_max,
            'mean': mamba_lip_mean,
        }
        
        # 3. Output Variance (same for all layers)
        results['output_variance'] = mamba_var
        
        # 4. Effective Rank
        print(f"📊 Computing Effective Rank...")
        mamba_rank, _ = compute_effective_rank(mamba_model, mamba_tokenizer, texts, layer_idx, device)
        
        results['effective_rank'] = mamba_rank
        
        mamba_results[layer_idx] = results
    
    # Analyze GPT-2 for all layers
    print(f"\n{'='*70}")
    print(f"ANALYZING GPT-2: {gpt2_num_layers} LAYERS")
    print(f"{'='*70}\n")
    
    for layer_idx in gpt2_layers_to_analyze:
        print(f"\n--- GPT-2 Layer {layer_idx}/{gpt2_num_layers-1} ---")
        
        results = {}
        
        # 1. Gradient Sensitivity
        print(f"📊 Computing Gradient Sensitivity...")
        gpt2_sensitivities = []
        for i, text in enumerate(texts[:20]):  # Use subset
            try:
                sens, _ = compute_gradient_sensitivity(gpt2_model, gpt2_tokenizer, text, layer_idx, device)
                gpt2_sensitivities.append(sens)
                if (i + 1) % 5 == 0:
                    print(f"      Processed {i + 1}/20 texts")
            except Exception as e:
                print(f"      Error on text {i}: {e}")
        
        results['gradient_sensitivity'] = {
            'mean': np.mean(gpt2_sensitivities) if gpt2_sensitivities else 0,
            'std': np.std(gpt2_sensitivities) if gpt2_sensitivities else 0,
        }
        
        # 2. Lipschitz Constant
        print(f"📊 Computing Lipschitz Constants...")
        gpt2_lip_max, gpt2_lip_mean = compute_lipschitz_constant(
            gpt2_model, gpt2_tokenizer, texts, layer_idx, device
        )
        
        results['lipschitz'] = {
            'max': gpt2_lip_max,
            'mean': gpt2_lip_mean,
        }
        
        # 3. Output Variance (same for all layers)
        results['output_variance'] = gpt2_var
        
        # 4. Effective Rank
        print(f"📊 Computing Effective Rank...")
        gpt2_rank, _ = compute_effective_rank(gpt2_model, gpt2_tokenizer, texts, layer_idx, device)
        
        results['effective_rank'] = gpt2_rank
        
        gpt2_results[layer_idx] = results
    
    # Print summary for all layers
    print("\n" + "="*70)
    print("GRADIENT AMPLIFICATION ANALYSIS SUMMARY")
    print("="*70)
    
    # Mamba summary
    print("\n" + "="*70)
    print("MAMBA RESULTS (All 23 Layers)")
    print("="*70)
    
    print("\n1. GRADIENT SENSITIVITY (higher = more amplification)")
    print(f"{'Layer':<8} {'Mean':<15} {'Std':<15}")
    print("-"*40)
    for layer_idx in sorted(mamba_layers_to_analyze):
        r = mamba_results[layer_idx]
        print(f"{layer_idx:<8} {r['gradient_sensitivity']['mean']:<15.4f} {r['gradient_sensitivity']['std']:<15.4f}")
    
    print("\n2. LIPSCHITZ CONSTANT - MAX (higher = more amplification)")
    print(f"{'Layer':<8} {'Max':<15} {'Mean':<15}")
    print("-"*40)
    for layer_idx in sorted(mamba_layers_to_analyze):
        r = mamba_results[layer_idx]
        print(f"{layer_idx:<8} {r['lipschitz']['max']:<15.4f} {r['lipschitz']['mean']:<15.4f}")
    
    print("\n3. EFFECTIVE RANK (lower = more focused representations)")
    print(f"{'Layer':<8} {'Rank':<15}")
    print("-"*25)
    for layer_idx in sorted(mamba_layers_to_analyze):
        r = mamba_results[layer_idx]
        print(f"{layer_idx:<8} {r['effective_rank']:<15.2f}")
    
    # GPT-2 summary
    print("\n" + "="*70)
    print("GPT-2 RESULTS (All 12 Layers)")
    print("="*70)
    
    print("\n1. GRADIENT SENSITIVITY (higher = more amplification)")
    print(f"{'Layer':<8} {'Mean':<15} {'Std':<15}")
    print("-"*40)
    for layer_idx in sorted(gpt2_layers_to_analyze):
        r = gpt2_results[layer_idx]
        print(f"{layer_idx:<8} {r['gradient_sensitivity']['mean']:<15.4f} {r['gradient_sensitivity']['std']:<15.4f}")
    
    print("\n2. LIPSCHITZ CONSTANT - MAX (higher = more amplification)")
    print(f"{'Layer':<8} {'Max':<15} {'Mean':<15}")
    print("-"*40)
    for layer_idx in sorted(gpt2_layers_to_analyze):
        r = gpt2_results[layer_idx]
        print(f"{layer_idx:<8} {r['lipschitz']['max']:<15.4f} {r['lipschitz']['mean']:<15.4f}")
    
    print("\n3. EFFECTIVE RANK (lower = more focused representations)")
    print(f"{'Layer':<8} {'Rank':<15}")
    print("-"*25)
    for layer_idx in sorted(gpt2_layers_to_analyze):
        r = gpt2_results[layer_idx]
        print(f"{layer_idx:<8} {r['effective_rank']:<15.2f}")
    
    # Comparison for overlapping layers (layers 0-11)
    print("\n" + "="*70)
    print("COMPARISON: MAMBA vs GPT-2 (Layers 0-11)")
    print("="*70)
    
    common_layers = sorted(set(mamba_layers_to_analyze) & set(gpt2_layers_to_analyze))
    
    print("\n1. GRADIENT SENSITIVITY COMPARISON")
    print(f"{'Layer':<8} {'Mamba':<15} {'GPT-2':<15} {'Ratio':<10}")
    print("-"*50)
    for layer_idx in common_layers:
        mamba_mean = mamba_results[layer_idx]['gradient_sensitivity']['mean']
        gpt2_mean = gpt2_results[layer_idx]['gradient_sensitivity']['mean']
        ratio = mamba_mean / (gpt2_mean + EPS) if gpt2_mean > 0 else 0
        print(f"{layer_idx:<8} {mamba_mean:<15.4f} {gpt2_mean:<15.4f} {ratio:<10.2f}x")
    
    print("\n2. LIPSCHITZ CONSTANT COMPARISON (Max)")
    print(f"{'Layer':<8} {'Mamba':<15} {'GPT-2':<15} {'Ratio':<10}")
    print("-"*50)
    for layer_idx in common_layers:
        mamba_max = mamba_results[layer_idx]['lipschitz']['max']
        gpt2_max = gpt2_results[layer_idx]['lipschitz']['max']
        ratio = mamba_max / (gpt2_max + EPS) if gpt2_max > 0 else 0
        print(f"{layer_idx:<8} {mamba_max:<15.4f} {gpt2_max:<15.4f} {ratio:<10.2f}x")
    
    # Overall statistics
    print("\n" + "="*70)
    print("OVERALL STATISTICS")
    print("="*70)
    
    all_mamba_sens = [mamba_results[l]['gradient_sensitivity']['mean'] for l in mamba_layers_to_analyze]
    all_gpt2_sens = [gpt2_results[l]['gradient_sensitivity']['mean'] for l in gpt2_layers_to_analyze]
    all_mamba_lip = [mamba_results[l]['lipschitz']['max'] for l in mamba_layers_to_analyze]
    all_gpt2_lip = [gpt2_results[l]['lipschitz']['max'] for l in gpt2_layers_to_analyze]
    
    print(f"\nMamba Average Gradient Sensitivity (all {mamba_num_layers} layers):")
    print(f"   Mean: {np.mean(all_mamba_sens):.4f} ± {np.std(all_mamba_sens):.4f}")
    
    print(f"\nGPT-2 Average Gradient Sensitivity (all {gpt2_num_layers} layers):")
    print(f"   Mean: {np.mean(all_gpt2_sens):.4f} ± {np.std(all_gpt2_sens):.4f}")
    
    if common_layers:
        common_mamba_sens = [mamba_results[l]['gradient_sensitivity']['mean'] for l in common_layers]
        common_gpt2_sens = [gpt2_results[l]['gradient_sensitivity']['mean'] for l in common_layers]
        if np.mean(common_gpt2_sens) > 0:
            ratio = np.mean(common_mamba_sens) / np.mean(common_gpt2_sens)
            print(f"\n   Average Ratio (Mamba/GPT-2) for layers 0-11: {ratio:.2f}x")
    
    print(f"\nMamba Average Lipschitz Constant (all {mamba_num_layers} layers):")
    print(f"   Mean Max: {np.mean(all_mamba_lip):.4f} ± {np.std(all_mamba_lip):.4f}")
    
    print(f"\nGPT-2 Average Lipschitz Constant (all {gpt2_num_layers} layers):")
    print(f"   Mean Max: {np.mean(all_gpt2_lip):.4f} ± {np.std(all_gpt2_lip):.4f}")
    
    if common_layers:
        common_mamba_lip = [mamba_results[l]['lipschitz']['max'] for l in common_layers]
        common_gpt2_lip = [gpt2_results[l]['lipschitz']['max'] for l in common_layers]
        if np.mean(common_gpt2_lip) > 0:
            ratio = np.mean(common_mamba_lip) / np.mean(common_gpt2_lip)
            print(f"\n   Average Ratio (Mamba/GPT-2) for layers 0-11: {ratio:.2f}x")
    
    print("="*70)
    
    # Save results - convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        """Recursively convert numpy types to native Python types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_serializable(item) for item in obj)
        else:
            return obj
    
    # Combine results
    all_results = {
        'mamba': mamba_results,
        'gpt2': gpt2_results,
        'mamba_num_layers': mamba_num_layers,
        'gpt2_num_layers': gpt2_num_layers,
        'output_variance': {
            'mamba': mamba_var,
            'gpt2': gpt2_var
        }
    }
    
    results_serializable = convert_to_serializable(all_results)
    with open('gradient_amplification_results_all_layers.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print("\n💾 Results saved to: gradient_amplification_results_all_layers.json")
    
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=None, 
                       help='Layer index to analyze (default: None = all layers)')
    parser.add_argument('--num_texts', type=int, default=30)
    args = parser.parse_args()
    
    run_comprehensive_analysis(layer_idx=args.layer, num_texts=args.num_texts)