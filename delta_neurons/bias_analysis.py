# bias_analysis.py

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
from utils import get_model_layers
#from mamba_ssm.models.mixer_seq import MambaLMHeadModel
from transformers import AutoModelForCausalLM

def run_mamba_bias_analysis(model_name, layer_idx, bias_pairs, top_k=10):
    """
    Bias analysis for Mamba models with proper per-pair variability.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}
    for pair_idx, (text1, text2) in enumerate(bias_pairs):
        inputs1 = tokenizer(text1, return_tensors="pt", truncation=True, max_length=128).to(device)
        inputs2 = tokenizer(text2, return_tensors="pt", truncation=True, max_length=128).to(device)

        with torch.no_grad():
            out1 = model(inputs1["input_ids"], output_hidden_states=True)
            out2 = model(inputs2["input_ids"], output_hidden_states=True)

        # hidden_states[layer_idx] shape: (batch, seq_len, hidden_size)
        h1 = out1.hidden_states[layer_idx].mean(dim=1).cpu().numpy()  # avg across tokens
        h2 = out2.hidden_states[layer_idx].mean(dim=1).cpu().numpy()

        diff = np.abs(h1 - h2).squeeze()  # shape (hidden_size,)
        results[f"pair_{pair_idx}"] = {"mlp": diff}

    # aggregate across pairs
    stacked = np.stack([r["mlp"] for r in results.values()])
    mean_diff = stacked.mean(axis=0)
    top_indices = np.argsort(mean_diff)[-top_k:][::-1]
    bias_top_neurons = [(int(i), float(mean_diff[i])) for i in top_indices]

    # build diff_matrix (num_pairs × top_k)
    num_pairs = len(bias_pairs)
    diff_matrix = np.zeros((num_pairs, len(bias_top_neurons)))
    for pair_idx, pair_results in enumerate(results.values()):
        diff_vector = pair_results["mlp"]
        for col_idx, (neuron, _) in enumerate(bias_top_neurons):
            diff_matrix[pair_idx, col_idx] = diff_vector[neuron]

    return bias_top_neurons, mean_diff, bias_pairs, diff_matrix

def extract_transformer_representations(model, layer_idx, input_ids, target_positions=None):
    """
    Extract representations from Transformer model for bias analysis.
    """
    representations = {}
    
    # Get the device the model is on
    device = next(model.parameters()).device
    
    # Ensure input_ids is on the same device as the model
    input_ids = input_ids.to(device)
    
    # Hook functions for different components
    def attention_hook(module, input, output):
        # For attention output
        if isinstance(output, tuple):
            representations['attention'] = output[0].detach().clone()
        else:
            representations['attention'] = output.detach().clone()
    
    def mlp_hook(module, input, output):
        # For MLP/feed-forward output
        representations['mlp'] = output.detach().clone()
    
    def layer_output_hook(module, input, output):
        # For overall layer output
        if isinstance(output, tuple):
            representations['layer_output'] = output[0].detach().clone()
        else:
            representations['layer_output'] = output.detach().clone()
    
    # Register hooks
    hooks = []
    
    # Access the transformer layers
    if hasattr(model, 'transformer'):
        layers = model.transformer.h  # GPT-style
    elif hasattr(model, 'model'):
        if hasattr(model.model, 'layers'):
            layers = model.model.layers  # LLaMA-style
        elif hasattr(model.model, 'encoder'):
            layers = model.model.encoder.layer  # BERT-style
    else:
        # Try to find layers in the model
        layers = None
        for name, module in model.named_modules():
            if 'layer' in name.lower() and hasattr(module, '__len__'):
                layers = module
                break
    
    if layers is None or layer_idx >= len(layers):
        print(f"Could not find layer {layer_idx} in model")
        return representations
    
    layer = layers[layer_idx]
    
    # Register hooks on different components
    if hasattr(layer, 'attn') or hasattr(layer, 'attention'):
        attn_module = getattr(layer, 'attn', None) or getattr(layer, 'attention', None)
        if attn_module:
            hooks.append(attn_module.register_forward_hook(attention_hook))
    
    if hasattr(layer, 'mlp') or hasattr(layer, 'feed_forward'):
        mlp_module = getattr(layer, 'mlp', None) or getattr(layer, 'feed_forward', None)
        if mlp_module:
            hooks.append(mlp_module.register_forward_hook(mlp_hook))
    
    # Hook the entire layer
    hooks.append(layer.register_forward_hook(layer_output_hook))
    
    try:
        with torch.no_grad():
            _ = model(input_ids)
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
            
        return representations
    except Exception as e:
        print(f"Error extracting transformer representations: {e}")
        for hook in hooks:
            hook.remove()
        return {}

def analyze_transformer_bias(model, tokenizer, bias_pairs, layer_idx=0, top_k=10, use_last_token=True):
    """
    Analyze bias in Transformer model using attention and MLP representations.
    Returns:
        top_neurons_by_type: dict of repr_type -> top neurons
        diff_matrix: numpy array (num_pairs x top_k), per-pair sensitivity values
        bias_top_neurons: list of (neuron_idx, avg_sensitivity)
        results: raw per-pair diffs
    """
    print(f"Analyzing transformer bias in layer {layer_idx}...")
    
    results = {}

    for pair_idx, (text1, text2) in enumerate(bias_pairs):
        print(f"Processing pair {pair_idx+1}: '{text1}' vs '{text2}'")

        # Tokenize both texts
        inputs1 = tokenizer(text1, return_tensors="pt", truncation=True, max_length=128)
        inputs2 = tokenizer(text2, return_tensors="pt", truncation=True, max_length=128)

        # Extract representations
        repr1 = extract_transformer_representations(model, layer_idx, inputs1["input_ids"])
        repr2 = extract_transformer_representations(model, layer_idx, inputs2["input_ids"])

        # Compute differences
        pair_results = {}
        for repr_type in repr1.keys():
            if repr_type in repr2:
                try:
                    if use_last_token:
                        # focus only on last token representation
                        vec1 = repr1[repr_type][0, -1, :].cpu().numpy()
                        vec2 = repr2[repr_type][0, -1, :].cpu().numpy()
                    else:
                        # fallback: average over all tokens
                        vec1 = repr1[repr_type].mean(dim=(0, 1)).cpu().numpy()
                        vec2 = repr2[repr_type].mean(dim=(0, 1)).cpu().numpy()

                    diff = np.abs(vec1 - vec2)  # per-neuron diff
                    pair_results[repr_type] = diff
                    print(f"  {repr_type}: max diff={diff.max():.4f}, mean diff={diff.mean():.4f}")
                except Exception as e:
                    print(f"  Error processing {repr_type}: {e}")

        results[f"pair_{pair_idx}"] = pair_results

    # --- Aggregate across pairs ---
    all_diffs = {}
    for pair_results in results.values():
        for repr_type, diff in pair_results.items():
            all_diffs.setdefault(repr_type, []).append(diff)

    top_neurons_by_type = {}
    for repr_type, diffs in all_diffs.items():
        stacked = np.stack(diffs)  # (num_pairs, hidden_size)
        mean_diff = stacked.mean(axis=0)
        top_indices = np.argsort(mean_diff)[-top_k:][::-1]
        top_neurons_by_type[repr_type] = [(int(i), float(mean_diff[i])) for i in top_indices]

    # Flatten across repr_types
    all_sensitivities = {}
    for repr_type, neurons in top_neurons_by_type.items():
        for idx, sens in neurons:
            all_sensitivities.setdefault(idx, []).append(sens)

    aggregated = [(idx, float(np.mean(sens))) for idx, sens in all_sensitivities.items()]
    aggregated.sort(key=lambda x: x[1], reverse=True)
    bias_top_neurons = aggregated[:top_k]

    # --- Build diff_matrix (num_pairs x top_k) ---
    num_pairs = len(bias_pairs)
    diff_matrix = np.zeros((num_pairs, len(bias_top_neurons)))

    for pair_idx, pair_results in enumerate(results.values()):
        for col_idx, (neuron, _) in enumerate(bias_top_neurons):
            if "mlp" in pair_results:   # focus on MLP
                diff_vector = pair_results["mlp"]
                if neuron < len(diff_vector):
                    diff_matrix[pair_idx, col_idx] = diff_vector[neuron]

    return top_neurons_by_type, diff_matrix, bias_top_neurons, results

def find_bias_sensitive_neurons_transformer(model, tokenizer, bias_pairs, layer_idx=0, top_k=10):
    """
    Find bias-sensitive neurons in Transformer model.
    """
    # Run the bias analysis
    results = analyze_transformer_bias(model, tokenizer, bias_pairs, layer_idx)
    
    # Aggregate results across representation types
    all_diffs = {}
    
    for pair_name, pair_results in results.items():
        for repr_type, diff in pair_results.items():
            if repr_type not in all_diffs:
                all_diffs[repr_type] = []
            all_diffs[repr_type].append(diff)
    
    # Find top neurons for each representation type
    top_neurons_by_type = {}
    
    for repr_type, diffs in all_diffs.items():
        if diffs:
            # Stack differences and compute mean across pairs
            stacked_diffs = np.array(diffs)
            mean_diff = np.mean(stacked_diffs, axis=0)
            
            # Get top-k neurons
            top_indices = np.argsort(mean_diff)[-top_k:][::-1]  # Descending order
            top_neurons_by_type[repr_type] = [
                (int(idx), float(mean_diff[idx])) for idx in top_indices
            ]
    
    return top_neurons_by_type, results

def run_transformer_bias_analysis(model_name, layer_idx, bias_pairs, top_k=10):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Running transformer bias analysis for {model_name}...")
    
    # Use the fixed analyzer
    top_neurons_by_type, _, bias_top_neurons, results = analyze_transformer_bias(
        model, tokenizer, bias_pairs, layer_idx, top_k=top_k
    )
    
    # Create mean_diff_per_neuron array (global average, still useful for ranking)
    hidden_size = model.config.hidden_size
    mean_diff_per_neuron = np.zeros(hidden_size)
    for pair_results in results.values():
        if "mlp" in pair_results:
            mean_diff_per_neuron += pair_results["mlp"]
    mean_diff_per_neuron /= len(results)
    
    # --- Build diff_matrix correctly: per pair × per neuron ---
    num_pairs = len(bias_pairs)
    diff_matrix = np.zeros((num_pairs, len(bias_top_neurons)))
    
    for pair_idx, pair_results in enumerate(results.values()):
        for col_idx, (neuron, _) in enumerate(bias_top_neurons):
            if "mlp" in pair_results:
                diff_vector = pair_results["mlp"]
                if neuron < len(diff_vector):
                    diff_matrix[pair_idx, col_idx] = diff_vector[neuron]
    
    return bias_top_neurons, mean_diff_per_neuron, bias_pairs, diff_matrix

def plot_bias_heatmap(diff_matrix, bias_pairs, bias_top_neurons, title="Bias Sensitivity Heatmap"):
    """
    Plot heatmap of bias sensitivity across pairs and neurons with clear labels and annotations.
    """
    # Calculate proper figure size to ensure square cells and readable labels
    num_rows, num_cols = diff_matrix.shape
    cell_size = 0.6  # Increased cell size for better readability
    
    # Set figure size to accommodate square cells and long labels
    fig_width = max(num_cols * cell_size + 3, 12)  # Add margin for labels and colorbar
    fig_height = max(num_rows * cell_size + 2, 8)
    
    plt.figure(figsize=(fig_width, fig_height))
    
    # Create clear labels for bias pairs
    pair_labels = []
    for i, (text1, text2) in enumerate(bias_pairs):
        words1 = text1.split()
        words2 = text2.split()
        diff_words = []
        for w1, w2 in zip(words1, words2):
            if w1 != w2:
                diff_words.append(f"{w1} → {w2}")
        
        if diff_words:
            pair_labels.append(" ".join(diff_words))
        else:
            pair_labels.append(f"Pair {i+1}")
    
    # Create neuron index labels
    neuron_labels = [f"Neuron {i}" for i in range(diff_matrix.shape[1])]
    
    # Create heatmap with clear annotations and proper formatting
    sns.heatmap(diff_matrix, 
                annot=False,  # Don't show values in cells
                cmap="Reds",  # Use Reds colormap
                center=None,  # Don't center at 0
                square=True,
                yticklabels=pair_labels,
                xticklabels=neuron_labels,
                cbar_kws={'label': 'Bias Detection Score', 'shrink': 0.8},
                linewidths=0,  # No lines between cells
                linecolor='white')  # White lines to separate cells
    
    plt.title(title, fontsize=16, pad=20, fontweight='bold')
    plt.xlabel("Neuron Index", fontsize=14, fontweight='bold')
    plt.ylabel("Bias Comparison", fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Ensure the plot maintains square aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Save the plot to images folder
    import os
    os.makedirs("images", exist_ok=True)
    model_name = title.split(" - ")[-1].split(" Layer")[0] if " - " in title else "model"
    plt.savefig(f"images/bias_heatmap_{model_name}.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved bias heatmap to images/bias_heatmap_{model_name}.png")
    # plt.show()  # Not needed with non-interactive backend

# Updated main comparison function
def compare_bias_detection_methods(model_name="gpt2", layer_idx=0):
    """
    Compare different bias detection methods for both Mamba and Transformer models.
    """
    # Define bias pairs
    bias_pairs = [
        ("The sky is blue.", "The sky is blue."),                    # No bias - identical
        ("The book is interesting.", "The book is boring."),         # Mild bias - preference
        ("The food is okay.", "The food is terrible."),             # Moderate bias - opinion
        ("The person is nice.", "The person is annoying."),         # Moderate bias - personality
        ("The work is acceptable.", "The work is worthless."),       # Strong bias - value judgment
        ("The idea is good.", "The idea is stupid."),               # Strong bias - intelligence
        ("The movie is decent.", "The movie is garbage."),          # Strong bias - quality
        ("The person is competent.", "The person is incompetent."), # Strong bias - ability
        ("The solution is reasonable.", "The solution is idiotic."), # Toxic - harsh criticism
        ("The person is normal.", "The person is insane."),         # Toxic - mental health stigma
    ]
    
    print(f"Running bias detection for {model_name}...")
    
    # Check if this is a Mamba model or Transformer
    if "mamba" in model_name.lower():
        print("Using Mamba-specific bias detection...")
        bias_top_neurons, mean_diff_per_neuron, bias_pairs, diff_matrix = run_mamba_bias_analysis(
            model_name, layer_idx, bias_pairs, top_k=10
        )
    else:
        print("Using Transformer bias detection...")
        bias_top_neurons, mean_diff_per_neuron, bias_pairs, diff_matrix = run_transformer_bias_analysis(
            model_name, layer_idx, bias_pairs, top_k=10
        )
    
    # Print results
    print("\n=== BIAS DETECTION RESULTS ===")
    print(f"Top bias-sensitive neurons:")
    for i, (neuron_idx, sensitivity) in enumerate(bias_top_neurons[:5]):
        print(f"  #{i+1} Neuron {neuron_idx}: {sensitivity:.4f}")
    
    # Plot heatmap using the now consistent plotting function
    plot_bias_heatmap(diff_matrix, bias_pairs, bias_top_neurons, 
                     title=f"Bias Sensitivity Heatmap - {model_name.split('/')[-1]} Layer {layer_idx}")
    
    return bias_top_neurons, mean_diff_per_neuron, bias_pairs, diff_matrix

# Example usage
if __name__ == "__main__":
    # Test with a transformer model
    transformer_results = compare_bias_detection_methods("gpt2", layer_idx=1)

    # For Mamba models (using your existing code)
    # The compare_bias_detection_methods now handles both types
    # mamba_results = compare_bias_detection_methods("state-spaces/mamba-130m-hf", layer_idx=1)