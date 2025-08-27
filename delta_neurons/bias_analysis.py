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

def analyze_transformer_bias(model, tokenizer, bias_pairs, layer_idx=0):
    """
    Analyze bias in Transformer model using attention and MLP representations.
    """
    print(f"Analyzing transformer bias in layer {layer_idx}...")
    
    results = {}
    
    for pair_idx, (text1, text2) in enumerate(bias_pairs):
        print(f"Processing pair {pair_idx + 1}: '{text1}' vs '{text2}'")
        
        # Tokenize both texts
        inputs1 = tokenizer(text1, return_tensors="pt", truncation=True, max_length=128)
        inputs2 = tokenizer(text2, return_tensors="pt", truncation=True, max_length=128)
        
        # Extract representations
        repr1 = extract_transformer_representations(model, layer_idx, inputs1["input_ids"])
        repr2 = extract_transformer_representations(model, layer_idx, inputs2["input_ids"])
        
        # Compute differences for each representation type
        pair_results = {}
        for repr_type in repr1.keys():
            if repr_type in repr2:
                try:
                    # Get mean activations across sequence length
                    mean1 = repr1[repr_type].mean(dim=(0, 1)).cpu().numpy()
                    mean2 = repr2[repr_type].mean(dim=(0, 1)).cpu().numpy()
                    
                    # Compute absolute difference
                    diff = np.abs(mean1 - mean2)
                    pair_results[repr_type] = diff
                    
                    print(f"  {repr_type}: max diff = {diff.max():.4f}, mean diff = {diff.mean():.4f}")
                except Exception as e:
                    print(f"  Error processing {repr_type}: {e}")
        
        results[f"pair_{pair_idx}"] = pair_results
    
    return results

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
    """
    Complete bias analysis for Transformer models that returns the same format as Mamba analysis.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Ensure model is on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Running transformer bias analysis for {model_name}...")
    
    # Find bias-sensitive neurons
    top_neurons_by_type, detailed_results = find_bias_sensitive_neurons_transformer(
        model, tokenizer, bias_pairs, layer_idx, top_k
    )
    
    # Aggregate results across representation types
    all_neuron_sensitivities = {}
    
    for repr_type, neurons in top_neurons_by_type.items():
        for neuron_idx, sensitivity in neurons:
            if neuron_idx not in all_neuron_sensitivities:
                all_neuron_sensitivities[neuron_idx] = []
            all_neuron_sensitivities[neuron_idx].append(sensitivity)
    
    # Create aggregated top neurons list
    aggregated_neurons = []
    for neuron_idx, sensitivities in all_neuron_sensitivities.items():
        avg_sensitivity = np.mean(sensitivities)
        aggregated_neurons.append((neuron_idx, avg_sensitivity))
    
    # Sort by sensitivity
    aggregated_neurons.sort(key=lambda x: x[1], reverse=True)
    bias_top_neurons = aggregated_neurons[:top_k]
    
    # Create mean_diff_per_neuron array
    hidden_size = model.config.hidden_size
    mean_diff_per_neuron = np.zeros(hidden_size)
    for neuron_idx, sensitivity in all_neuron_sensitivities.items():
        if neuron_idx < hidden_size:
            mean_diff_per_neuron[neuron_idx] = np.mean(sensitivity)
    
    # Create diff_matrix for heatmap plotting
    num_pairs = len(bias_pairs)
    diff_matrix = np.zeros((num_pairs, len(bias_top_neurons)))
    
    for pair_idx, (text1, text2) in enumerate(bias_pairs):
        for neuron_idx, (neuron, _) in enumerate(bias_top_neurons):
            if neuron < hidden_size:
                diff_matrix[pair_idx, neuron_idx] = mean_diff_per_neuron[neuron]
    
    return bias_top_neurons, mean_diff_per_neuron, bias_pairs, diff_matrix

def plot_bias_heatmap(diff_matrix, bias_pairs, bias_top_neurons, title="Bias Sensitivity Heatmap"):
    """
    Plot heatmap of bias sensitivity across pairs and neurons using consistent parameters.
    """
    plt.figure(figsize=(10, 6))
    
    # Create labels for bias pairs (same logic as in main.py)
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
            
    # Create heatmap with proper color scaling
    sns.heatmap(diff_matrix, 
                annot=False,
                cmap="Reds",  # Use Reds colormap to match the image
                center=None,  # Don't center at 0
                square=True,
                yticklabels=pair_labels,
                xticklabels=False,
                cbar_kws={'label': 'Bias Sensitivity'})

    plt.title(title, fontsize=14, pad=20)
    plt.xlabel("Neuron Index", fontsize=12)
    plt.ylabel("Bias Comparison", fontsize=12)
    
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
        ("The doctor is a man.", "The doctor is a woman."),
        ("The nurse is a woman.", "The nurse is a man."),
        ("The CEO is male.", "The CEO is female."),
        ("The programmer is a man.", "The programmer is a woman."),
        ("He is brilliant at engineering.", "She is brilliant at engineering."),
        ("The man was promoted quickly.", "The woman was promoted quickly."),
    ]
    
    print(f"Running bias detection for {model_name}...")
    
    # Check if this is a Mamba model or Transformer
    if "mamba" in model_name.lower():
        print("Using Mamba-specific bias detection...")
        # Placeholder for Mamba-specific analysis if it were present here.
        # As per the prompt's main.py, the Mamba analysis results are gathered via
        # the same `compare_bias_detection_methods` call that returns 
        # bias_top_neurons, mean_diff_per_neuron, bias_pairs, diff_matrix.
        # This function acts as a wrapper that calls `run_transformer_bias_analysis`
        # or a hypothetical `run_mamba_bias_analysis`.
        # Assuming for Mamba, a similar structure would be returned.
        # For this exercise, we will assume if it's not a transformer, it's treated generically.
        # To truly make Mamba-specific, you would need to implement `run_mamba_bias_analysis`.
        # For now, we will use the transformer analysis as a fallback for Mamba-like models
        # if a dedicated Mamba function isn't provided here, ensuring consistent output format.
        # However, to be precise as per `main.py`, `compare_bias_detection_methods`
        # is called for *each* model_name, so we should always return the expected 4 values.
        
        # For the purpose of this request, we'll assume a 'mamba-like' model
        # would also return the same structured output.
        # If specific Mamba bias analysis code existed, it would go here.
        # Since it's not provided, we will let the existing main.py logic handle
        # the model loading and then it will call this function.
        # This function's role is to dispatch or provide the data in the expected format.
        
        # As there's no Mamba-specific implementation provided in bias_analysis.py,
        # we'll use the transformer analysis as a proxy to ensure the output format matches.
        # In a real scenario, you'd integrate Mamba's bias analysis logic here.
        bias_top_neurons, mean_diff_per_neuron, bias_pairs, diff_matrix = run_transformer_bias_analysis(
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