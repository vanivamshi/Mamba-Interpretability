#!/usr/bin/env python3

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from universality_analysis import (
    analyze_universality_fixed,
    combined_multitask_embedding_alignment,
)
from delta_extraction import find_delta_sensitive_neurons_fixed
from attention_neurons import integrate_mamba_attention_neurons
from datetime import datetime
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create timestamp for plot filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Analysis timestamp: {timestamp}")

# Ensure plots directory exists
plots_dir = 'plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    print(f"Created plots directory: {plots_dir}")

# ---------------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------------

print("\n=== Loading Dataset ===")
try:
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")
    texts = [item["text"] for item in dataset if item["text"].strip() != ""]
    #texts = texts[:100]  # reduce for faster testing
    print(f"Loaded {len(texts)} non-empty samples from Wikitext.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Machine learning models require large datasets.",
        "Natural language processing involves understanding text."
    ]
    print(f"Using {len(texts)} dummy texts instead.")

# ---------------------------------------------------------------
# Load Models
# ---------------------------------------------------------------

print("\nLoading Mamba model...")
mamba_model_name = "state-spaces/mamba-130m-hf"
mamba_tokenizer = AutoTokenizer.from_pretrained(mamba_model_name)
mamba_model = AutoModelForCausalLM.from_pretrained(mamba_model_name).to(device).eval()
print("Mamba model loaded.")

print("\nLoading Transformer model (GPT-2)...")
transformer_model_name = "gpt2"
transformer_tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
transformer_model = AutoModelForCausalLM.from_pretrained(transformer_model_name).to(device).eval()
print("Transformer model loaded.")

# ---------------------------------------------------------------
# Define Universality Tasks
# ---------------------------------------------------------------

universality_tasks = {
    "factual": [
        "The capital of France is",
        "The largest planet is",
        "Water freezes at"
    ],
    "mathematical": [
        "Two plus two equals",
        "The square root of 16 is",
        "Ten divided by two is"
    ],
    "linguistic": [
        "The plural of mouse is",
        "The past tense of run is",
        "The opposite of hot is"
    ]
}

# ---------------------------------------------------------------
# Run Universality Analysis
# ---------------------------------------------------------------

print("\nRunning universality analysis on Mamba...")
mamba_uni = analyze_universality_fixed(
    mamba_model, mamba_tokenizer, universality_tasks, layer_idx=0, top_k=10
)

print("\nRunning universality analysis on Transformer...")
transformer_uni = analyze_universality_fixed(
    transformer_model, transformer_tokenizer, universality_tasks, layer_idx=0, top_k=10
)

# Store the original top-k results for reference
mamba_top_k = mamba_uni
transformer_top_k = transformer_uni

# ---------------------------------------------------------------
# Plot Universality Comparison
# ---------------------------------------------------------------

# Get all neuron indices and scores for both models
all_mamba_indices = list(range(mamba_model.config.hidden_size))
all_transformer_indices = list(range(transformer_model.config.hidden_size))

# Get universality scores for all neurons
print("\nComputing universality scores for all neurons...")

# For Mamba - compute for all neurons
mamba_all_scores = []
print("Computing Mamba universality scores...")
for i in range(mamba_model.config.hidden_size):
    # Create a single-neuron universality task
    single_neuron_task = {f"neuron_{i}": [f"Neuron {i} activation"]}
    try:
        uni_result = analyze_universality_fixed(
            mamba_model, mamba_tokenizer, single_neuron_task, layer_idx=0, top_k=1
        )
        if uni_result:
            mamba_all_scores.append(uni_result[0][1])  # Get the score
        else:
            mamba_all_scores.append(0.0)
    except Exception as e:
        print(f"Warning: Error computing universality for Mamba neuron {i}: {e}")
        mamba_all_scores.append(0.0)
    
    # Progress indicator
    if (i + 1) % 100 == 0:
        print(f"  Processed {i + 1}/{mamba_model.config.hidden_size} Mamba neurons")

# For Transformer - compute for all neurons
transformer_all_scores = []
print("Computing Transformer universality scores...")
for i in range(transformer_model.config.hidden_size):
    # Create a single-neuron universality task
    single_neuron_task = {f"neuron_{i}": [f"Neuron {i} activation"]}
    try:
        uni_result = analyze_universality_fixed(
            transformer_model, transformer_tokenizer, single_neuron_task, layer_idx=0, top_k=1
        )
        if uni_result:
            transformer_all_scores.append(uni_result[0][1])  # Get the score
        else:
            transformer_all_scores.append(0.0)
    except Exception as e:
        print(f"Warning: Error computing universality for Transformer neuron {i}: {e}")
        transformer_all_scores.append(0.0)
    
    # Progress indicator
    if (i + 1) % 100 == 0:
        print(f"  Processed {i + 1}/{transformer_model.config.hidden_size} Transformer neurons")

# Convert to numpy arrays
mamba_all_scores = np.array(mamba_all_scores)
transformer_all_scores = np.array(transformer_all_scores)

# Create two separate plots
# Plot 1: Mamba Universality Scores
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.bar(range(len(all_mamba_indices)), mamba_all_scores, color='skyblue', alpha=0.7)
plt.xlabel("Neuron Index")
plt.ylabel("Universality Score")
plt.title(f"Mamba Universality Scores (All {len(all_mamba_indices)} Neurons)")
plt.grid(True, alpha=0.3)

# Plot 2: Transformer Universality Scores
plt.subplot(1, 2, 2)
plt.bar(range(len(all_transformer_indices)), transformer_all_scores, color='salmon', alpha=0.7)
plt.xlabel("Neuron Index")
plt.ylabel("Universality Score")
plt.title(f"Transformer Universality Scores (All {len(all_transformer_indices)} Neurons)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'plots/universality_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved universality comparison plot to plots/universality_comparison_{timestamp}.png")

# Also create a combined plot showing top neurons from both models
plt.figure(figsize=(12, 6))
top_k_combined = 20

# Get top neurons from each model
top_mamba_indices = np.argsort(mamba_all_scores)[-top_k_combined:][::-1]
top_transformer_indices = np.argsort(transformer_all_scores)[-top_k_combined:][::-1]

# Plot top neurons comparison
x_pos = np.arange(top_k_combined)
plt.bar(x_pos - 0.2, mamba_all_scores[top_mamba_indices], width=0.4, label="Mamba", color='skyblue', alpha=0.8)
plt.bar(x_pos + 0.2, transformer_all_scores[top_transformer_indices], width=0.4, label="Transformer", color='salmon', alpha=0.8)

plt.xlabel("Top Neuron Rank")
plt.ylabel("Universality Score")
plt.title(f"Top {top_k_combined} Neurons - Mamba vs Transformer")
plt.legend()
plt.xticks(x_pos, [f"#{i+1}" for i in range(top_k_combined)])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'plots/universality_top_neurons_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved top neurons comparison plot to plots/universality_top_neurons_{timestamp}.png")

# ---------------------------------------------------------------
# Delta Variance Analysis
# ---------------------------------------------------------------

print("\nAnalyzing delta-sensitive neurons in Mamba...")
delta_mamba = find_delta_sensitive_neurons_fixed(
    mamba_model, mamba_tokenizer,
    texts[:1000],  # can adjust as desired
    layer_idx=0, top_k=100
)

print("\nAnalyzing variance of hidden neurons in Transformer...")
def transformer_neuron_variance(model, tokenizer, texts, layer_idx=0, top_k=100):
    activations = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[layer_idx].mean(dim=1).squeeze(0)
        activations.append(hidden.cpu().numpy())
    activations = np.stack(activations)
    variances = np.var(activations, axis=0)
    top_dims = np.argsort(variances)[-top_k:]
    #top_dims = np.argsort(variances)
    return [(int(i), float(variances[i])) for i in top_dims[::-1]]

delta_transformer = transformer_neuron_variance(
    transformer_model, transformer_tokenizer,
    texts[:1000],
    layer_idx=0, top_k=100
)

# Plot delta vs variance comparison
mamba_vars = [v for _, v in delta_mamba]
transformer_vars = [v for _, v in delta_transformer]

plt.figure(figsize=(15, 8))

# Plot 1: Top 100 Mamba Delta Neurons
plt.subplot(2, 1, 1)
mamba_indices = np.arange(len(mamba_vars))
plt.bar(mamba_indices, mamba_vars, color='lightgreen', alpha=0.7, width=0.8)
plt.xlabel("Neuron Rank")
plt.ylabel("Delta Variance")
plt.title(f"Top {len(mamba_vars)} Mamba Δ-Sensitive Neurons")
plt.xticks(mamba_indices[::10], [f"#{i+1}" for i in mamba_indices[::10]])  # Show every 10th label
plt.grid(True, alpha=0.3)

# Plot 2: Top 100 Transformer Variance Neurons
plt.subplot(2, 1, 2)
transformer_indices = np.arange(len(transformer_vars))
plt.bar(transformer_indices, transformer_vars, color='orange', alpha=0.7, width=0.8)
plt.xlabel("Neuron Rank")
plt.ylabel("Variance")
plt.title(f"Top {len(transformer_vars)} Transformer Variance Neurons")
plt.xticks(transformer_indices[::10], [f"#{i+1}" for i in transformer_indices[::10]])  # Show every 10th label
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'plots/delta_variance_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved delta variance comparison plot to plots/delta_variance_comparison_{timestamp}.png")

# Additional comparison plot: Side-by-side comparison of top neurons
plt.figure(figsize=(16, 8))

# Normalize values for better comparison (0-1 scale)
mamba_normalized = np.array(mamba_vars) / np.max(mamba_vars)
transformer_normalized = np.array(transformer_vars) / np.max(transformer_vars)

# Plot normalized comparison
x = np.arange(len(mamba_vars))
plt.plot(x, mamba_normalized, 'o-', color='lightgreen', linewidth=2, markersize=4, label='Mamba Δ Neurons', alpha=0.8)
plt.plot(x, transformer_normalized, 's-', color='orange', linewidth=2, markersize=4, label='Transformer Neurons', alpha=0.8)

plt.xlabel("Neuron Rank (Top 100)")
plt.ylabel("Normalized Score")
plt.title("Comparison of Top 100 Neurons: Mamba Δ-Sensitive vs Transformer Variance")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(x[::10], [f"#{i+1}" for i in x[::10]])

plt.tight_layout()
plt.savefig(f'plots/delta_variance_comparison_normalized_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved normalized delta variance comparison plot to plots/delta_variance_comparison_normalized_{timestamp}.png")

# ---------------------------------------------------------------
# Combined PCA Embedding Plot
# ---------------------------------------------------------------

print("\nCreating combined PCA plot for both models...")
combined_multitask_embedding_alignment(
    mamba_model, mamba_tokenizer,
    transformer_model, transformer_tokenizer,
    universality_tasks,
    layer_idx=0,
    method="pca",
    timestamp=timestamp
)

# ---------------------------------------------------------------
# Mamba Attention Neurons Analysis
# ---------------------------------------------------------------

print("\n=== Mamba Attention Neurons Analysis ===")

# Prepare sample input for mamba attention neurons analysis
# We'll use the first few texts from our dataset
sample_texts = texts[:5]  # Use first 5 texts for analysis
sample_inputs = []
for text in sample_texts:
    inputs = mamba_tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    sample_inputs.append(inputs)

print(f"Analyzing mamba attention neurons for {len(sample_inputs)} sample texts...")

# Analyze mamba attention neurons for different layers
layer_indices = [0, 6, 12, 18]  # Key layers to analyze
methods = ['attention_weighted', 'rollout']  # Methods to use

mamba_attention_results = {}
for i, inputs in enumerate(sample_inputs):
    print(f"\nAnalyzing text {i+1}/{len(sample_inputs)}...")
    
    try:
        # Extract input_ids for the model
        input_ids = inputs['input_ids']
        
        # Integrate mamba attention neurons analysis
        results = integrate_mamba_attention_neurons(
            mamba_model, 
            input_ids,
            layer_indices=layer_indices,
            methods=methods
        )
        
        mamba_attention_results[f"text_{i+1}"] = results
        
        # Visualize neurons for the first layer
        if results['mamba_neurons'] and 'attention_weighted' in results['mamba_neurons']:
            if 0 in results['mamba_neurons']['attention_weighted']:
                print(f"Visualizing neurons for text {i+1}, layer 0...")
                results['analyzer'].visualize_neurons(
                    results['mamba_neurons']['attention_weighted'],
                    layer_idx=0,
                    save_path=f'plots/mamba_neurons_text_{i+1}_layer_0_{timestamp}.png'
                )
        
    except Exception as e:
        print(f"Error analyzing text {i+1}: {e}")
        continue

# Create summary of mamba attention neurons analysis
if mamba_attention_results:
    print("\nCreating mamba attention neurons summary...")
    
    # Aggregate results across all texts
    all_analysis_results = {}
    for text_key, results in mamba_attention_results.items():
        for method in methods:
            if method not in all_analysis_results:
                all_analysis_results[method] = {}
            
            for layer_idx in layer_indices:
                if layer_idx not in all_analysis_results[method]:
                    all_analysis_results[method][layer_idx] = []
                
                if method in results['analysis_results'] and layer_idx in results['analysis_results'][method]:
                    all_analysis_results[method][layer_idx].append(results['analysis_results'][method][layer_idx])
    
    # Create summary plots
    for method in methods:
        if method in all_analysis_results:
            print(f"\nCreating summary plot for {method} method...")
            
            # Plot average neuron activations across all texts for each layer
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Mamba Attention Neurons Summary - {method.replace("_", " ").title()} Method', fontsize=16)
            
            for i, layer_idx in enumerate(layer_indices[:4]):  # Plot first 4 layers
                if layer_idx in all_analysis_results[method]:
                    layer_results = all_analysis_results[method][layer_idx]
                    
                    if layer_results:
                        # Extract neuron activations
                        activations = []
                        for result in layer_results:
                            if result and 'neuron_activations' in result:
                                activations.append(result['neuron_activations'])
                        
                        if activations:
                            # Average across all texts
                            avg_activations = torch.stack(activations).mean(dim=0)
                            
                            # Plot
                            row, col = i // 2, i % 2
                            axes[row, col].bar(range(len(avg_activations)), avg_activations.cpu().numpy())
                            axes[row, col].set_title(f'Layer {layer_idx} - Avg Neuron Activations')
                            axes[row, col].set_xlabel('Neuron Index')
                            axes[row, col].set_ylabel('Activation Value')
            
            plt.tight_layout()
            plt.savefig(f'plots/mamba_attention_neurons_summary_{method}_{timestamp}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {method} summary plot to plots/mamba_attention_neurons_summary_{method}_{timestamp}.png")

# Compare traditional delta analysis with mamba attention neurons
if mamba_attention_results and delta_mamba:
    print("\n=== Comparing Traditional vs Attention-Based Neuron Analysis ===")
    
    # Extract attention-based neuron activations for layer 0
    attention_neurons_layer0 = []
    for text_key, results in mamba_attention_results.items():
        if 'attention_weighted' in results['mamba_neurons'] and 0 in results['mamba_neurons']['attention_weighted']:
            neurons = results['mamba_neurons']['attention_weighted'][0]
            if neurons and 'neuron_activations' in neurons:
                attention_neurons_layer0.append(neurons['neuron_activations'])
    
    if attention_neurons_layer0:
        # Average attention-based neurons across all texts
        avg_attention_neurons = torch.stack(attention_neurons_layer0).mean(dim=0)
        
        # Compare with traditional delta analysis
        delta_values = torch.tensor([v for _, v in delta_mamba])
        
        # Create comparison plot
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Traditional delta analysis
        plt.subplot(1, 2, 1)
        indices = [idx for idx, _ in delta_mamba]
        plt.bar(range(len(indices)), delta_values, color='lightgreen', alpha=0.7)
        plt.title('Traditional Delta-Sensitive Neurons')
        plt.xlabel('Neuron Index')
        plt.ylabel('Delta Variance')
        plt.xticks(range(len(indices)), [str(idx) for idx in indices])
        
        # Plot 2: Attention-based neurons
        plt.subplot(1, 2, 2)
        # Select top neurons from attention analysis
        top_attention_indices = torch.argsort(avg_attention_neurons, descending=True)[:len(delta_mamba)]
        top_attention_values = avg_attention_neurons[top_attention_indices]
        plt.bar(range(len(top_attention_indices)), top_attention_values, color='skyblue', alpha=0.7)
        plt.title('Attention-Based Mamba Neurons')
        plt.xlabel('Neuron Rank')
        plt.ylabel('Attention Activation')
        plt.xticks(range(len(top_attention_indices)), [f'#{idx.item()}' for idx in top_attention_indices])
        
        plt.tight_layout()
        plt.savefig(f'plots/traditional_vs_attention_neurons_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison plot to plots/traditional_vs_attention_neurons_{timestamp}.png")
        
        # Calculate correlation between methods
        if len(delta_values) == len(top_attention_values):
            correlation = torch.corrcoef(torch.stack([delta_values, top_attention_values]))[0, 1].item()
            print(f"Correlation between traditional and attention-based methods: {correlation:.4f}")
            
            # Save correlation to summary
            with open(summary_file, 'a') as f:
                f.write(f"\nMethod Comparison:\n")
                f.write("-" * 25 + "\n")
                f.write(f"Correlation between traditional delta analysis and attention-based neurons: {correlation:.4f}\n")

# ---------------------------------------------------------------
# Save Analysis Summary
# ---------------------------------------------------------------

print("\nSaving analysis summary...")
summary_file = f'plots/analysis_summary_{timestamp}.txt'
with open(summary_file, 'w') as f:
    f.write(f"Analysis Summary - {timestamp}\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Device used: {device}\n")
    f.write(f"Number of texts analyzed: {len(texts)}\n")
    f.write(f"Mamba model: {mamba_model_name}\n")
    f.write(f"Transformer model: {transformer_model_name}\n\n")
    
    f.write("Universality Analysis Results:\n")
    f.write("-" * 30 + "\n")
    
    # Mamba universality results
    f.write("Mamba Universality Analysis:\n")
    f.write(f"Total neurons analyzed: {len(all_mamba_indices)}\n")
    f.write(f"Mean universality score: {mamba_all_scores.mean():.4f}\n")
    f.write(f"Std universality score: {mamba_all_scores.std():.4f}\n")
    f.write("Top 10 Mamba universal neurons:\n")
    top_mamba_indices = np.argsort(mamba_all_scores)[-10:][::-1]
    for i, idx in enumerate(top_mamba_indices):
        f.write(f"  {i+1}. Neuron {idx}: {mamba_all_scores[idx]:.4f}\n")
    
    f.write("\nTransformer Universality Analysis:\n")
    f.write(f"Total neurons analyzed: {len(all_transformer_indices)}\n")
    f.write(f"Mean universality score: {transformer_all_scores.mean():.4f}\n")
    f.write(f"Std universality score: {transformer_all_scores.std():.4f}\n")
    f.write("Top 10 Transformer universal neurons:\n")
    top_transformer_indices = np.argsort(transformer_all_scores)[-10:][::-1]
    for i, idx in enumerate(top_transformer_indices):
        f.write(f"  {i+1}. Neuron {idx}: {transformer_all_scores[idx]:.4f}\n")
    
    # Add comparison statistics
    f.write(f"\nModel Comparison:\n")
    f.write(f"Mamba vs Transformer universality correlation: {np.corrcoef(mamba_all_scores[:min(len(mamba_all_scores), len(transformer_all_scores))], transformer_all_scores[:min(len(mamba_all_scores), len(transformer_all_scores))])[0,1]:.4f}\n")
    f.write(f"Mamba universality range: [{mamba_all_scores.min():.4f}, {mamba_all_scores.max():.4f}]\n")
    f.write(f"Transformer universality range: [{transformer_all_scores.min():.4f}, {transformer_all_scores.max():.4f}]\n")
    
    f.write("\nDelta Analysis Results:\n")
    f.write("-" * 25 + "\n")
    f.write("Top Mamba delta-sensitive neurons:\n")
    for i, (idx, var) in enumerate(delta_mamba):
        f.write(f"  {i+1}. Neuron {idx}: variance {var:.4f}\n")
    f.write("\nTop Transformer variance neurons:\n")
    for i, (idx, var) in enumerate(delta_transformer):
        f.write(f"  {i+1}. Neuron {idx}: variance {var:.4f}\n")
    
    # Add Mamba Attention Neurons Analysis Summary
    if mamba_attention_results:
        f.write("\nMamba Attention Neurons Analysis Results:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Number of texts analyzed: {len(mamba_attention_results)}\n")
        f.write(f"Layers analyzed: {layer_indices}\n")
        f.write(f"Methods used: {methods}\n\n")
        
        for method in methods:
            f.write(f"{method.replace('_', ' ').title()} Method Results:\n")
            if method in all_analysis_results:
                for layer_idx in layer_indices:
                    if layer_idx in all_analysis_results[method]:
                        layer_results = all_analysis_results[method][layer_idx]
                        if layer_results:
                            # Calculate average statistics
                            avg_activations = []
                            avg_importance = []
                            for result in layer_results:
                                if result and 'neuron_activations' in result:
                                    avg_activations.append(result['neuron_activations'])
                                if result and 'neuron_importance' in result:
                                    avg_importance.append(result['neuron_importance'])
                            
                            if avg_activations:
                                avg_act = torch.stack(avg_activations).mean(dim=0)
                                top_neurons = torch.argsort(avg_act, descending=True)[:5]
                                f.write(f"  Layer {layer_idx}:\n")
                                f.write(f"    Top 5 neurons: {[(idx.item(), avg_act[idx].item()) for idx in top_neurons]}\n")
                                f.write(f"    Average activation: {avg_act.mean().item():.4f}\n")
                                f.write(f"    Activation std: {avg_act.std().item():.4f}\n")
            f.write("\n")

print(f"Analysis summary saved to {summary_file}")
print("\nAll plots complete!")
