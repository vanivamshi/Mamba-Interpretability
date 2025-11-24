#!/usr/bin/env python3

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from datasets import load_dataset

# Import our efficient analysis functions
from efficient_analysis import (
    analyze_universality_all_layers, 
    find_delta_sensitive_neurons_all_layers, 
    find_projection_dominant_neurons_all_layers,
    find_specialty_neurons_all_layers,
    find_knowledge_neurons_all_layers,
    find_dead_neurons_all_layers,
    find_causal_neurons_all_layers
)

# Try to import seaborn for better heatmaps
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Seaborn not found, using matplotlib for heatmaps.")

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = f'plots/all_layers_{timestamp}'
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
        print(f"Loaded {len(texts)} non-empty samples from Wikitext.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world.",
            "Machine learning models require large datasets.",
            "Natural language processing involves understanding text."
        ] * 10
        print(f"Using {len(texts)} dummy texts instead.")

    # ---------------------------------------------------------------
    # Load Mamba Model
    # ---------------------------------------------------------------
    print("\nLoading Mamba model...")
    mamba_model_name = "state-spaces/mamba-130m-hf"
    try:
        mamba_tokenizer = AutoTokenizer.from_pretrained(mamba_model_name)
        mamba_model = AutoModelForCausalLM.from_pretrained(mamba_model_name).to(device).eval()
        print("Mamba model loaded.")
    except Exception as e:
        print(f"Error loading Mamba model: {e}")
        return

    # ---------------------------------------------------------------
    # Universality Analysis (All Layers)
    # ---------------------------------------------------------------
    universality_tasks = {
        "factual": [
            "The capital of France is", "The largest planet is", "Water freezes at"
        ],
        "mathematical": [
            "Two plus two equals", "The square root of 16 is", "Ten divided by two is"
        ],
        "linguistic": [
            "The plural of mouse is", "The past tense of run is", "The opposite of hot is"
        ]
    }

    print("\nRunning Universality Analysis for ALL layers...")
    universality_results = analyze_universality_all_layers(
        mamba_model, mamba_tokenizer, universality_tasks
    )
    
    # Save results
    with open(f'{plots_dir}/universality_results.json', 'w') as f:
        json.dump(universality_results, f)

    # ---------------------------------------------------------------
    # Delta Variance Analysis (All Layers)
    # ---------------------------------------------------------------
    print("\nRunning Delta Variance Analysis for ALL layers...")
    # Use a subset of texts for speed, but enough for variance
    subset_texts = texts[:200] 
    delta_results = find_delta_sensitive_neurons_all_layers(
        mamba_model, mamba_tokenizer, subset_texts
    )
    
    # Save results
    with open(f'{plots_dir}/delta_results.json', 'w') as f:
        json.dump(delta_results, f)

    # ---------------------------------------------------------------
    # Projection Dominant Neurons Analysis (All Layers)
    # ---------------------------------------------------------------
    print("\nRunning Projection Dominant Neurons Analysis for ALL layers...")
    projection_results = find_projection_dominant_neurons_all_layers(mamba_model)
    
    # Save results
    with open(f'{plots_dir}/projection_results.json', 'w') as f:
        json.dump(projection_results, f)

    # ---------------------------------------------------------------
    # Speciality Neurons Analysis (All Layers)
    # ---------------------------------------------------------------
    print("\nRunning Speciality Neurons Analysis for ALL layers...")
    speciality_texts = {
        "technology": [
            "Artificial intelligence is transforming industries.",
            "The new smartphone features a high-resolution camera.",
            "Cloud computing enables scalable infrastructure."
        ],
        "nature": [
            "The rainforest is home to diverse wildlife.",
            "Photosynthesis is how plants make food.",
            "The ocean currents regulate global climate."
        ]
    }
    speciality_results = find_specialty_neurons_all_layers(mamba_model, mamba_tokenizer, speciality_texts)
    with open(f'{plots_dir}/speciality_results.json', 'w') as f:
        # Convert tuple keys to string for JSON
        json_ready_speciality = {}
        for layer, scores in speciality_results.items():
            json_ready_speciality[layer] = {}
            for cls, neuron_scores in scores.items():
                json_ready_speciality[layer][cls] = neuron_scores
        json.dump(json_ready_speciality, f)

    # ---------------------------------------------------------------
    # Knowledge Neurons Analysis (All Layers) - SKIPPED due to compatibility issues
    # ---------------------------------------------------------------
    print("\nSkipping Knowledge Neurons Analysis (compatibility issues with Mamba)...")
    knowledge_results = {}
    # knowledge_facts = [
    #     {"ground_truth": "Paris", "texts": ["The capital of France is"]},
    #     {"ground_truth": "Einstein", "texts": ["The theory of relativity was proposed by"]}
    # ]
    # knowledge_results = find_knowledge_neurons_all_layers(mamba_model, mamba_tokenizer, knowledge_facts)
    with open(f'{plots_dir}/knowledge_results.json', 'w') as f:
        json.dump(knowledge_results, f)

    # ---------------------------------------------------------------
    # Dead Neurons Analysis (All Layers)
    # ---------------------------------------------------------------
    print("\nRunning Dead Neurons Analysis for ALL layers...")
    # Use a smaller subset for dead neurons to speed up
    dead_subset_texts = texts[:20] 
    dead_indices, activation_freqs = find_dead_neurons_all_layers(mamba_model, mamba_tokenizer, dead_subset_texts)
    with open(f'{plots_dir}/dead_results.json', 'w') as f:
        json.dump({"indices": dead_indices, "frequencies": activation_freqs}, f)

    # ---------------------------------------------------------------
    # Causal Neurons Analysis (All Layers)
    # ---------------------------------------------------------------
    print("\nRunning Causal Neurons Analysis for ALL layers...")
    causal_prompts = ["The quick brown fox"]
    # Target token ID for "fox" or similar - just picking a common token for demo
    target_token_id = mamba_tokenizer.encode(" fox")[0] 
    causal_results = find_causal_neurons_all_layers(mamba_model, mamba_tokenizer, causal_prompts, target_token_id=target_token_id)
    with open(f'{plots_dir}/causal_results.json', 'w') as f:
        json.dump(causal_results, f)

    # ---------------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------------
    print("\nGenerating plots...")
    
    # Helper to convert dict to matrix
    def dict_to_matrix(data_dict):
        layers = sorted([int(k) for k in data_dict.keys()])
        if not layers:
            return None, None
        
        # Check size consistency
        first_layer = data_dict[layers[0]] if isinstance(layers[0], int) else data_dict[str(layers[0])]
        expected_size = len(first_layer)
        
        matrix = []
        valid_layers = []
        for l in layers:
            key = l if isinstance(list(data_dict.keys())[0], int) else str(l)
            if key in data_dict:
                row = data_dict[key]
                if len(row) == expected_size:
                    matrix.append(row)
                    valid_layers.append(l)
                else:
                    print(f"Warning: Layer {l} has size {len(row)}, expected {expected_size}. Skipping.")
        
        if not matrix:
            return None, None
            
        return np.array(matrix), valid_layers

    # 1. Universality Heatmap
    uni_matrix, uni_layers = dict_to_matrix(universality_results)
    if uni_matrix is not None:
        plt.figure(figsize=(15, 10))
        if HAS_SEABORN:
            sns.heatmap(uni_matrix, cmap='viridis', cbar_kws={'label': 'Universality Score'})
        else:
            plt.imshow(uni_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Universality Score')
        
        plt.title("Mamba Universality Scores across Layers")
        plt.xlabel("Neuron Index")
        plt.ylabel("Layer Index")
        plt.yticks(np.arange(len(uni_layers)) + 0.5, uni_layers)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/universality_heatmap.png', dpi=300)
        plt.close()
        print("Saved universality_heatmap.png")
        
        # 1b. Individual Layer Plots (as requested)
        os.makedirs(f'{plots_dir}/universality_layers', exist_ok=True)
        for i, layer_idx in enumerate(uni_layers):
            scores = uni_matrix[i]
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(scores)), scores, color='skyblue', alpha=0.8)
            plt.xlabel("Neuron Index")
            plt.ylabel("Universality Score")
            plt.title(f"Mamba Universality Scores - Layer {layer_idx}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/universality_layers/layer_{layer_idx}.png', dpi=150)
            plt.close()
        print("Saved individual universality layer plots.")

    # 2. Delta Variance Heatmap
    delta_matrix, delta_layers = dict_to_matrix(delta_results)
    if delta_matrix is not None:
        plt.figure(figsize=(15, 10))
        # Log scale might be better for variance if it spans orders of magnitude
        # But let's try linear first or log1p
        if HAS_SEABORN:
            sns.heatmap(np.log1p(delta_matrix), cmap='magma', cbar_kws={'label': 'Log(Variance + 1)'})
        else:
            plt.imshow(np.log1p(delta_matrix), aspect='auto', cmap='magma', interpolation='nearest')
            plt.colorbar(label='Log(Variance + 1)')
            
        plt.title("Mamba Delta Variance across Layers (Log Scale)")
        plt.xlabel("Neuron Index")
        plt.ylabel("Layer Index")
        plt.yticks(np.arange(len(delta_layers)) + 0.5, delta_layers)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/delta_variance_heatmap.png', dpi=300)
        plt.close()
        print("Saved delta_variance_heatmap.png")

        # 2b. Individual Layer Plots
        os.makedirs(f'{plots_dir}/delta_layers', exist_ok=True)
        for i, layer_idx in enumerate(delta_layers):
            variances = delta_matrix[i]
            # Plot top 100 like in the original script
            top_indices = np.argsort(variances)[-100:][::-1]
            top_values = variances[top_indices]
            
            plt.figure(figsize=(12, 6))
            plt.bar(range(100), top_values, color='lightgreen', alpha=0.8)
            plt.xlabel("Neuron Rank (Top 100)")
            plt.ylabel("Delta Variance")
            plt.title(f"Mamba Delta-Sensitive Neurons - Layer {layer_idx}")
            plt.xticks(range(0, 100, 10), [f"#{j+1}" for j in range(0, 100, 10)])
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/delta_layers/layer_{layer_idx}.png', dpi=150)
            plt.close()
        print("Saved individual delta layer plots.")

    # 3. Projection Dominant Heatmap
    proj_matrix, proj_layers = dict_to_matrix(projection_results)
    if proj_matrix is not None:
        plt.figure(figsize=(15, 10))
        if HAS_SEABORN:
            sns.heatmap(proj_matrix, cmap='plasma', cbar_kws={'label': 'Projection Magnitude'})
        else:
            plt.imshow(proj_matrix, aspect='auto', cmap='plasma', interpolation='nearest')
            plt.colorbar(label='Projection Magnitude')
            
        plt.title("Mamba Projection Dominant Neurons across Layers")
        plt.xlabel("Neuron Index")
        plt.ylabel("Layer Index")
        plt.yticks(np.arange(len(proj_layers)) + 0.5, proj_layers)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/projection_heatmap.png', dpi=300)
        plt.close()
        print("Saved projection_heatmap.png")

        # 3b. Individual Layer Plots
        os.makedirs(f'{plots_dir}/projection_layers', exist_ok=True)
        for i, layer_idx in enumerate(proj_layers):
            mags = proj_matrix[i]
            # Plot top 100
            top_indices = np.argsort(mags)[-100:][::-1]
            top_values = mags[top_indices]
            
            plt.figure(figsize=(12, 6))
            plt.bar(range(100), top_values, color='orange', alpha=0.8)
            plt.xlabel("Neuron Rank (Top 100)")
            plt.ylabel("Projection Magnitude")
            plt.title(f"Mamba Projection-Dominant Neurons - Layer {layer_idx}")
            plt.xticks(range(0, 100, 10), [f"#{j+1}" for j in range(0, 100, 10)])
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/projection_layers/layer_{layer_idx}.png', dpi=150)
            plt.close()
        print("Saved individual projection layer plots.")

    # 4. Speciality Heatmap (Max score across classes)
    # Convert speciality results to matrix: layer -> neuron -> max_score
    speciality_matrix_data = {}
    for layer, cls_scores in speciality_results.items():
        # Initialize with zeros
        layer_max_scores = np.zeros(mamba_model.config.hidden_size)
        for cls, scores in cls_scores.items():
            for idx, score in scores:
                if idx < len(layer_max_scores):
                    layer_max_scores[idx] = max(layer_max_scores[idx], score)
        speciality_matrix_data[layer] = layer_max_scores.tolist()

    spec_matrix, spec_layers = dict_to_matrix(speciality_matrix_data)
    if spec_matrix is not None:
        plt.figure(figsize=(15, 10))
        if HAS_SEABORN:
            sns.heatmap(spec_matrix, cmap='coolwarm', cbar_kws={'label': 'Max Speciality Score'})
        else:
            plt.imshow(spec_matrix, aspect='auto', cmap='coolwarm', interpolation='nearest')
            plt.colorbar(label='Max Speciality Score')
        plt.title("Mamba Speciality Neurons across Layers")
        plt.xlabel("Neuron Index")
        plt.ylabel("Layer Index")
        plt.yticks(np.arange(len(spec_layers)) + 0.5, spec_layers)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/speciality_heatmap.png', dpi=300)
        plt.close()
        print("Saved speciality_heatmap.png")

        # Individual Layer Plots
        os.makedirs(f'{plots_dir}/speciality_layers', exist_ok=True)
        for i, layer_idx in enumerate(spec_layers):
            scores = spec_matrix[i]
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(scores)), scores, color='salmon', alpha=0.8)
            plt.xlabel("Neuron Index")
            plt.ylabel("Speciality Score")
            plt.title(f"Mamba Speciality Neurons - Layer {layer_idx}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/speciality_layers/layer_{layer_idx}.png', dpi=150)
            plt.close()
        print("Saved individual speciality layer plots.")

    # 5. Knowledge Heatmap (Max attribution across facts)
    knowledge_matrix_data = {}
    for layer, fact_scores in knowledge_results.items():
        layer_max_scores = np.zeros(mamba_model.config.hidden_size)
        for fact, scores in fact_scores.items():
            # scores might be list of (idx, score) or just list of scores
            if isinstance(scores, list) and len(scores) > 0 and isinstance(scores[0], tuple):
                for idx, score in scores:
                    if idx < len(layer_max_scores):
                        layer_max_scores[idx] = max(layer_max_scores[idx], abs(score))
            elif isinstance(scores, list):
                # It's a list of scores for all neurons
                for idx, score in enumerate(scores):
                    if idx < len(layer_max_scores):
                        layer_max_scores[idx] = max(layer_max_scores[idx], abs(score))
        knowledge_matrix_data[layer] = layer_max_scores.tolist()

    know_matrix, know_layers = dict_to_matrix(knowledge_matrix_data)
    if know_matrix is not None:
        plt.figure(figsize=(15, 10))
        if HAS_SEABORN:
            sns.heatmap(know_matrix, cmap='Reds', cbar_kws={'label': 'Knowledge Attribution'})
        else:
            plt.imshow(know_matrix, aspect='auto', cmap='Reds', interpolation='nearest')
            plt.colorbar(label='Knowledge Attribution')
        plt.title("Mamba Knowledge Neurons across Layers")
        plt.xlabel("Neuron Index")
        plt.ylabel("Layer Index")
        plt.yticks(np.arange(len(know_layers)) + 0.5, know_layers)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/knowledge_heatmap.png', dpi=300)
        plt.close()
        print("Saved knowledge_heatmap.png")

        # Individual Layer Plots
        os.makedirs(f'{plots_dir}/knowledge_layers', exist_ok=True)
        for i, layer_idx in enumerate(know_layers):
            scores = know_matrix[i]
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(scores)), scores, color='red', alpha=0.8)
            plt.xlabel("Neuron Index")
            plt.ylabel("Attribution Score")
            plt.title(f"Mamba Knowledge Neurons - Layer {layer_idx}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/knowledge_layers/layer_{layer_idx}.png', dpi=150)
            plt.close()
        print("Saved individual knowledge layer plots.")

    # 6. Dead Neurons Heatmap (Activation Frequency)
    dead_matrix, dead_layers = dict_to_matrix(activation_freqs)
    if dead_matrix is not None:
        plt.figure(figsize=(15, 10))
        # Log scale for frequency
        if HAS_SEABORN:
            sns.heatmap(np.log1p(dead_matrix), cmap='Greys_r', cbar_kws={'label': 'Log(Frequency + 1)'})
        else:
            plt.imshow(np.log1p(dead_matrix), aspect='auto', cmap='Greys_r', interpolation='nearest')
            plt.colorbar(label='Log(Frequency + 1)')
        plt.title("Mamba Neuron Activation Frequency across Layers")
        plt.xlabel("Neuron Index")
        plt.ylabel("Layer Index")
        plt.yticks(np.arange(len(dead_layers)) + 0.5, dead_layers)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/dead_heatmap.png', dpi=300)
        plt.close()
        print("Saved dead_heatmap.png")

        # Individual Layer Plots
        os.makedirs(f'{plots_dir}/dead_layers', exist_ok=True)
        for i, layer_idx in enumerate(dead_layers):
            freqs = dead_matrix[i]
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(freqs)), freqs, color='gray', alpha=0.8)
            plt.xlabel("Neuron Index")
            plt.ylabel("Activation Frequency")
            plt.title(f"Mamba Activation Frequency - Layer {layer_idx}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/dead_layers/layer_{layer_idx}.png', dpi=150)
            plt.close()
        print("Saved individual dead layer plots.")

    # 7. Causal Neurons Heatmap
    # Convert causal results (list of tuples) to matrix
    causal_matrix_data = {}
    for layer, scores in causal_results.items():
        layer_scores = np.zeros(mamba_model.config.hidden_size)
        for idx, score in scores:
            if idx < len(layer_scores):
                layer_scores[idx] = score
        causal_matrix_data[layer] = layer_scores.tolist()

    causal_matrix, causal_layers = dict_to_matrix(causal_matrix_data)
    if causal_matrix is not None:
        plt.figure(figsize=(15, 10))
        if HAS_SEABORN:
            sns.heatmap(causal_matrix, cmap='Purples', cbar_kws={'label': 'Causal Impact'})
        else:
            plt.imshow(causal_matrix, aspect='auto', cmap='Purples', interpolation='nearest')
            plt.colorbar(label='Causal Impact')
        plt.title("Mamba Causal Neurons across Layers")
        plt.xlabel("Neuron Index")
        plt.ylabel("Layer Index")
        plt.yticks(np.arange(len(causal_layers)) + 0.5, causal_layers)
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/causal_heatmap.png', dpi=300)
        plt.close()
        print("Saved causal_heatmap.png")

        # Individual Layer Plots
        os.makedirs(f'{plots_dir}/causal_layers', exist_ok=True)
        for i, layer_idx in enumerate(causal_layers):
            scores = causal_matrix[i]
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(scores)), scores, color='purple', alpha=0.8)
            plt.xlabel("Neuron Index")
            plt.ylabel("Causal Impact Score")
            plt.title(f"Mamba Causal Neurons - Layer {layer_idx}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/causal_layers/layer_{layer_idx}.png', dpi=150)
            plt.close()
        print("Saved individual causal layer plots.")

    # 4. Summary Plot: Average Score per Layer
    plt.figure(figsize=(12, 6))
    
    if uni_matrix is not None:
        avg_uni = np.mean(uni_matrix, axis=1)
        avg_uni_norm = avg_uni / np.max(avg_uni)
        plt.plot(uni_layers, avg_uni_norm, 'o-', label='Avg Universality', color='skyblue', linewidth=2)
        
    if delta_matrix is not None:
        avg_delta = np.mean(delta_matrix, axis=1)
        avg_delta_norm = avg_delta / np.max(avg_delta)
        plt.plot(delta_layers, avg_delta_norm, 's-', label='Avg Delta Variance', color='lightgreen', linewidth=2)

    if proj_matrix is not None:
        avg_proj = np.mean(proj_matrix, axis=1)
        avg_proj_norm = avg_proj / np.max(avg_proj)
        plt.plot(proj_layers, avg_proj_norm, '^-', label='Avg Projection Magnitude', color='orange', linewidth=2)

    if spec_matrix is not None:
        avg_spec = np.mean(spec_matrix, axis=1)
        avg_spec_norm = avg_spec / (np.max(avg_spec) + 1e-6)
        plt.plot(spec_layers, avg_spec_norm, 'x-', label='Avg Speciality', color='salmon', linewidth=2)

    if know_matrix is not None:
        avg_know = np.mean(know_matrix, axis=1)
        avg_know_norm = avg_know / (np.max(avg_know) + 1e-6)
        plt.plot(know_layers, avg_know_norm, 'd-', label='Avg Knowledge', color='red', linewidth=2)

    if dead_matrix is not None:
        # For dead neurons, maybe we want "Dead Ratio" instead of average frequency?
        # Or average frequency (which is inverse of deadness)
        avg_freq = np.mean(dead_matrix, axis=1)
        avg_freq_norm = avg_freq / (np.max(avg_freq) + 1e-6)
        plt.plot(dead_layers, avg_freq_norm, 'o-', label='Avg Activation Freq', color='gray', linewidth=2)

    if causal_matrix is not None:
        avg_causal = np.mean(causal_matrix, axis=1)
        avg_causal_norm = avg_causal / (np.max(avg_causal) + 1e-6)
        plt.plot(causal_layers, avg_causal_norm, '*-', label='Avg Causal Impact', color='purple', linewidth=2)

        
    plt.xlabel("Layer Index")
    plt.ylabel("Normalized Average Score")
    plt.title("Layer-wise Importance Trends")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(uni_layers if uni_layers else delta_layers)
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/layer_trends.png', dpi=300)
    plt.close()
    print("Saved layer_trends.png")

    print(f"\nAll analysis complete. Results saved to {plots_dir}")

if __name__ == "__main__":
    main()
