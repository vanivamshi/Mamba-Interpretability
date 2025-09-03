#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from delta_extraction import find_delta_sensitive_neurons_fixed, find_attention_sensitive_neurons, find_combined_delta_attention_neurons
from utils import get_model_layers
from attention_neurons import integrate_mamba_attention_neurons, MambaAttentionNeurons

# --- Updated projection analysis function supporting GPT-2 attention projections ---
def find_projection_dominant_neurons_fixed(model, layer_idx=0, top_k=10):
    """
    Find neurons with dominant projection weights
    Works for both Mamba and standard Transformers (e.g. GPT-2).
    """
    try:
        layers = get_model_layers(model)
        if layers is None or layer_idx >= len(layers):
            print("Could not access model layers. Using dummy values.")
            return [(i, float(i + 1)) for i in range(top_k)]

        layer = layers[layer_idx]
        projection_weights = None

        possible_projections = [
            # Mamba
            lambda l: l.mixer.x_proj.weight
                if hasattr(l, 'mixer') and hasattr(l.mixer, 'x_proj')
                and hasattr(l.mixer.x_proj, 'weight')
                else None,
            # GPT-2 (attention projection)
            lambda l: l.attn.c_proj.weight
                if hasattr(l, 'attn') and hasattr(l.attn, 'c_proj')
                and hasattr(l.attn.c_proj, 'weight')
                else None,
        ]

        for proj_fn in possible_projections:
            try:
                weights = proj_fn(layer)
                if weights is not None:
                    projection_weights = weights.detach().cpu().numpy()
                    break
            except AttributeError:
                continue

        if projection_weights is None:
            print("Could not find projection weights. Using dummy values.")
            return [(i, float(i + 1)) for i in range(top_k)]

        # Compute magnitude per neuron
        if projection_weights.ndim == 2:
            magnitudes = np.linalg.norm(projection_weights, axis=0)
        else:
            magnitudes = np.abs(projection_weights)

        # Determine hidden size
        dummy_input = torch.randint(0, 100, (1, 10))
        with torch.no_grad():
            output = model(dummy_input, output_hidden_states=True)
            hidden_size = output.hidden_states[layer_idx].shape[-1]

        valid_indices = [i for i in range(len(magnitudes)) if i < hidden_size]
        magnitudes = magnitudes[valid_indices]

        if len(magnitudes) < top_k:
            top_k = len(magnitudes)

        top_dims = np.argsort(magnitudes)[-top_k:]
        return [(int(i), float(magnitudes[i])) for i in top_dims[::-1]]

    except Exception as e:
        print(f"Error in find_projection_dominant_neurons_fixed: {e}")
        return [(i, float(i + 1)) for i in range(top_k)]


# --- Main program ---
def main():
    # Load dataset
    print("\n=== Loading Dataset ===")
    try:
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")
        texts = [item["text"] for item in dataset if item["text"].strip() != ""]
        texts = texts[:100]  # Reduce for faster testing
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

    # Models to compare
    models_to_compare = {
        "Mamba-130M": "state-spaces/mamba-130m-hf",
        "GPT-2": "gpt2",
    }

    results = {}

    for label, model_name in models_to_compare.items():
        print(f"\n=== Loading model: {label} ===")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        model.to("cpu")

        # Delta analysis
        print(f"\nRunning delta-sensitive neuron analysis for {label}...")
        delta_results = find_delta_sensitive_neurons_fixed(
            model, tokenizer, texts, layer_idx=0, top_k=1000
        )

        # Projection analysis
        print(f"Running projection-dominant neuron analysis for {label}...")
        projection_results = find_projection_dominant_neurons_fixed(
            model, layer_idx=0, top_k=1000
        )

        # Attention analysis
        print(f"Running attention neuron analysis for {label}...")
        try:
            # Create sample input for attention analysis
            sample_text = texts[0] if texts else "Sample text for analysis"
            sample_input = tokenizer(sample_text, return_tensors="pt")["input_ids"]
            attention_neurons = integrate_mamba_attention_neurons(
                model, sample_input, layer_indices=[0], methods=['attention_weighted']
            )
        except Exception as e:
            print(f"Attention analysis failed for {label}: {e}")
            attention_neurons = None

        # Combined delta + attention analysis
        print(f"Running combined delta + attention analysis for {label}...")
        try:
            combined_neurons = find_combined_delta_attention_neurons(
                model, tokenizer, texts, layer_idx=0, top_k=1000,
                delta_weight=0.5, attention_weight=0.5
            )
        except Exception as e:
            print(f"Combined analysis failed for {label}: {e}")
            combined_neurons = None

        results[label] = {
            "delta_variance": delta_results,
            "projection_magnitudes": projection_results,
            "attention_neurons": attention_neurons,
            "combined_analysis": combined_neurons,
        }

    # Print top neurons based on z-scores
    print_top_zscores(results, "delta_variance")
    print_top_zscores(results, "projection_magnitudes")
    print_top_zscores(results, "attention_neurons")
    print_top_zscores(results, "combined_analysis")

    # Plot
    plot_zscores(results, "delta_variance", "Delta-Sensitive Neuron Z-scores")
    plot_zscores(results, "projection_magnitudes", "Projection-Dominant Neuron Z-scores")
    plot_zscores(results, "attention_neurons", "Attention Neuron Z-scores")
    plot_zscores(results, "combined_analysis", "Combined Delta+Attention Neuron Z-scores")

    # Visualize attention weights
    visualize_attention_weights(results)
    
    # Export attention analysis results
    export_attention_analysis(results)

    # Compare attention patterns across different models
    compare_attention_patterns(results)

    # Display attention summary
    display_attention_summary(results)


def print_top_zscores(results, measure_name):
    """
    Prints top neurons ranked by z-score for each model.
    """
    for label, metrics in results.items():
        if measure_name == "delta_variance":
            data = metrics["delta_variance"]
        elif measure_name == "projection_magnitudes":
            data = metrics["projection_magnitudes"]
        elif measure_name == "attention_neurons":
            if metrics["attention_neurons"] is None:
                continue
            # Extract attention neuron activations from the analysis results
            attention_data = metrics["attention_neurons"]
            if "analysis_results" in attention_data and "attention_weighted" in attention_data["analysis_results"]:
                layer_0_data = attention_data["analysis_results"]["attention_weighted"].get(0, {})
                if "neuron_activations" in layer_0_data:
                    # Convert to the expected format [(neuron_idx, value), ...]
                    activations = layer_0_data["neuron_activations"]
                    if hasattr(activations, 'cpu'):
                        activations = activations.cpu().numpy()
                    data = [(i, float(activations[i])) for i in range(len(activations))]
                else:
                    continue
            else:
                continue
        elif measure_name == "combined_analysis":
            if metrics["combined_analysis"] is None:
                continue
            data = metrics["combined_analysis"]
        else:
            continue

        values = np.array([v for _, v in data])
        mean_val = values.mean()
        std_val = values.std()

        z_scores = (values - mean_val) / (std_val + 1e-8)
        z_data = list(zip([i for i, _ in data], z_scores))
        z_data = sorted(z_data, key=lambda x: -x[1])

        print(f"\nTop {measure_name} z-scores for {label}:")
        for neuron, z in z_data[:10]:
            print(f"  Neuron {neuron} - z-score: {z:.2f}")


def plot_zscores(results, measure_name, title):
    # Create images folder if it doesn't exist
    images_dir = "images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        print(f"Created directory: {images_dir}")
    
    labels = list(results.keys())
    fig, ax = plt.subplots(figsize=(12,6))

    neurons = None  # Initialize neurons variable
    for i, label in enumerate(labels):
        if measure_name == "delta_variance":
            data = results[label]["delta_variance"]
        elif measure_name == "projection_magnitudes":
            data = results[label]["projection_magnitudes"]
        elif measure_name == "attention_neurons":
            if results[label]["attention_neurons"] is None:
                continue
            # Extract attention neuron activations from the analysis results
            attention_data = results[label]["attention_neurons"]
            if "analysis_results" in attention_data and "attention_weighted" in attention_data["analysis_results"]:
                layer_0_data = attention_data["analysis_results"]["attention_weighted"].get(0, {})
                if "neuron_activations" in layer_0_data:
                    # Convert to the expected format [(neuron_idx, value), ...]
                    activations = layer_0_data["neuron_activations"]
                    if hasattr(activations, 'cpu'):
                        activations = activations.cpu().numpy()
                    data = [(i, float(activations[i])) for i in range(len(activations))]
                else:
                    continue
            else:
                continue
        elif measure_name == "combined_analysis":
            if results[label]["combined_analysis"] is None:
                continue
            data = results[label]["combined_analysis"]
        else:
            continue

        values = np.array([v for _, v in data])
        mean_val = values.mean()
        std_val = values.std()
        z_scores = (values - mean_val) / (std_val + 1e-8)

        # Plot all neurons instead of just top 10
        x = np.arange(len(values))
        ax.plot(x, z_scores, label=label, alpha=0.7, linewidth=0.5)
        ax.scatter(x, z_scores, alpha=0.6, s=10)

    ax.set_ylabel("Z-score")
    ax.set_xlabel("Neuron Index")
    ax.set_title(title + " - All Neurons")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot to file
    filename = f"{images_dir}/{measure_name}_all_neurons.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {filename}")
    
    plt.show()


def visualize_attention_weights(results, save_dir="images"):
    """
    Visualize attention weights for each model.
    
    Args:
        results: Dictionary containing results for each model
        save_dir: Directory to save visualizations
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    
    for label, metrics in results.items():
        if metrics["attention_neurons"] is None:
            continue
            
        try:
            attention_data = metrics["attention_neurons"]
            if "analyzer" in attention_data and "mamba_neurons" in attention_data:
                analyzer = attention_data["analyzer"]
                mamba_neurons = attention_data["mamba_neurons"]
                
                # Visualize neurons for the first layer
                if "attention_weighted" in mamba_neurons and 0 in mamba_neurons["attention_weighted"]:
                    save_path = os.path.join(save_dir, f"{label}_attention_neurons.png")
                    analyzer.visualize_neurons(mamba_neurons["attention_weighted"], layer_idx=0, save_path=save_path)
                    print(f"Saved attention visualization for {label} to {save_path}")
                    
        except Exception as e:
            print(f"Error visualizing attention weights for {label}: {e}")


def export_attention_analysis(results, save_dir="attention_analysis"):
    """
    Export attention analysis results to files for further analysis.
    
    Args:
        results: Dictionary containing results for each model
        save_dir: Directory to save exported data
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    
    for label, metrics in results.items():
        if metrics["attention_neurons"] is None:
            continue
            
        try:
            attention_data = metrics["attention_neurons"]
            
            # Export attention data
            export_data = {
                "model": label,
                "attention_data": {},
                "mamba_neurons": {},
                "analysis_results": {}
            }
            
            # Extract and convert attention data
            if "attention_data" in attention_data:
                for layer_idx, layer_data in attention_data["attention_data"].items():
                    export_data["attention_data"][str(layer_idx)] = {}
                    for key, value in layer_data.items():
                        if hasattr(value, 'cpu'):
                            export_data["attention_data"][str(layer_idx)][key] = value.cpu().numpy().tolist()
                        else:
                            export_data["attention_data"][str(layer_idx)][key] = value.tolist() if hasattr(value, 'tolist') else value
            
            # Export mamba neurons
            if "mamba_neurons" in attention_data:
                for method, method_data in attention_data["mamba_neurons"].items():
                    export_data["mamba_neurons"][method] = {}
                    for layer_idx, layer_data in method_data.items():
                        if layer_data:
                            export_data["mamba_neurons"][method][str(layer_idx)] = {}
                            for key, value in layer_data.items():
                                if hasattr(value, 'cpu'):
                                    export_data["mamba_neurons"][method][str(layer_idx)][key] = value.cpu().numpy().tolist()
                                else:
                                    export_data["mamba_neurons"][method][str(layer_idx)][key] = value.tolist() if hasattr(value, 'tolist') else value
            
            # Export analysis results
            if "analysis_results" in attention_data:
                export_data["analysis_results"] = attention_data["analysis_results"]
            
            # Save to file
            import json
            filename = os.path.join(save_dir, f"{label}_attention_analysis.json")
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"Exported attention analysis for {label} to {filename}")
            
        except Exception as e:
            print(f"Error exporting attention analysis for {label}: {e}")


def compare_attention_patterns(results, save_dir="images"):
    """
    Compare attention patterns across different models.
    
    Args:
        results: Dictionary containing results for each model
        save_dir: Directory to save comparison plots
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    
    try:
        # Extract attention scores for comparison
        model_attention_scores = {}
        
        for label, metrics in results.items():
            if metrics["attention_neurons"] is None:
                continue
                
            attention_data = metrics["attention_neurons"]
            if "analysis_results" in attention_data and "attention_weighted" in attention_data["analysis_results"]:
                layer_0_data = attention_data["analysis_results"]["attention_weighted"].get(0, {})
                if "neuron_activations" in layer_0_data:
                    activations = layer_0_data["neuron_activations"]
                    if hasattr(activations, 'cpu'):
                        activations = activations.cpu().numpy()
                    model_attention_scores[label] = activations
        
        if len(model_attention_scores) < 2:
            print("Need at least 2 models with attention data for comparison")
            return
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Attention Pattern Comparison Across Models', fontsize=16)
        
        # Plot 1: Attention score distributions
        for i, (label, scores) in enumerate(model_attention_scores.items()):
            axes[0, 0].hist(scores, alpha=0.7, label=label, bins=30)
        axes[0, 0].set_title('Attention Score Distributions')
        axes[0, 0].set_xlabel('Attention Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Plot 2: Top neurons comparison
        top_k = min(20, min(len(scores) for scores in model_attention_scores.values()))
        for i, (label, scores) in enumerate(model_attention_scores.items()):
            top_indices = np.argsort(scores)[-top_k:]
            top_scores = scores[top_indices]
            x = np.arange(top_k) + i * 0.3
            axes[0, 1].bar(x, top_scores, width=0.3, label=label, alpha=0.7)
        
        axes[0, 1].set_title(f'Top {top_k} Neurons Comparison')
        axes[0, 1].set_xlabel('Neuron Rank')
        axes[0, 1].set_ylabel('Attention Score')
        axes[0, 1].set_xticks(np.arange(top_k) + 0.3 * (len(model_attention_scores) - 1) / 2)
        axes[0, 1].set_xticklabels([f'#{i+1}' for i in range(top_k)])
        axes[0, 1].legend()
        
        # Plot 3: Attention correlation matrix
        if len(model_attention_scores) > 1:
            labels = list(model_attention_scores.keys())
            correlation_matrix = np.zeros((len(labels), len(labels)))
            
            for i, label1 in enumerate(labels):
                for j, label2 in enumerate(labels):
                    scores1 = model_attention_scores[label1]
                    scores2 = model_attention_scores[label2]
                    # Pad or truncate to same length
                    min_len = min(len(scores1), len(scores2))
                    corr = np.corrcoef(scores1[:min_len], scores2[:min_len])[0, 1]
                    correlation_matrix[i, j] = corr if not np.isnan(corr) else 0.0
            
            im = axes[1, 0].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            axes[1, 0].set_title('Attention Pattern Correlation Matrix')
            axes[1, 0].set_xticks(range(len(labels)))
            axes[1, 0].set_yticks(range(len(labels)))
            axes[1, 0].set_xticklabels(labels, rotation=45)
            axes[1, 0].set_yticklabels(labels)
            plt.colorbar(im, ax=axes[1, 0])
        
        # Plot 4: Attention variance comparison
        variances = [np.var(scores) for scores in model_attention_scores.values()]
        labels = list(model_attention_scores.keys())
        axes[1, 1].bar(labels, variances, alpha=0.7)
        axes[1, 1].set_title('Attention Score Variance Comparison')
        axes[1, 1].set_ylabel('Variance')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save the plot
        filename = os.path.join(save_dir, "attention_pattern_comparison.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved attention pattern comparison to {filename}")
        
        plt.close()
        
    except Exception as e:
        print(f"Error in attention pattern comparison: {e}")


def display_attention_summary(results):
    """
    Display a summary of attention analysis results.
    
    Args:
        results: Dictionary containing results for each model
    """
    print("\n" + "="*80)
    print("ATTENTION ANALYSIS SUMMARY")
    print("="*80)
    
    for label, metrics in results.items():
        print(f"\n{label}:")
        print("-" * 40)
        
        if metrics["attention_neurons"] is None:
            print("  ❌ Attention analysis failed")
            continue
        
        attention_data = metrics["attention_neurons"]
        
        # Summary of attention data
        if "attention_data" in attention_data:
            num_layers = len(attention_data["attention_data"])
            print(f"  ✅ Attention data extracted from {num_layers} layers")
        
        # Summary of mamba neurons
        if "mamba_neurons" in attention_data:
            methods = list(attention_data["mamba_neurons"].keys())
            print(f"  ✅ Mamba neurons created using methods: {', '.join(methods)}")
        
        # Summary of analysis results
        if "analysis_results" in attention_data:
            analysis = attention_data["analysis_results"]
            if "attention_weighted" in analysis and 0 in analysis["attention_weighted"]:
                layer_0_analysis = analysis["attention_weighted"][0]
                if "num_neurons" in layer_0_analysis:
                    print(f"  ✅ Layer 0: {layer_0_analysis['num_neurons']} neurons analyzed")
                if "mean_activation" in layer_0_analysis:
                    print(f"  📊 Mean activation: {layer_0_analysis['mean_activation']:.4f}")
                if "activation_std" in layer_0_analysis:
                    print(f"  📊 Activation std: {layer_0_analysis['activation_std']:.4f}")
                if "neuron_diversity" in layer_0_analysis:
                    print(f"  📊 Neuron diversity: {layer_0_analysis['neuron_diversity']:.4f}")
        
        # Summary of combined analysis
        if metrics["combined_analysis"] is not None:
            print(f"  🔗 Combined delta+attention analysis: {len(metrics['combined_analysis'])} neurons")
        
        print()
    
    print("="*80)


if __name__ == "__main__":
    main()
