# main.py (Fixed: proper ablation study, positional neurons, and perplexity evaluation)

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
import argparse
import os
from utils import debug_model_structure, get_model_layers
from delta_extraction import find_delta_sensitive_neurons_fixed, evaluate_perplexity
from neuron_characterization import run_complete_neuron_analysis
from visualization_module import create_comprehensive_report
from attention_neurons import integrate_mamba_attention_neurons, MambaAttentionNeurons

import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

OUTPUT_DIR = "analysis_outputs"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def setup_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_analysis_texts(text_limit=300): # text_limit=None
    try:
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")
        texts = [item["text"] for item in dataset if item["text"].strip()]
        texts = [text for text in texts if len(text.split()) > 10 and not text.startswith("=")]
        # Apply text limit if specified (None means use all texts)
        if text_limit is not None and text_limit > 0:
            texts = texts[:text_limit]
        return texts
    except:
        return [
            "Artificial intelligence is transforming industries.",
            "The quick brown fox jumps over the lazy dog.",
            "Transformer models have revolutionized NLP tasks.",
            "Quantum computing promises exponential speedup.",
            "She loves chocolate. She hates chocolate."
        ]


def save_results(results, filename=None):
    from datetime import datetime
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if filename is None:
        filename = os.path.join(OUTPUT_DIR, f"neuron_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    def convert_numpy(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return str(obj)

    def recursive_convert(obj):
        if isinstance(obj, dict): return {k: recursive_convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [recursive_convert(v) for v in obj]
        if isinstance(obj, tuple): return tuple(recursive_convert(v) for v in obj)
        return convert_numpy(obj)

    with open(filename, 'w') as f:
        json.dump(recursive_convert(results), f, indent=2)
    print(f"\n✅ Results saved to: {filename}")


def register_ablation_hook(layer, neuron_indices, mode="zero"):
    """Register a hook to ablate specific neurons."""
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            # Handle tuple outputs (common in some models)
            hidden_states = output[0]
            if len(neuron_indices) > 0:
                valid_indices = [i for i in neuron_indices if 0 <= i < hidden_states.shape[-1]]
                if valid_indices:
                    if mode == "zero":
                        hidden_states[:, :, valid_indices] = 0
                    elif mode == "mean":
                        for idx in valid_indices:
                            mean_val = hidden_states[:, :, idx].mean()
                            hidden_states[:, :, idx] = mean_val
            return (hidden_states, *output[1:])
        else:
            # Handle tensor outputs
            if len(neuron_indices) > 0:
                valid_indices = [i for i in neuron_indices if 0 <= i < output.shape[-1]]
                if valid_indices:
                    if mode == "zero":
                        output[:, :, valid_indices] = 0
                    elif mode == "mean":
                        for idx in valid_indices:
                            mean_val = output[:, :, idx].mean()
                            output[:, :, idx] = mean_val
            return output
    
    return layer.register_forward_hook(hook_fn)


def evaluate_perplexity_with_ablation(model, tokenizer, texts, device, layer_idx=None, neuron_indices=None, mode="zero"):
    """
    Evaluate perplexity with optional neuron ablation.
    """
    model.to(device)
    model.eval()
    
    hook = None
    if layer_idx is not None and neuron_indices is not None and len(neuron_indices) > 0:
        layers = get_model_layers(model)
        if layers is not None and 0 <= layer_idx < len(layers):
            hook = register_ablation_hook(layers[layer_idx], neuron_indices, mode)
            print(f"Registered ablation hook for {len(neuron_indices)} neurons in layer {layer_idx}")
    
    total_loss = 0.0
    total_tokens = 0
    
    try:
        with torch.no_grad():
            for text in texts:
                if len(text.strip()) == 0:
                    continue
                    
                try:
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    
                    if torch.isfinite(loss):
                        total_loss += loss.item() * inputs["input_ids"].numel()
                        total_tokens += inputs["input_ids"].numel()
                        
                except Exception as e:
                    print(f"Error processing text: {str(e)[:100]}...")
                    continue
                    
    finally:
        if hook is not None:
            hook.remove()
    
    if total_tokens == 0:
        return float('inf')
    
    perplexity = np.exp(total_loss / total_tokens)
    return perplexity


def run_comprehensive_ablation_study(model, tokenizer, texts, layer_idx, dead_neurons, positional_neurons, delta_neurons):
    """
    Run comprehensive ablation study comparing different neuron types.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n=== Running Comprehensive Ablation Study ===")
    
    # Baseline perplexity
    print("Computing baseline perplexity...")
    baseline_ppl = evaluate_perplexity_with_ablation(model, tokenizer, texts, device)
    print(f"Baseline perplexity: {baseline_ppl:.3f}")
    
    results = {"baseline": baseline_ppl}
    
    # Ablate dead neurons
    if dead_neurons and len(dead_neurons) > 0:
        print(f"Ablating {len(dead_neurons)} dead neurons...")
        dead_ppl = evaluate_perplexity_with_ablation(
            model, tokenizer, texts, device, 
            layer_idx=layer_idx, neuron_indices=dead_neurons, mode="zero"
        )
        results["dead_ablated"] = dead_ppl
        print(f"Dead neurons ablated perplexity: {dead_ppl:.3f}")
    
    # Ablate positional neurons
    if positional_neurons and len(positional_neurons) > 0:
        print(f"Ablating {len(positional_neurons)} positional neurons...")
        pos_ppl = evaluate_perplexity_with_ablation(
            model, tokenizer, texts, device,
            layer_idx=layer_idx, neuron_indices=positional_neurons, mode="zero"
        )
        results["positional_ablated"] = pos_ppl
        print(f"Positional neurons ablated perplexity: {pos_ppl:.3f}")
    
    # Ablate delta-sensitive neurons
    if delta_neurons and len(delta_neurons) > 0:
        print(f"Ablating {len(delta_neurons)} delta-sensitive neurons...")
        delta_ppl = evaluate_perplexity_with_ablation(
            model, tokenizer, texts, device,
            layer_idx=layer_idx, neuron_indices=delta_neurons, mode="zero"
        )
        results["delta_ablated"] = delta_ppl
        print(f"Delta neurons ablated perplexity: {delta_ppl:.3f}")
    
    # Ablate combination of dead + positional
    combined_neurons = []
    if dead_neurons:
        combined_neurons.extend(dead_neurons)
    if positional_neurons:
        combined_neurons.extend(positional_neurons)
    
    if combined_neurons:
        # Remove duplicates
        combined_neurons = list(set(combined_neurons))
        print(f"Ablating {len(combined_neurons)} combined (dead + positional) neurons...")
        combined_ppl = evaluate_perplexity_with_ablation(
            model, tokenizer, texts, device,
            layer_idx=layer_idx, neuron_indices=combined_neurons, mode="zero"
        )
        results["dead_and_positional_ablated"] = combined_ppl
        print(f"Combined ablated perplexity: {combined_ppl:.3f}")
    
    return results


def run_rarely_active_ablation_study(model, tokenizer, texts, layer_idx, rare_neurons):
    """
    Run ablation study for rarely active neurons.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    if rare_neurons and len(rare_neurons) > 0:
        print(f"Ablating {len(rare_neurons)} rarely active neurons...")
        rare_ppl = evaluate_perplexity_with_ablation(
            model, tokenizer, texts, device,
            layer_idx=layer_idx, neuron_indices=rare_neurons, mode="zero"
        )
        results["rarely_active_ablated"] = rare_ppl
        print(f"Rarely active neurons ablated perplexity: {rare_ppl:.3f}")
    else:
        print("⚠️ No rarely active neurons found for ablation")

    return results


def plot_ablation_results(ablation_results, model_name, layer_idx):
    """Plot ablation study results with proper visualization."""
    if not ablation_results or len(ablation_results) <= 1:
        print("No ablation results to plot")
        return
    
    conditions = list(ablation_results.keys())
    perplexities = list(ablation_results.values())

    label_mapping = {
        "baseline": "Baseline",
        "dead_ablated": "Dead Neurons\nAblated",
        "positional_ablated": "Positional Neurons\nAblated",
        "delta_ablated": "Delta-Sensitive\nNeurons Ablated",
        "dead_and_positional_ablated": "Dead + Positional\nAblated",
        "rarely_active_ablated": "Rarely Active\nNeurons Ablated"   # 🔥 new
    }
    readable_labels = [label_mapping.get(cond, cond) for cond in conditions]

    plt.figure(figsize=(12, 6))
    colors = sns.color_palette("husl", len(conditions))  # more scalable
    bars = plt.bar(range(len(conditions)), perplexities, color=colors, alpha=0.75, edgecolor='black')

    for i, (bar, ppl) in enumerate(zip(bars, perplexities)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(perplexities) * 0.01,
                f'{ppl:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.xticks(range(len(conditions)), readable_labels, rotation=45, ha='right')
    plt.ylabel("Perplexity", fontsize=12)
    plt.title(f"Perplexity After Neuron Group Ablation\n{model_name} Layer {layer_idx}", fontsize=14)

    if "baseline" in ablation_results:
        baseline_val = ablation_results["baseline"]
        plt.axhline(y=baseline_val, color='red', linestyle='--', alpha=0.5, 
                   label=f'Baseline ({baseline_val:.2f})')
        plt.legend()

    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    base_name = f"{model_name.replace('/', '_')}_layer{layer_idx}"
    path = os.path.join(PLOTS_DIR, f"{base_name}_ablation_comprehensive.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Ablation plot saved to: {path}")


def run_comprehensive_analysis(model, tokenizer, texts, layer_idx=0, top_k=10, model_name="model"):
    """Run comprehensive neuron analysis with proper ablation study."""
    results = {
        'model_config': {
            'hidden_size': model.config.hidden_size,
            'num_layers': getattr(model.config, 'num_hidden_layers', 'unknown'),
            'layer_analyzed': layer_idx
        },
        'analysis_results': {}
    }
    
    # Find delta-sensitive neurons
    print("Finding delta-sensitive neurons...")
    delta_results = find_delta_sensitive_neurons_fixed(model, tokenizer, texts[:300], layer_idx, top_k)
    delta_neurons = [n for n, _ in delta_results[:5]]  # Top 5 delta neurons
    results['analysis_results']['delta_sensitive_neurons'] = delta_results
    
    # Run attention neurons analysis
    print("Running attention neurons analysis...")
    try:
        # Create sample input for attention analysis
        sample_text = texts[0] if texts else "Sample text for analysis"
        sample_input = tokenizer(sample_text, return_tensors="pt")["input_ids"]
        attention_neurons = integrate_mamba_attention_neurons(
            model, sample_input, layer_indices=[layer_idx], methods=['attention_weighted']
        )
        results['analysis_results']['attention_neurons'] = attention_neurons
        print("✅ Attention neurons analysis completed successfully")
    except Exception as e:
        print(f"❌ Attention neurons analysis failed: {e}")
        results['analysis_results']['attention_neurons'] = None
    
    # Run complete neuron analysis (includes dead, positional, overlap, and pruning)
    print("Running complete neuron characterization...")
    neuron_analysis = run_complete_neuron_analysis(
        model, tokenizer, texts[:300], layer_idx=layer_idx, 
        delta_neurons=delta_neurons, pruning_ratio=0.15
    )
    
    # Extract individual components for compatibility
    dead_neurons = neuron_analysis['dead_neurons']
    positional_neurons = neuron_analysis['positional_neurons']
    
    # Run comprehensive ablation study
    ablation_results = run_comprehensive_ablation_study(
        model, tokenizer, texts[:300], layer_idx, 
        dead_neurons, positional_neurons, delta_neurons
    )
    
    # Merge all results
    results['analysis_results'].update(neuron_analysis)
    results['analysis_results']['ablation_study'] = ablation_results
    
    # Create comprehensive visualizations
    create_comprehensive_report(
        results['analysis_results'], model_name, layer_idx, PLOTS_DIR
    )
    
    # Print detailed summary
    print_analysis_summary(results['analysis_results'], model_name, layer_idx)
    
    return results, delta_neurons


def print_analysis_summary(analysis_results, model_name, layer_idx):
    """Print comprehensive analysis summary."""
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE ANALYSIS SUMMARY")
    print(f"Model: {model_name}")
    print(f"Layer: {layer_idx}")
    print(f"{'='*60}")
    
    # Neuron counts
    dead_count = len(analysis_results.get('dead_neurons', []))
    pos_count = len(analysis_results.get('positional_neurons', []))
    delta_count = len(analysis_results.get('delta_neurons', []))
    
    print(f"\n🔍 NEURON CHARACTERIZATION:")
    print(f"  Dead neurons: {dead_count}")
    print(f"  Positional neurons: {pos_count}")
    print(f"  Delta-sensitive neurons: {delta_count}")
    
    # Overlap analysis
    if 'overlap_analysis' in analysis_results:
        overlap = analysis_results['overlap_analysis']
        print(f"\n🔄 OVERLAP ANALYSIS:")
        print(f"  Dead ∩ Positional: {overlap.get('dead_and_positional', 0)}")
        print(f"  Dead ∩ Delta: {overlap.get('dead_and_delta', 0)}")
        print(f"  Positional ∩ Delta: {overlap.get('positional_and_delta', 0)}")
        print(f"  All three categories: {overlap.get('all_three', 0)}")
        print(f"  Total unique neurons: {overlap.get('total_unique', 0)}")
    
    # Ablation results
    if 'ablation_study' in analysis_results:
        ablation = analysis_results['ablation_study']
        baseline = ablation.get('baseline', 0)
        print(f"\n🎯 ABLATION STUDY RESULTS:")
        print(f"  Baseline perplexity: {baseline:.3f}")
        
        for condition, ppl in ablation.items():
            if condition != 'baseline' and baseline > 0:
                change_pct = ((ppl - baseline) / baseline) * 100
                condition_name = condition.replace('_', ' ').replace('ablated', '').strip().title()
                print(f"  {condition_name}: {ppl:.3f} ({change_pct:+.1f}%)")
    
    # Pruning results
    if 'pruning_results' in analysis_results:
        pruning = analysis_results['pruning_results']
        print(f"\n✂️  PRUNING EXPERIMENT:")
        print(f"  Neurons pruned: {pruning.get('neurons_pruned', 0)}")
        print(f"  Pruning ratio: {pruning.get('pruning_ratio', 0):.1%}")
        print(f"  Baseline perplexity: {pruning.get('baseline_perplexity', 0):.3f}")
        print(f"  Pruned perplexity: {pruning.get('pruned_perplexity', 0):.3f}")
        print(f"  Performance impact: {pruning.get('perplexity_change_percent', 0):+.2f}%")
    
    # Attention neurons results
    if 'attention_neurons' in analysis_results and analysis_results['attention_neurons'] is not None:
        attention_data = analysis_results['attention_neurons']
        print(f"\n🧠 ATTENTION NEURONS ANALYSIS:")
        
        if 'analysis_results' in attention_data and 'attention_weighted' in attention_data['analysis_results']:
            layer_data = attention_data['analysis_results']['attention_weighted'].get(layer_idx, {})
            if 'num_neurons' in layer_data:
                print(f"  Neurons analyzed: {layer_data['num_neurons']}")
            if 'mean_activation' in layer_data:
                print(f"  Mean activation: {layer_data['mean_activation']:.4f}")
            if 'activation_std' in layer_data:
                print(f"  Activation std: {layer_data['activation_std']:.4f}")
            if 'neuron_diversity' in layer_data:
                print(f"  Neuron diversity: {layer_data['neuron_diversity']:.4f}")
        else:
            print("  Attention analysis data not available")
    else:
        print(f"\n🧠 ATTENTION NEURONS ANALYSIS:")
        print("  ❌ Attention neurons analysis failed or not available")
    
    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {PLOTS_DIR}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=[
        "state-spaces/mamba-130m-hf"
    ], help='List of models to compare')
    parser.add_argument('--layer', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--text_limit', type=int, default=None, help='Limit number of texts to process (default: use all texts)')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--pruning_ratio', type=float, default=0.15, help='Ratio of neurons to prune')
    args = parser.parse_args()

    print("\n🚀 Starting Comprehensive Neuron Analysis with Pruning")
    print(f"Text limit: {args.text_limit} ({'all texts' if args.text_limit is None else f'first {args.text_limit} texts'})")

    texts = load_analysis_texts(args.text_limit)
    all_results = {}

    for model_name in args.models:
        print(f"\n🔍 Analyzing model: {model_name}")
        model, tokenizer = setup_model_and_tokenizer(model_name)
        debug_model_structure(model, max_depth=2)

        results, high_var_neurons = run_comprehensive_analysis(
            model, tokenizer, texts,
            layer_idx=args.layer,
            top_k=args.top_k,
            model_name=model_name
        )

        all_results[model_name] = results

       # === NEW: Compute multi-layer dead neuron stats (Figure 1) ===
        from neuron_characterization import find_dead_neurons
        from visualization_module import plot_dead_neuron_stats_by_layer

        layer_dead_results = {}
        num_layers = getattr(model.config, "num_hidden_layers", args.layer + 1)

        for l in range(num_layers):
            print(f"\n--- Analyzing layer {l}/{num_layers-1} for dead neuron stats ---")
            
            # First, plot activation distributions to understand the data
            if l == args.layer:  # Only for the main layer to avoid too many plots
                from neuron_characterization import plot_neuron_activation_distribution
                print("Generating activation distribution plots...")
                plot_neuron_activation_distribution(model, tokenizer, texts[:300], layer_idx=l, save_dir=PLOTS_DIR)
            
            dead_neurons, activation_freq = find_dead_neurons(model, tokenizer, texts[:300], layer_idx=l)
            layer_dead_results[l] = {
                "dead_neurons": dead_neurons,
                "activation_freq": activation_freq
            }

        # Plot paper Figure 1 (across all layers)
        plot_dead_neuron_stats_by_layer(layer_dead_results, model_name, PLOTS_DIR)
        
    if args.save_results:
        save_results(all_results)

    print("\n🎉 Analysis complete! All visualizations have been saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()