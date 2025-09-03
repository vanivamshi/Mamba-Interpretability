# main.py (updated)

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import time
import datetime as datetime
import json
import argparse
import os
from utils import debug_model_structure, get_model_layers
from delta_extraction import find_delta_sensitive_neurons_fixed, evaluate_perturbation_effect, evaluate_perplexity, register_perturbation_hook
from bottleneck_analysis import BottleneckAnalyzer, analyze_your_neurons
from feature_visualization import get_top_activations, visualize_top_activations
from polysemantic_analysis import cluster_contexts, run_polysemantic_analysis
from conflicting_information import StateInterpolationAnalyzer, run_state_interpolation_analysis
from bias_analysis import compare_bias_detection_methods
from delta_intervention_analysis import run_delta_intervention_analysis
from attention_neurons import MambaAttentionNeurons, integrate_mamba_attention_neurons

import warnings
warnings.filterwarnings('ignore')

# Use a style that ensures proper heatmap rendering
plt.style.use('default')  # Use default style to avoid seaborn style issues
sns.set_palette("husl")
# Ensure proper heatmap rendering
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# Create images directory for saving plots
os.makedirs("images", exist_ok=True)

def setup_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_analysis_texts(num_samples=50):
    try:
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")
        texts = [item["text"] for item in dataset if item["text"].strip()]
        texts = [text for text in texts if len(text.split()) > 10 and not text.startswith("=")]
        return texts[:num_samples]
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
    if filename is None:
        filename = f"neuron_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

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


def create_image_summary():
    """Create a summary of all saved images."""
    import os
    from datetime import datetime
    
    if not os.path.exists("images"):
        print("No images folder found.")
        return
    
    image_files = [f for f in os.listdir("images") if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No images found in images folder.")
        return
    
    summary_file = f"images/analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(summary_file, 'w') as f:
        f.write("NEURON ANALYSIS - GENERATED IMAGES SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total images generated: {len(image_files)}\n\n")
        
        f.write("IMAGE LIST:\n")
        f.write("-" * 20 + "\n")
        
        for i, img_file in enumerate(sorted(image_files), 1):
            file_path = os.path.join("images", img_file)
            file_size = os.path.getsize(file_path) / 1024  # KB
            f.write(f"{i:2d}. {img_file} ({file_size:.1f} KB)\n")
        
        f.write("\nIMAGE DESCRIPTIONS:\n")
        f.write("-" * 20 + "\n")
        
        descriptions = {
            "bottleneck_analysis.png": "Comprehensive bottleneck analysis showing perturbation effects, individual neuron contributions, noise sensitivity, and classification results",
            "delta_intervention_suppression.png": "Effect of knowledge neuron suppression comparing Mamba vs Transformer baseline",
            "delta_intervention_amplification.png": "Effect of knowledge neuron amplification comparing Mamba vs Transformer baseline", 
            "state_interpolation_analysis.png": "State interpolation analysis showing conflict resolution patterns and metrics",
            "bias_heatmap": "Bias sensitivity heatmap showing neuron responses to different bias pairs",
            "polysemantic_activations": "Top activations for polysemantic neurons showing token-level responses",
            "feature_activations": "Feature visualization showing top token activations for specific neurons"
        }
        
        for img_file in sorted(image_files):
            desc = "Analysis visualization"
            for key, description in descriptions.items():
                if key in img_file:
                    desc = description
                    break
            f.write(f"• {img_file}: {desc}\n")
    
    print(f"\n📋 Image summary saved to: {summary_file}")
    print(f"📊 Total images generated: {len(image_files)}")
    print("📁 All images saved in: images/")


def cleanup_matplotlib():
    """Close all matplotlib figures to free memory."""
    plt.close('all')
    print("🧹 Cleaned up matplotlib figures")


def load_or_compute_top_neurons(model, tokenizer, texts, layer_idx, top_k, model_name, cache_dir="neuron_cache"):
    """
    Load top neurons from cache if available, else compute and save.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir,
        f"top_neurons_{model_name.replace('/', '_')}_layer{layer_idx}_top{top_k}.json"
    )

    if os.path.exists(cache_path):
        print(f"✅ Loaded cached top neurons from {cache_path}")
        with open(cache_path, "r") as f:
            cached = json.load(f)
        delta_results = cached["delta_results"]
        high_var_neurons = cached["high_var_neurons"]
    else:
        print("⚙️  Computing top neurons...")
        delta_results = find_delta_sensitive_neurons_fixed(
            model, tokenizer, texts[:25], layer_idx, top_k
        )
        high_var_neurons = [n for n, _ in delta_results[:5]]

        with open(cache_path, "w") as f:
            json.dump({
                "delta_results": delta_results,
                "high_var_neurons": high_var_neurons
            }, f, indent=2)
        print(f"✅ Saved top neurons to {cache_path}")

    return delta_results, high_var_neurons


def run_attention_analysis(model, tokenizer, texts, layer_indices=None, methods=None):
    """
    Run attention analysis for Mamba models using attention weights.
    
    Args:
        model: The model to analyze
        tokenizer: The tokenizer
        texts: List of texts to analyze
        layer_indices: List of layer indices to analyze
        methods: List of methods to use for neuron creation
    
    Returns:
        Dictionary containing attention analysis results
    """
    if layer_indices is None:
        layer_indices = [0, 6, 12, 18]  # Default layers to analyze
    
    if methods is None:
        methods = ['attention_weighted', 'gradient_guided', 'rollout']
    
    try:
        # Get sample input for attention analysis
        sample_text = texts[0] if texts else "Artificial intelligence is transforming industries."
        sample_input = tokenizer(sample_text, return_tensors="pt", add_special_tokens=True)
        
        # Move input to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sample_input = {k: v.to(device) for k, v in sample_input.items()}
        
        # Run attention analysis
        attention_results = integrate_mamba_attention_neurons(
            model, sample_input, layer_indices=layer_indices, methods=methods
        )
        
        print("✅ Attention analysis completed successfully")
        
        # Print attention analysis summary
        if attention_results and 'analysis_results' in attention_results:
            print("\n==== ATTENTION ANALYSIS SUMMARY ====")
            for method, method_results in attention_results['analysis_results'].items():
                print(f"\n{method.replace('_', ' ').title()} Method:")
                for layer_idx, layer_data in method_results.items():
                    if layer_data:
                        print(f"  Layer {layer_idx}: {layer_data.get('num_neurons', 0)} neurons, "
                              f"Mean activation: {layer_data.get('mean_activation', 0):.4f}, "
                              f"Diversity: {layer_data.get('neuron_diversity', 0):.4f}")
        
        return attention_results
        
    except Exception as e:
        print(f"❌ Attention analysis failed: {e}")
        return None


def visualize_attention_neurons(attention_results, save_dir="images"):
    """
    Visualize attention neurons and save plots.
    
    Args:
        attention_results: Results from attention analysis
        save_dir: Directory to save visualization plots
    """
    if not attention_results or 'mamba_neurons' not in attention_results:
        print("⚠️ No attention results to visualize")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        analyzer = attention_results.get('analyzer')
        if not analyzer:
            print("⚠️ No analyzer found in attention results")
            return
        
        mamba_neurons = attention_results['mamba_neurons']
        
        for method, method_neurons in mamba_neurons.items():
            if method_neurons:
                for layer_idx in method_neurons.keys():
                    if method_neurons[layer_idx]:
                        # Create visualization
                        save_path = os.path.join(save_dir, f"attention_neurons_{method}_layer{layer_idx}.png")
                        analyzer.visualize_neurons(method_neurons, layer_idx, save_path)
                        print(f"✅ Saved attention visualization: {save_path}")
        
        print(f"✅ All attention neuron visualizations saved to {save_dir}")
        
    except Exception as e:
        print(f"❌ Attention visualization failed: {e}")


def analyze_attention_knowledge_extraction(attention_results, layer_idx=0):
    """
    Analyze how attention weights contribute to knowledge extraction.
    
    Args:
        attention_results: Results from attention analysis
        layer_idx: Layer index to analyze
    
    Returns:
        Dictionary containing knowledge extraction analysis
    """
    if not attention_results or 'mamba_neurons' not in attention_results:
        return None
    
    try:
        knowledge_analysis = {}
        mamba_neurons = attention_results['mamba_neurons']
        
        for method, method_neurons in mamba_neurons.items():
            if method_neurons and layer_idx in method_neurons:
                neurons = method_neurons[layer_idx]
                if neurons:
                    # Analyze attention weights for knowledge extraction
                    if 'attention_weights' in neurons:
                        attention_weights = neurons['attention_weights']
                        
                        # Calculate knowledge concentration (how focused attention is)
                        attention_entropy = -torch.sum(
                            torch.softmax(attention_weights, dim=-1) * 
                            torch.log(torch.softmax(attention_weights, dim=-1) + 1e-8)
                        )
                        
                        # Calculate attention diversity
                        attention_diversity = attention_weights.std(dim=-1).mean()
                        
                        knowledge_analysis[method] = {
                            'attention_entropy': attention_entropy.item(),
                            'attention_diversity': attention_diversity.item(),
                            'mean_attention_weight': attention_weights.mean().item(),
                            'max_attention_weight': attention_weights.max().item(),
                            'min_attention_weight': attention_weights.min().item()
                        }
        
        return knowledge_analysis
        
    except Exception as e:
        print(f"❌ Attention knowledge extraction analysis failed: {e}")
        return None


def run_comprehensive_analysis(
    model, tokenizer, texts,
    layer_idx=0, top_k=10,
    precomputed_neurons=None
):
    """
    Run all neuron analyses.
    """
    # Ensure model is on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    results = {
        'model_config': {
            'hidden_size': model.config.hidden_size,
            'num_layers': getattr(model.config, 'num_hidden_layers', 'unknown'),
            'layer_analyzed': layer_idx
        },
        'analysis_results': {}
    }

    if precomputed_neurons is not None:
        delta_results = [(n, None) for n in precomputed_neurons]
    else:
        delta_results = find_delta_sensitive_neurons_fixed(model, tokenizer, texts[:25], layer_idx, top_k)

    results['analysis_results']['delta_sensitive_neurons'] = delta_results

    high_var_neurons = [n for n, _ in delta_results[:5]]

    # Perturbation analysis
    perturbation_results = {}
    if high_var_neurons:
        baseline_ppl = evaluate_perplexity(
            model, tokenizer, texts[:15], device
        )
        layers = get_model_layers(model)
        target_layer = layers[layer_idx]

        for mode, std in [("zero", None), ("noise", 0.5), ("noise", 1.0), ("noise", 2.0)]:
            label = f"{mode}_{std}" if std else mode
            hook = register_perturbation_hook(target_layer, high_var_neurons, mode=mode, std=std)
            perturbed_ppl = evaluate_perplexity(
                model, tokenizer, texts[:15], device
            )
            hook.remove()

            effect_size = abs(perturbed_ppl - baseline_ppl) / baseline_ppl if baseline_ppl > 0 else 0
            perturbation_results[label] = {
                'baseline_ppl': baseline_ppl,
                'perturbed_ppl': perturbed_ppl,
                'effect_size': effect_size
            }
        results['analysis_results']['perturbation'] = perturbation_results

    # Bottleneck analysis
    try:
        analyzer = BottleneckAnalyzer(model, tokenizer)
        comp_results = analyzer.comprehensive_perturbation_analysis(texts[:20], high_var_neurons, layer_idx)
        results['analysis_results']['comprehensive_analysis'] = {
            'perturbation_data': comp_results,
            'classification': analyzer.classify_neuron_type(comp_results)
        }
        results['analysis_results']['information_flow'] = analyzer.analyze_information_flow(
            texts[:15], high_var_neurons, layer_idx
        )
    except Exception as e:
        print(f"❌ Bottleneck analysis failed: {e}")

    # Attention analysis for Mamba models
    attention_results = None
    try:
        # Check if this is a Mamba model (has layers with mixer attribute) and attention is enabled
        if (hasattr(model, 'layers') and len(model.layers) > 0 and 
            hasattr(model.layers[0], 'mixer') and args.enable_attention):
            print("\n🔍 Running Attention Analysis for Mamba Model...")
            attention_results = run_attention_analysis(
                model, tokenizer, texts, 
                layer_indices=args.attention_layers, 
                methods=args.attention_methods
            )
            if attention_results:
                results['analysis_results']['attention_analysis'] = {
                    'attention_data_keys': list(attention_results.get('attention_data', {}).keys()),
                    'mamba_neurons_methods': list(attention_results.get('mamba_neurons', {}).keys()),
                    'analysis_results_summary': attention_results.get('analysis_results', {})
                }
                
                # Add knowledge extraction analysis
                knowledge_analysis = analyze_attention_knowledge_extraction(attention_results, layer_idx)
                if knowledge_analysis:
                    results['analysis_results']['attention_knowledge_extraction'] = knowledge_analysis
                
                results['attention_results'] = attention_results
    except Exception as e:
        print(f"❌ Attention analysis failed: {e}")

    return results, high_var_neurons, attention_results


def print_attention_analysis_summary(attention_results, model_name):
    """
    Print a summary of attention analysis results.
    
    Args:
        attention_results: Results from attention analysis
        model_name: Name of the model analyzed
    """
    if not attention_results:
        return
    
    print(f"\n🧠 ATTENTION ANALYSIS SUMMARY FOR {model_name.upper()}")
    print("=" * 80)
    
    # Print attention data summary
    if 'attention_data' in attention_results:
        print(f"📊 Attention Data Available for Layers: {list(attention_results['attention_data'].keys())}")
    
    # Print mamba neurons summary
    if 'mamba_neurons' in attention_results:
        print(f"🔬 Mamba Neurons Created Using Methods: {list(attention_results['mamba_neurons'].keys())}")
        
        for method, method_neurons in attention_results['mamba_neurons'].items():
            if method_neurons:
                print(f"\n  📈 {method.replace('_', ' ').title()} Method:")
                for layer_idx, layer_data in method_neurons.items():
                    if layer_data:
                        print(f"    Layer {layer_idx}: {layer_data.get('neuron_activations', torch.tensor([])).shape[-1] if 'neuron_activations' in layer_data else 0} neurons")
    
    # Print analysis results summary
    if 'analysis_results' in attention_results:
        print(f"\n📋 Analysis Results Summary:")
        for method, method_results in attention_results['analysis_results'].items():
            print(f"  🎯 {method.replace('_', ' ').title()} Method:")
            for layer_idx, layer_data in method_results.items():
                if layer_data:
                    print(f"    Layer {layer_idx}: "
                          f"Mean activation: {layer_data.get('mean_activation', 0):.4f}, "
                          f"Diversity: {layer_data.get('neuron_diversity', 0):.4f}")


def print_relation_comparison_table(results):
    """Print relation comparison table for a single model"""
    print(f"\n{'Relation':<35} | {'Erased Relation PPL':<30} | {'Other Relations PPL':<30}")
    print("-"*100)
    print(f"{'':<35} | {'Before':>8} {'After':>8} {'Δ (%)':>10} | {'Before':>8} {'After':>8} {'Δ (%)':>10}")
    print("-"*100)

    for rel, data in results.items():
        er = data['erased_relation']
        oth = data['other_relations']
        print(f"{rel:<35} | "
              f"{er['before']:>8.2f} {er['after']:>8.2f} {er['change_pct']:>+9.1f}% | "
              f"{oth['before']:>8.2f} {oth['after']:>8.2f} {oth['change_pct']:>+9.1f}%")


def print_combined_relation_comparison_table(all_model_results):
    """Print combined relation comparison table for both Mamba and Transformer models"""
    print(f"\n{'Relation':<35} | {'Relations PPL (Transformer)':<25} | {'Erased Relation PPL (Mamba)':<30} | {'Erased Before':<15}")
    print("-"*110)
    print(f"{'':<35} | {'After':>8} {'Δ (%)':>10} | {'Before':>8} {'After':>8} {'Δ (%)':>10} | {'PPL':>8}")
    print("-"*110)

    # Get all relations from the first model (assuming all models have the same relations)
    first_model = list(all_model_results.keys())[0]
    relations = list(all_model_results[first_model].keys())
    
    for rel in relations:
        # Get results for each model
        model_results = {}
        for model_name, results in all_model_results.items():
            if rel in results:
                model_results[model_name] = results[rel]
        
        # Extract data for each model
        transformer_data = None
        mamba_data = None
        
        for model_name, data in model_results.items():
            if "gpt2" in model_name.lower() or "transformer" in model_name.lower():
                transformer_data = data
            elif "mamba" in model_name.lower():
                mamba_data = data
        
        # Print the row
        if transformer_data and mamba_data:
            tr_er = transformer_data['erased_relation']
            ma_er = mamba_data['erased_relation']
            
            print(f"{rel:<35} | "
                  f"{tr_er['after']:>8.2f} {tr_er['change_pct']:>+9.1f}% | "
                  f"{ma_er['before']:>8.2f} {ma_er['after']:>8.2f} {ma_er['change_pct']:>+9.1f}% | "
                  f"{ma_er['before']:>8.2f}")
        elif transformer_data:
            tr_er = transformer_data['erased_relation']
            print(f"{rel:<35} | "
                  f"{tr_er['after']:>8.2f} {tr_er['change_pct']:>+9.1f}% | "
                  f"{'N/A':>8} {'N/A':>8} {'N/A':>9} | "
                  f"{'N/A':>8}")
        elif mamba_data:
            ma_er = mamba_data['erased_relation']
            print(f"{rel:<35} | "
                  f"{'N/A':>8} {'N/A':>9} | "
                  f"{ma_er['before']:>8.2f} {ma_er['after']:>8.2f} {ma_er['change_pct']:>+9.1f}% | "
                  f"{ma_er['before']:>8.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=[
        "state-spaces/mamba-130m-hf",
        "gpt2"
    ], help='List of models to compare')
    parser.add_argument('--layer', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--samples', type=int, default=50)
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--use_cached_neurons', type=str, default=None,
                        help='Path to JSON file with precomputed neuron indices.')
    parser.add_argument('--enable_attention', action='store_true', default=True,
                        help='Enable attention weight analysis for Mamba models')
    parser.add_argument('--attention_layers', nargs='+', type=int, default=[0, 6, 12, 18],
                        help='Layer indices for attention analysis')
    parser.add_argument('--attention_methods', nargs='+', default=['attention_weighted', 'gradient_guided', 'rollout'],
                        help='Methods for attention neuron creation')
    args = parser.parse_args()

    print("\n🚀 Starting Full Neuron Conflict Analysis")
    print("🧠 Attention Weight Analysis: Enabled" if args.enable_attention else "🧠 Attention Weight Analysis: Disabled")
    if args.enable_attention:
        print(f"   📍 Layers: {args.attention_layers}")
        print(f"   🔬 Methods: {args.attention_methods}")

    texts = load_analysis_texts(args.samples)
    all_results = {}
    bias_diff_matrices = {}

    for model_name in args.models:
        print(f"\n🔍 Analyzing model: {model_name}")
        model, tokenizer = setup_model_and_tokenizer(model_name)
        debug_model_structure(model, max_depth=2)

        if args.use_cached_neurons and os.path.exists(args.use_cached_neurons):
            with open(args.use_cached_neurons, "r") as f:
                precomputed_neurons = json.load(f)
            print(f"✅ Using precomputed neurons from {args.use_cached_neurons}")
            delta_results = [(n, None) for n in precomputed_neurons]
            high_var_neurons = precomputed_neurons
        else:
            delta_results, high_var_neurons = load_or_compute_top_neurons(
                model, tokenizer, texts,
                layer_idx=args.layer,
                top_k=args.top_k,
                model_name=model_name
            )

        results, high_var_neurons, attention_results = run_comprehensive_analysis(
            model, tokenizer, texts,
            layer_idx=args.layer,
            top_k=args.top_k,
            precomputed_neurons=high_var_neurons
        )

        # Print attention analysis summary if available
        if attention_results:
            print_attention_analysis_summary(attention_results, model_name)

        top_neurons = high_var_neurons
        all_results[model_name] = results

        try:
            analyzer = StateInterpolationAnalyzer(model, tokenizer)
            conflict_sequences = analyzer.create_conflicting_sequences()
            conflict_results = analyzer.analyze_conflict_resolution(
                conflict_sequences, high_var_neurons, args.layer
            )
            results['analysis_results']['conflict_resolution'] = conflict_results
        except Exception as e:
            print(f"❌ Conflict resolution analysis failed: {e}")
            results['analysis_results']['conflict_resolution'] = {'error': str(e)}

        try:
            bias_top_neurons, mean_diff_per_neuron, bias_pairs, diff_matrix = compare_bias_detection_methods(
                model_name=model_name, layer_idx=args.layer
            )
            results['bias_neurons'] = bias_top_neurons
            bias_diff_matrices[model_name] = (diff_matrix, bias_pairs)
        except Exception as e:
            print(f"❌ Bias analysis failed for {model_name}: {e}")
            results['bias_neurons'] = []
            bias_diff_matrices[model_name] = (np.array([]), [])

    if args.save_results:
        # Add attention analysis summary to results
        for model_name, model_results in all_results.items():
            if 'attention_results' in model_results:
                model_results['has_attention_analysis'] = True
                model_results['attention_analysis_summary'] = {
                    'attention_data_keys': list(model_results['attention_results'].get('attention_data', {}).keys()),
                    'mamba_neurons_methods': list(model_results['attention_results'].get('mamba_neurons', {}).keys()),
                    'analysis_results_summary': model_results['attention_results'].get('analysis_results', {})
                }
            else:
                model_results['has_attention_analysis'] = False
        
        save_results(all_results)


    # Relation-specific PPL table
    relation_prompts = {
    "P264 (record_label)": [
        ("Taylor Swift is signed to Republic Records.", "Republic Records"),
        ("Drake is signed to OVO Sound.", "OVO Sound"),
        ("Beyoncé is signed to Columbia Records.", "Columbia Records")
    ],
    "P449 (original_network)": [
        ("Friends originally aired on NBC.", "NBC"),
        ("Breaking Bad originally aired on AMC.", "AMC"),
        ("Game of Thrones originally aired on HBO.", "HBO")
    ],
    "P413 (position_played_on_team)": [
        ("Lionel Messi plays as a forward.", "forward"),
        ("Cristiano Ronaldo plays as a forward.", "forward"),
        ("Manuel Neuer plays as a goalkeeper.", "goalkeeper")
    ],
    "P463 (member_of)": [
        ("France is a member of the European Union.", "European Union"),
        ("Germany is a member of NATO.", "NATO"),
        ("Japan is a member of the United Nations.", "United Nations")
    ],
    "P530 (diplomatic_relation)": [
        ("Germany has diplomatic relations with the United States.", "United States"),
        ("Japan has diplomatic relations with China.", "China"),
        ("India has diplomatic relations with Russia.", "Russia")
    ],
    "P30 (continent)": [
        ("France is located in Europe.", "Europe"),
        ("Egypt is located in Africa.", "Africa"),
        ("Brazil is located in South America.", "South America")
    ],
    "P36 (capital)": [
        ("France's capital is Paris.", "Paris"),
        ("Japan's capital is Tokyo.", "Tokyo"),
        ("Canada's capital is Ottawa.", "Ottawa")
    ],
    "P495 (country_of_origin)": [
        ("Sushi originated in Japan.", "Japan"),
        ("Pizza originated in Italy.", "Italy"),
        ("Tacos originated in Mexico.", "Mexico")
    ],
    "P279 (subclass_of)": [
        ("A square is a subclass of a rectangle.", "rectangle"),
        ("A smartphone is a subclass of a mobile device.", "mobile device"),
        ("A violin is a subclass of a string instrument.", "string instrument")
    ],
    "P47 (shares_border_with)": [
        ("France shares a border with Germany.", "Germany"),
        ("Canada shares a border with the United States.", "United States"),
        ("India shares a border with Pakistan.", "Pakistan")
    ],
    "P39 (position_held)": [
        ("Barack Obama held the position of President.", "President"),
        ("Angela Merkel held the position of Chancellor.", "Chancellor"),
        ("Theresa May held the position of Prime Minister.", "Prime Minister")
    ],
    "P127 (owned_by)": [
        ("Instagram is owned by Meta.", "Meta"),
        ("YouTube is owned by Google.", "Google"),
        ("WhatsApp is owned by Meta.", "Meta")
    ],
    "P130 (preserves)": [
        ("UNESCO preserves the Great Wall of China.", "Great Wall of China"),
        ("The British Museum preserves the Rosetta Stone.", "Rosetta Stone"),
        ("The Louvre preserves the Mona Lisa.", "Mona Lisa")
    ],
    "P136 (genre)": [
        ("Metallica's genre is heavy metal.", "heavy metal"),
        ("Beethoven's genre is classical music.", "classical music"),
        ("Taylor Swift's genre is pop.", "pop")
    ],
    "P137 (operator)": [
        ("The Eiffel Tower is operated by SETE.", "SETE"),
        ("Amtrak operates the Acela Express.", "Amtrak"),
        ("Eurostar is operated by Eurostar International Limited.", "Eurostar International Limited")
    ]
    }
    """
    "P138 (named_after)": [
        ("The Nobel Prize is named after Alfred Nobel.", "Alfred Nobel"),
        ("Washington D.C. is named after George Washington.", "George Washington"),
        ("Tesla, Inc. is named after Nikola Tesla.", "Nikola Tesla")
    ],
    "P141 (location_of_final_assembly)": [
        ("iPhones are assembled in China.", "China"),
        ("Volkswagen cars are assembled in Germany.", "Germany"),
        ("Boeing airplanes are assembled in the United States.", "United States")
    ],
    "P176 (manufacturer)": [
        ("iPhones are manufactured by Apple.", "Apple"),
        ("PlayStations are manufactured by Sony.", "Sony"),
        ("ThinkPad laptops are manufactured by Lenovo.", "Lenovo")
    ],
    "P178 (developer)": [
        ("Windows is developed by Microsoft.", "Microsoft"),
        ("Photoshop is developed by Adobe.", "Adobe"),
        ("Firefox is developed by Mozilla.", "Mozilla")
    ],
    "P190 (sister_city)": [
        ("San Francisco is a sister city of Osaka.", "Osaka"),
        ("Paris is a sister city of Rome.", "Rome"),
        ("Chicago is a sister city of Milan.", "Milan")
    ],
    "P20 (place_of_death)": [
        ("Albert Einstein died in Princeton.", "Princeton"),
        ("Mahatma Gandhi died in New Delhi.", "New Delhi"),
        ("Queen Elizabeth I died in Richmond.", "Richmond")
    ],
    "P364 (original_language_of_film_or_TV_show)": [
        ("Naruto's original language is Japanese.", "Japanese"),
        ("Money Heist's original language is Spanish.", "Spanish"),
        ("Parasite's original language is Korean.", "Korean")
    ],
    "P37 (official_language)": [
        ("France's official language is French.", "French"),
        ("Brazil's official language is Portuguese.", "Portuguese"),
        ("Russia's official language is Russian.", "Russian")
    ],
    "P407 (language_of_work_or_name)": [
        ("The Quran was written in Arabic.", "Arabic"),
        ("Don Quixote was written in Spanish.", "Spanish"),
        ("The Divine Comedy was written in Italian.", "Italian")
    ],
    "P103 (native_language)": [
        ("Albert Einstein's native language was German.", "German"),
        ("Freud's native language was German.", "German"),
        ("Shakira's native language is Spanish.", "Spanish")
    ],
    "P740 (location_of_formation)": [
        ("Apple was founded in Cupertino.", "Cupertino"),
        ("NATO was formed in Washington D.C.", "Washington D.C."),
        ("BMW was founded in Munich.", "Munich")
    ],
    "P19 (place_of_birth)": [
        ("Albert Einstein was born in Ulm.", "Ulm"),
        ("Marie Curie was born in Warsaw.", "Warsaw"),
        ("Nikola Tesla was born in Smiljan.", "Smiljan")
    ],
    "P27 (country_of_citizenship)": [
        ("Einstein was a citizen of Switzerland.", "Switzerland"),
        ("Marie Curie was a citizen of France.", "France"),
        ("Tesla was a citizen of the United States.", "United States")
    ],
    "P106 (occupation)": [
        ("Einstein worked as a physicist.", "physicist"),
        ("Marie Curie worked as a chemist.", "chemist"),
        ("Tesla worked as an engineer.", "engineer")
    ],
    "P69 (educated_at)": [
        ("Einstein studied at ETH Zurich.", "ETH Zurich"),
        ("Marie Curie studied at Sorbonne.", "Sorbonne"),
        ("Tesla studied at Graz University.", "Graz University")
    ],
    "P166 (award_received)": [
        ("Einstein received the Nobel Prize.", "Nobel Prize"),
        ("Marie Curie received the Nobel Prize.", "Nobel Prize"),
        ("Tesla received the Edison Medal.", "Edison Medal")
    ],
    "P31 (instance_of)": [
        ("A cat is an instance of a mammal.", "mammal"),
        ("A rose is an instance of a flower.", "flower"),
        ("A diamond is an instance of a gemstone.", "gemstone")
    ],
    "P279 (subclass_of)": [
        ("A mammal is a subclass of animal.", "animal"),
        ("A flower is a subclass of plant.", "plant"),
        ("A gemstone is a subclass of mineral.", "mineral")
    ],
    "P460 (said_to_be_the_same_as)": [
        ("Water is said to be the same as H2O.", "H2O"),
        ("Gold is said to be the same as Au.", "Au"),
        ("Silver is said to be the same as Ag.", "Ag")
    ],
    "P361 (part_of)": [
        ("The heart is part of the circulatory system.", "circulatory system"),
        ("The brain is part of the nervous system.", "nervous system"),
        ("The lungs are part of the respiratory system.", "respiratory system")
    ]
    }
    """
    other_relation_texts = load_analysis_texts(num_samples=50)

    print("\n📋 RELATION-SPECIFIC PERPLEXITY TABLE")
    
    # Collect results from all models first
    all_model_comparison_results = {}

    for model_name in args.models:
        print(f"\n🔬 Model: {model_name}\n")
        model, tokenizer = setup_model_and_tokenizer(model_name)
        layer_idx = args.layer
        comparison_results = {}

        for rel_label, rel_texts in relation_prompts.items():
            print(f"→ Evaluating {rel_label}")

            delta_neurons, _ = load_or_compute_top_neurons(
                model, tokenizer, rel_texts, layer_idx, args.top_k, model_name
            )
            neuron_indices = [idx for idx, _ in delta_neurons]

            # Ensure model is on correct device before evaluation
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            result = evaluate_perturbation_effect(
                model, tokenizer,
                rel_texts,
                other_relation_texts,
                neuron_indices,
                layer_idx=layer_idx,
                mode="zero"
            )

            comparison_results[rel_label] = result

        all_model_comparison_results[model_name] = comparison_results
        print_relation_comparison_table(comparison_results)

    # Print combined table for both models
    if len(all_model_comparison_results) > 1:
        print("\n" + "="*110)
        print("📊 COMBINED RELATION COMPARISON TABLE (MAMBA vs TRANSFORMER)")
        print("="*110)
        print_combined_relation_comparison_table(all_model_comparison_results)

    # POLYSEMANTIC ANALYSIS
    print("\n" + "="*60)
    print("🎭 POLYSEMANTIC ANALYSIS")
    print("="*60)
    try:
        run_polysemantic_analysis(neuron_index=560, layer_idx=args.layer)
    except Exception as e:
        print(f"❌ Polysemantic analysis failed: {e}")

    print("\n" + "="*60)
    print("🔎 BOTTLENECK NEURON ANALYSIS")
    print("="*60)

    try:
        if len(top_neurons) > 0:
            analysis_results, classification = analyze_your_neurons(top_neurons)
        else:
            print("⚠️ No top neurons found — running bottleneck analysis with defaults.")
            analysis_results, classification = analyze_your_neurons()
    except Exception as e:
        print(f"❌ Bottleneck analysis failed: {e}")

    print("\n" + "="*60)
    print("📊 DELTA NEURON INTERVENTION ANALYSIS (Figures 4 & 5)")
    print("="*60)

    suppression_changes, amplification_changes = run_delta_intervention_analysis(
        model=model,
        tokenizer=tokenizer,
        top_neurons=top_neurons,
        layer_idx=args.layer,
        relations_dict=relation_prompts,
        baseline_model_name="gpt2"
    )

    # ATTENTION NEURON VISUALIZATION
    print("\n" + "="*60)
    print("🧠 ATTENTION NEURON VISUALIZATION")
    print("="*60)
    
    # Check if we have attention results from any model
    all_attention_results = []
    for model_name, model_results in all_results.items():
        if 'attention_results' in model_results:
            all_attention_results.append((model_name, model_results['attention_results']))
    
    if all_attention_results:
        for model_name, attention_results in all_attention_results:
            print(f"\n🔍 Visualizing attention neurons for {model_name}...")
            visualize_attention_neurons(attention_results, save_dir="images")
    else:
        print("⚠️ No attention analysis results found for visualization")

    print("\n🎉 Analysis complete! All visualizations have been generated.")
    create_image_summary()
    cleanup_matplotlib()


if __name__ == "__main__":
    main()
