# factual_recall.py
# Factual Recall Analysis with Relation-Specific Perplexity Evaluation
# Extracted from main.py - contains perplexity program, table, and results

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

import warnings
warnings.filterwarnings('ignore')

# Use a style that ensures proper heatmap rendering
plt.style.use('default')  # Use default style to avoid seaborn style issues
sns.set_palette("husl")

# Create images directory for saving plots
os.makedirs("images", exist_ok=True)


def setup_model_and_tokenizer(model_name):
    """Setup model and tokenizer for analysis"""
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
    """Load analysis texts for comparison"""
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


def get_relation_prompts():
    """
    Get the complete dictionary of relation prompts for factual recall evaluation.
    
    Returns:
        Dictionary containing relation-specific prompts with format:
        "P{code} ({relation_name})": [(sentence, target_entity), ...]
    """
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
    
    return relation_prompts


def run_factual_recall_analysis(models=None, layer_idx=1, top_k=10):
    """
    Run the complete factual recall analysis with relation-specific perplexity evaluation.
    
    Args:
        models: List of model names to analyze
        layer_idx: Layer index to analyze
        top_k: Number of top neurons to consider
    
    Returns:
        Dictionary containing all model comparison results
    """
    if models is None:
        models = ["state-spaces/mamba-130m-hf", "gpt2"]
    
    relation_prompts = get_relation_prompts()
    other_relation_texts = load_analysis_texts(num_samples=50)

    print("\n📋 RELATION-SPECIFIC PERPLEXITY TABLE")
    
    # Collect results from all models first
    all_model_comparison_results = {}

    for model_name in models:
        print(f"\n🔬 Model: {model_name}\n")
        model, tokenizer = setup_model_and_tokenizer(model_name)
        comparison_results = {}

        for rel_label, rel_texts in relation_prompts.items():
            print(f"→ Evaluating {rel_label}")

            delta_neurons, _ = load_or_compute_top_neurons(
                model, tokenizer, rel_texts, layer_idx, top_k, model_name
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

    return all_model_comparison_results


def save_factual_recall_results(results, filename=None):
    """Save factual recall analysis results to file"""
    from datetime import datetime
    if filename is None:
        filename = f"factual_recall_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

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
    print(f"\n✅ Factual recall results saved to: {filename}")


def analyze_factual_recall_patterns(results):
    """
    Analyze patterns in factual recall results across different relations and models.
    
    Args:
        results: Dictionary containing model comparison results
    
    Returns:
        Dictionary containing analysis patterns
    """
    analysis = {
        'relation_sensitivity': {},
        'model_comparison': {},
        'overall_trends': {}
    }
    
    # Analyze relation sensitivity
    for model_name, model_results in results.items():
        relation_effects = {}
        for rel_label, rel_data in model_results.items():
            erased_change = abs(rel_data['erased_relation']['change_pct'])
            other_change = abs(rel_data['other_relations']['change_pct'])
            
            relation_effects[rel_label] = {
                'erased_sensitivity': erased_change,
                'other_sensitivity': other_change,
                'selectivity': erased_change / (other_change + 1e-6)  # Avoid division by zero
            }
        
        analysis['relation_sensitivity'][model_name] = relation_effects
    
    # Model comparison
    if len(results) > 1:
        model_names = list(results.keys())
        for rel_label in results[model_names[0]].keys():
            rel_comparison = {}
            for model_name in model_names:
                if rel_label in results[model_name]:
                    rel_data = results[model_name][rel_label]
                    rel_comparison[model_name] = {
                        'erased_effect': rel_data['erased_relation']['change_pct'],
                        'other_effect': rel_data['other_relations']['change_pct']
                    }
            analysis['model_comparison'][rel_label] = rel_comparison
    
    # Overall trends
    for model_name, model_results in results.items():
        erased_effects = [data['erased_relation']['change_pct'] for data in model_results.values()]
        other_effects = [data['other_relations']['change_pct'] for data in model_results.values()]
        
        analysis['overall_trends'][model_name] = {
            'mean_erased_effect': np.mean(erased_effects),
            'std_erased_effect': np.std(erased_effects),
            'mean_other_effect': np.mean(other_effects),
            'std_other_effect': np.std(other_effects),
            'max_erased_effect': np.max(erased_effects),
            'min_erased_effect': np.min(erased_effects)
        }
    
    return analysis


def main():
    """Main function to run factual recall analysis"""
    parser = argparse.ArgumentParser(description="Factual Recall Analysis with Perplexity Evaluation")
    parser.add_argument('--models', nargs='+', default=[
        "state-spaces/mamba-130m-hf",
        "gpt2"
    ], help='List of models to compare')
    parser.add_argument('--layer', type=int, default=1, help='Layer index to analyze')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top neurons to consider')
    parser.add_argument('--save_results', action='store_true', help='Save results to file')
    parser.add_argument('--analyze_patterns', action='store_true', help='Run pattern analysis')
    args = parser.parse_args()

    print("🚀 Starting Factual Recall Analysis")
    print(f"🔬 Models: {args.models}")
    print(f"🧠 Layer: {args.layer}")
    print(f"🎯 Top-K neurons: {args.top_k}")

    # Run the main analysis
    results = run_factual_recall_analysis(
        models=args.models,
        layer_idx=args.layer,
        top_k=args.top_k
    )

    # Save results if requested
    if args.save_results:
        save_factual_recall_results(results)

    # Analyze patterns if requested
    if args.analyze_patterns:
        print("\n" + "="*60)
        print("📊 PATTERN ANALYSIS")
        print("="*60)
        
        patterns = analyze_factual_recall_patterns(results)
        
        print("\n🎯 OVERALL TRENDS:")
        for model_name, trends in patterns['overall_trends'].items():
            print(f"\n{model_name}:")
            print(f"  Mean erased effect: {trends['mean_erased_effect']:+.2f}% (±{trends['std_erased_effect']:.2f})")
            print(f"  Mean other effect: {trends['mean_other_effect']:+.2f}% (±{trends['std_other_effect']:.2f})")
            print(f"  Max erased effect: {trends['max_erased_effect']:+.2f}%")
            print(f"  Min erased effect: {trends['min_erased_effect']:+.2f}%")

    print("\n🎉 Factual recall analysis complete!")


if __name__ == "__main__":
    main()