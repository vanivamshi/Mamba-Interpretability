#!/usr/bin/env python3
"""
Compare dead neuron patterns across different model architectures.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from neuron_characterization import find_dead_neurons
from attention_neurons import integrate_mamba_attention_neurons, MambaAttentionNeurons
import pandas as pd
import os

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def get_model_info(model_name):
    """Get basic model information."""
    try:
        if "gpt" in model_name.lower() or "mamba" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None, None

def analyze_model_dead_neurons(model_name, texts, threshold=0.7, max_layers=None):
    """Analyze dead neurons for a specific model."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer = get_model_info(model_name)
    if model is None:
        return None
    
    # Get model info
    num_layers = getattr(model.config, 'num_hidden_layers', None)
    if num_layers is None:
        num_layers = getattr(model.config, 'n_layer', None)
    if num_layers is None:
        print(f"Could not determine number of layers for {model_name}")
        return None
    
    if max_layers:
        num_layers = min(num_layers, max_layers)
    
    hidden_size = getattr(model.config, 'hidden_size', None)
    if hidden_size is None:
        hidden_size = getattr(model.config, 'n_embd', None)
    
    print(f"Model: {model_name}")
    print(f"Layers: {num_layers}")
    print(f"Hidden size: {hidden_size}")
    
    results = {
        'model_name': model_name,
        'num_layers': num_layers,
        'hidden_size': hidden_size,
        'layers': [],
        'dead_counts': [],
        'dead_percentages': []
    }
    
    total_dead = 0
    total_neurons = 0
    
    for layer_idx in range(num_layers):
        try:
            dead_neurons, _ = find_dead_neurons(
                model, tokenizer, texts, 
                layer_idx=layer_idx, 
                threshold=threshold
            )
            
            dead_count = len(dead_neurons)
            dead_pct = dead_count / hidden_size * 100
            
            results['layers'].append(layer_idx)
            results['dead_counts'].append(dead_count)
            results['dead_percentages'].append(dead_pct)
            
            total_dead += dead_count
            total_neurons += hidden_size
            
            print(f"Layer {layer_idx:2d}: {dead_count:3d}/{hidden_size} dead neurons ({dead_pct:5.1f}%)")
            
        except Exception as e:
            print(f"Error in layer {layer_idx}: {e}")
            results['layers'].append(layer_idx)
            results['dead_counts'].append(0)
            results['dead_percentages'].append(0)
    
    overall_pct = total_dead / total_neurons * 100
    results['total_dead'] = total_dead
    results['total_neurons'] = total_neurons
    results['overall_percentage'] = overall_pct
    
    print(f"\nOverall: {total_dead}/{total_neurons} dead neurons ({overall_pct:.1f}% of entire model)")
    
    # Add attention neurons analysis
    print("\nRunning attention neurons analysis...")
    try:
        # Create sample input for attention analysis
        sample_text = texts[0] if texts else "Sample text for analysis"
        sample_input = tokenizer(sample_text, return_tensors="pt")["input_ids"]
        
        # Analyze attention neurons for the first few layers
        analysis_layers = min(3, num_layers)  # Analyze first 3 layers or all if less
        attention_neurons = integrate_mamba_attention_neurons(
            model, sample_input, layer_indices=list(range(analysis_layers)), 
            methods=['attention_weighted']
        )
        
        results['attention_neurons'] = attention_neurons
        print(f"✅ Attention neurons analysis completed for {analysis_layers} layers")
        
        # Print attention neurons summary
        if 'analysis_results' in attention_neurons and 'attention_weighted' in attention_neurons['analysis_results']:
            for layer_idx in range(analysis_layers):
                if layer_idx in attention_neurons['analysis_results']['attention_weighted']:
                    layer_data = attention_neurons['analysis_results']['attention_weighted'][layer_idx]
                    if 'num_neurons' in layer_data:
                        print(f"  Layer {layer_idx}: {layer_data['num_neurons']} neurons analyzed")
                        if 'mean_activation' in layer_data:
                            print(f"    Mean activation: {layer_data['mean_activation']:.4f}")
                        if 'neuron_diversity' in layer_data:
                            print(f"    Neuron diversity: {layer_data['neuron_diversity']:.4f}")
        
    except Exception as e:
        print(f"❌ Attention neurons analysis failed: {e}")
        results['attention_neurons'] = None
    
    return results

def plot_comparison(all_results, save_dir="plots"):
    """Create individual comparison plots (no subplots)."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare data for plotting
    plot_data = []
    for result in all_results:
        if result is None:
            continue
        for i, layer in enumerate(result['layers']):
            plot_data.append({
                'Model': result['model_name'],
                'Layer': layer,
                'Dead_Percentage': result['dead_percentages'][i],
                'Dead_Count': result['dead_counts'][i],
                'Relative_Layer': layer / max(result['layers']) if result['layers'] else 0
            })
    
    df = pd.DataFrame(plot_data)
    
    # ===== 1. Layer-wise comparison =====
    plt.figure(figsize=(8, 6))
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        plt.plot(model_data['Relative_Layer'], model_data['Dead_Percentage'], 
                 marker='o', label=model, linewidth=2, markersize=4)
    plt.xlabel('Layer (relative depth)')
    plt.ylabel('Dead Neurons (%)')
    plt.title('Dead Neurons by Layer Across Models')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(save_dir, "dead_neurons_by_layer.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {filepath}")
    
    # ===== 2. Overall comparison =====
    overall_data = []
    for result in all_results:
        if result is None:
            continue
        overall_data.append({
            'Model': result['model_name'],
            'Overall_Dead_Percentage': result['overall_percentage'],
            'Total_Dead': result['total_dead'],
            'Total_Neurons': result['total_neurons']
        })
    overall_df = pd.DataFrame(overall_data)

    plt.figure(figsize=(8, 6))
    bars = plt.bar(overall_df['Model'], overall_df['Overall_Dead_Percentage'], 
                   alpha=0.7, edgecolor='black')
    for bar, pct in zip(bars, overall_df['Overall_Dead_Percentage']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    plt.ylabel('Overall Dead Neurons (%)')
    plt.title('Overall Dead Neuron Percentage by Model')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(save_dir, "overall_dead_percentage.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {filepath}")
    
    # ===== 3. Heatmap =====
    plt.figure(figsize=(10, 6))
    pivot_data = df.pivot(index='Model', columns='Layer', values='Dead_Percentage')
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd')
    plt.title('Dead Neuron Percentage Heatmap')
    plt.xlabel('Layer')
    plt.ylabel('Model')
    plt.tight_layout()
    filepath = os.path.join(save_dir, "dead_neuron_heatmap.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {filepath}")
    
    # ===== 4. Distribution comparison =====
    plt.figure(figsize=(8, 6))
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        plt.hist(model_data['Dead_Percentage'], alpha=0.6, label=model, bins=10)
    plt.xlabel('Dead Neuron Percentage')
    plt.ylabel('Number of Layers')
    plt.title('Distribution of Dead Neuron Percentages')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(save_dir, "dead_neuron_distribution.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {filepath}")
    
    # ===== 5. Attention Neurons Comparison =====
    print("\nCreating attention neurons comparison plots...")
    attention_data = []
    for result in all_results:
        if result is None or 'attention_neurons' not in result or result['attention_neurons'] is None:
            continue
            
        attention_neurons = result['attention_neurons']
        if 'analysis_results' in attention_neurons and 'attention_weighted' in attention_neurons['analysis_results']:
            for layer_idx in attention_neurons['analysis_results']['attention_weighted']:
                layer_data = attention_neurons['analysis_results']['attention_weighted'][layer_idx]
                if 'neuron_activations' in layer_data:
                    activations = layer_data['neuron_activations']
                    if hasattr(activations, 'cpu'):
                        activations = activations.cpu().numpy()
                    
                    attention_data.append({
                        'Model': result['model_name'],
                        'Layer': layer_idx,
                        'Mean_Activation': np.mean(activations),
                        'Std_Activation': np.std(activations),
                        'Neuron_Diversity': layer_data.get('neuron_diversity', 0)
                    })
    
    if attention_data:
        attention_df = pd.DataFrame(attention_data)
        
        # Plot attention neuron activations
        plt.figure(figsize=(10, 6))
        for model in attention_df['Model'].unique():
            model_data = attention_df[attention_df['Model'] == model]
            plt.plot(model_data['Layer'], model_data['Mean_Activation'], 
                     marker='o', label=model, linewidth=2, markersize=6)
        plt.xlabel('Layer')
        plt.ylabel('Mean Neuron Activation')
        plt.title('Attention Neuron Activations by Layer Across Models')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        filepath = os.path.join(save_dir, "attention_neurons_activations.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Saved: {filepath}")
        
        # Plot attention neuron diversity
        plt.figure(figsize=(10, 6))
        for model in attention_df['Model'].unique():
            model_data = attention_df[attention_df['Model'] == model]
            plt.plot(model_data['Layer'], model_data['Neuron_Diversity'], 
                     marker='s', label=model, linewidth=2, markersize=6)
        plt.xlabel('Layer')
        plt.ylabel('Neuron Diversity Score')
        plt.title('Attention Neuron Diversity by Layer Across Models')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        filepath = os.path.join(save_dir, "attention_neurons_diversity.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Saved: {filepath}")
        
        # Print attention neurons summary
        print(f"\n{'='*80}")
        print(f"ATTENTION NEURONS SUMMARY")
        print(f"{'='*80}")
        print(f"{'Model':<25} {'Layer':<6} {'Mean Act':<10} {'Std Act':<10} {'Diversity':<10}")
        print(f"{'-'*80}")
        for _, row in attention_df.iterrows():
            print(f"{row['Model']:<25} {row['Layer']:<6} {row['Mean_Activation']:<10.4f} "
                  f"{row['Std_Activation']:<10.4f} {row['Neuron_Diversity']:<10.4f}")
    else:
        print("⚠️  No attention neurons data available for comparison")
    
    # Print summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY COMPARISON")
    print(f"{'='*80}")
    print(f"{'Model':<25} {'Layers':<8} {'Hidden':<8} {'Dead %':<8} {'Total Dead':<12}")
    print(f"{'-'*80}")
    
    for result in all_results:
        if result is None:
            continue
        print(f"{result['model_name']:<25} {result['num_layers']:<8} {result['hidden_size']:<8} "
              f"{result['overall_percentage']:<8.1f} {result['total_dead']:<12}")
    
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_limit', type=int, default=None, help='Limit number of texts to process (default: use all texts)')
    parser.add_argument('--models', nargs='+', default=[
        "state-spaces/mamba-130m-hf",  # Mamba model
        "gpt2",                        # GPT-2 small
        "gpt2-medium",                 # GPT-2 medium
        "microsoft/DialoGPT-medium",   # DialoGPT
        "distilbert-base-uncased",     # DistilBERT
        "bert-base-uncased"            # BERT base
    ], help='Models to compare')
    args = parser.parse_args()
    
    # Define models to compare
    models = args.models
    
    # Load texts with limit
    try:
        from datasets import load_dataset
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")
        texts = [item["text"] for item in dataset if item["text"].strip()]
        texts = [text for text in texts if len(text.split()) > 10 and not text.startswith("=")]
        # Apply text limit if specified (None means use all texts)
        if args.text_limit is not None and args.text_limit > 0:
            texts = texts[:args.text_limit]
        print(f"Using {len(texts)} texts for analysis")
    except:
        # Fallback texts
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming industries worldwide.",
            "Machine learning models require large amounts of training data.",
            "Natural language processing has advanced significantly in recent years.",
            "Deep learning architectures continue to evolve rapidly."
        ]
        if args.text_limit is not None and args.text_limit > 0:
            texts = texts[:args.text_limit]
        print(f"Using {len(texts)} fallback texts for analysis")
    
    print("🚀 Starting Model Comparison Analysis")
    print(f"Testing {len(models)} models with threshold = 0.7")
    
    all_results = []
    
    for model_name in models:
        try:
            result = analyze_model_dead_neurons(model_name, texts, threshold=0.7, max_layers=12)
            all_results.append(result)
        except Exception as e:
            print(f"Failed to analyze {model_name}: {e}")
            all_results.append(None)
    
    # Create comparison plots
    df = plot_comparison(all_results)
    
    print(f"\n🎉 Analysis complete! Check the plots directory for visualizations.")

if __name__ == "__main__":
    main() 
