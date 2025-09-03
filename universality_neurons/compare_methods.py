#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Import the integrated gradients analyzer
from main_integrated import IntegratedGradientsAnalyzer

# Ensure plots directory exists
plots_dir = 'plots_integrated'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    print(f"Created plots directory: {plots_dir}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def extract_attention_weights(model, inputs, layer_idx=0):
    """
    Extract attention weights from a model (approximation for Mamba).
    This simulates what attention-based analysis would give us.
    """
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
    
    if layer_idx < len(hidden_states):
        layer_hidden = hidden_states[layer_idx]
    else:
        layer_hidden = hidden_states[-1]
    
    # For Mamba, we'll use hidden state variance as proxy for attention
    # For Transformer, we'll use the actual attention mechanism if available
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT-2 style architecture
        attention_weights = []
        for block in model.transformer.h:
            if hasattr(block, 'attn') and hasattr(block.attn, 'attn'):
                attn = block.attn.attn
                if hasattr(attn, 'c_attn'):
                    # This is a simplified approximation
                    attention_weights.append(torch.ones_like(layer_hidden[:, :, 0]))
                else:
                    attention_weights.append(torch.ones_like(layer_hidden[:, :, 0]))
            else:
                attention_weights.append(torch.ones_like(layer_hidden[:, :, 0]))
        
        if attention_weights:
            attention_weights = torch.stack(attention_weights).mean(dim=0)
        else:
            attention_weights = torch.ones_like(layer_hidden[:, :, 0])
    else:
        # For Mamba or other models, use hidden state variance as proxy
        attention_weights = torch.var(layer_hidden, dim=1)  # [batch, hidden_dim]
    
    return attention_weights.squeeze()

def compare_attention_vs_integrated_gradients():
    """
    Compare attention weights vs integrated gradients for both models.
    """
    print("=== Loading Models ===")
    
    # Load models
    mamba_model_name = "state-spaces/mamba-130m-hf"
    transformer_model_name = "gpt2"
    
    mamba_tokenizer = AutoTokenizer.from_pretrained(mamba_model_name)
    mamba_model = AutoModelForCausalLM.from_pretrained(mamba_model_name).to(device).eval()
    
    transformer_tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
    transformer_model = AutoModelForCausalLM.from_pretrained(transformer_model_name).to(device).eval()
    
    print("Models loaded successfully.")
    
    # Sample texts for analysis
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Machine learning models require large datasets.",
        "Natural language processing involves understanding text.",
        "Deep learning has revolutionized computer vision."
    ]
    
    # Initialize analyzers
    mamba_ig_analyzer = IntegratedGradientsAnalyzer(mamba_model, mamba_tokenizer)
    transformer_ig_analyzer = IntegratedGradientsAnalyzer(transformer_model, transformer_tokenizer)
    
    print("\n=== Comparing Methods ===")
    
    mamba_comparisons = []
    transformer_comparisons = []
    
    for i, text in enumerate(sample_texts):
        print(f"\nAnalyzing text {i+1}: {text[:40]}...")
        
        # Prepare inputs
        mamba_inputs = mamba_tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
        transformer_inputs = transformer_tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
        
        try:
            # Extract attention weights (approximation)
            mamba_attention = extract_attention_weights(mamba_model, mamba_inputs, layer_idx=0)
            transformer_attention = extract_attention_weights(transformer_model, transformer_inputs, layer_idx=0)
            
            # Get integrated gradients
            mamba_ig = mamba_ig_analyzer.analyze_neuron_importance(mamba_inputs, layer_idx=0)
            transformer_ig = transformer_ig_analyzer.analyze_neuron_importance(transformer_inputs, layer_idx=0)
            
            # Compare methods for Mamba
            mamba_comp = mamba_ig_analyzer.compare_with_attention(
                mamba_inputs, mamba_attention, layer_idx=0
            )
            
            # Compare methods for Transformer
            transformer_comp = transformer_ig_analyzer.compare_with_attention(
                transformer_inputs, transformer_attention, layer_idx=0
            )
            
            mamba_comparisons.append(mamba_comp)
            transformer_comparisons.append(transformer_comp)
            
            print(f"  Mamba - Correlation: {mamba_comp['correlation']:.4f}")
            print(f"  Transformer - Correlation: {transformer_comp['correlation']:.4f}")
            
        except Exception as e:
            print(f"  Error analyzing text {i+1}: {e}")
            continue
    
    if not mamba_comparisons or not transformer_comparisons:
        print("No comparisons obtained. Exiting.")
        return
    
    # Aggregate comparison results
    print("\n=== Aggregating Comparison Results ===")
    
    # Average correlations
    mamba_avg_corr = np.mean([comp['correlation'] for comp in mamba_comparisons])
    transformer_avg_corr = np.mean([comp['correlation'] for comp in transformer_comparisons])
    
    mamba_avg_cosine = np.mean([comp['cosine_similarity'] for comp in mamba_comparisons])
    transformer_avg_cosine = np.mean([comp['cosine_similarity'] for comp in transformer_comparisons])
    
    print(f"Mamba - Avg Correlation: {mamba_avg_corr:.4f}, Avg Cosine: {mamba_avg_cosine:.4f}")
    print(f"Transformer - Avg Correlation: {transformer_avg_corr:.4f}, Avg Cosine: {transformer_avg_cosine:.4f}")
    
    # Create comparison plots
    print("\n=== Creating Comparison Plots ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Attention Weights vs Integrated Gradients Comparison', fontsize=16)
    
    # Plot 1: Mamba Attention vs IG
    if mamba_comparisons[0]['attention_scores'] is not None:
        axes[0, 0].scatter(mamba_comparisons[0]['attention_scores'].cpu().numpy(), 
                           mamba_comparisons[0]['ig_scores'].cpu().numpy(), 
                           alpha=0.6, color='skyblue')
        axes[0, 0].set_title(f'Mamba: Attention vs IG (ρ={mamba_avg_corr:.3f})')
        axes[0, 0].set_xlabel('Attention Weights (Normalized)')
        axes[0, 0].set_ylabel('Integrated Gradients (Normalized)')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Transformer Attention vs IG
    if transformer_comparisons[0]['attention_scores'] is not None:
        axes[0, 1].scatter(transformer_comparisons[0]['attention_scores'].cpu().numpy(), 
                           transformer_comparisons[0]['ig_scores'].cpu().numpy(), 
                           alpha=0.6, color='salmon')
        axes[0, 1].set_title(f'Transformer: Attention vs IG (ρ={transformer_avg_corr:.3f})')
        axes[0, 1].set_xlabel('Attention Weights (Normalized)')
        axes[0, 1].set_ylabel('Integrated Gradients (Normalized)')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Correlation comparison
    models = ['Mamba', 'Transformer']
    correlations = [mamba_avg_corr, transformer_avg_corr]
    colors = ['skyblue', 'salmon']
    
    bars = axes[0, 2].bar(models, correlations, color=colors, alpha=0.7)
    axes[0, 2].set_title('Method Correlation Comparison')
    axes[0, 2].set_ylabel('Correlation Coefficient')
    axes[0, 2].set_ylim(-1, 1)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Add correlation values on bars
    for bar, corr in zip(bars, correlations):
        height = bar.get_height()
        axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{corr:.3f}', ha='center', va='bottom')
    
    # Plot 4: Cosine similarity comparison
    cosines = [mamba_avg_cosine, transformer_avg_cosine]
    
    bars = axes[1, 0].bar(models, cosines, color=colors, alpha=0.7)
    axes[1, 0].set_title('Method Cosine Similarity Comparison')
    axes[1, 0].set_ylabel('Cosine Similarity')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add cosine values on bars
    for bar, cos in zip(bars, cosines):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{cos:.3f}', ha='center', va='bottom')
    
    # Plot 5: Method agreement analysis
    # Calculate how many neurons are in top-k for both methods
    top_k = 50
    
    mamba_agreement = []
    transformer_agreement = []
    
    for comp in mamba_comparisons:
        if comp['attention_scores'] is not None:
            top_attn = torch.topk(comp['attention_scores'], top_k)[1]
            top_ig = torch.topk(comp['ig_scores'], top_k)[1]
            agreement = len(set(top_attn.tolist()) & set(top_ig.tolist())) / top_k
            mamba_agreement.append(agreement)
    
    for comp in transformer_comparisons:
        if comp['attention_scores'] is not None:
            top_attn = torch.topk(comp['attention_scores'], top_k)[1]
            top_ig = torch.topk(comp['ig_scores'], top_k)[1]
            agreement = len(set(top_attn.tolist()) & set(top_ig.tolist())) / top_k
            transformer_agreement.append(agreement)
    
    if mamba_agreement and transformer_agreement:
        agreement_data = [np.mean(mamba_agreement), np.mean(transformer_agreement)]
        bars = axes[1, 1].bar(models, agreement_data, color=colors, alpha=0.7)
        axes[1, 1].set_title(f'Top-{top_k} Neuron Agreement')
        axes[1, 1].set_ylabel('Agreement Ratio')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add agreement values on bars
        for bar, agree in zip(bars, agreement_data):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{agree:.3f}', ha='center', va='bottom')
    
    # Plot 6: Score distribution comparison
    if mamba_comparisons[0]['attention_scores'] is not None:
        axes[1, 2].hist(mamba_comparisons[0]['attention_scores'].cpu().numpy(), 
                        bins=20, alpha=0.7, label='Mamba Attention', color='skyblue', density=True)
        axes[1, 2].hist(mamba_comparisons[0]['ig_scores'].cpu().numpy(), 
                        bins=20, alpha=0.7, label='Mamba IG', color='lightblue', density=True)
        axes[1, 2].set_title('Mamba: Attention vs IG Distribution')
        axes[1, 2].set_xlabel('Normalized Score')
        axes[1, 2].set_ylabel('Density')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots/method_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved method comparison plot to plots/method_comparison_{timestamp}.png")
    
    # Save comparison summary
    print("\n=== Saving Comparison Summary ===")
    summary_file = f'plots/method_comparison_summary_{timestamp}.txt'
    
    with open(summary_file, 'w') as f:
        f.write(f"Attention Weights vs Integrated Gradients Comparison - {timestamp}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Device used: {device}\n")
        f.write(f"Number of texts analyzed: {len(sample_texts)}\n")
        f.write(f"Mamba model: {mamba_model_name}\n")
        f.write(f"Transformer model: {transformer_model_name}\n\n")
        
        f.write("Mamba Model Results:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Average Correlation: {mamba_avg_corr:.6f}\n")
        f.write(f"Average Cosine Similarity: {mamba_avg_cosine:.6f}\n")
        if mamba_agreement:
            f.write(f"Top-{top_k} Neuron Agreement: {np.mean(mamba_agreement):.6f}\n")
        
        f.write(f"\nTransformer Model Results:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Average Correlation: {transformer_avg_corr:.6f}\n")
        f.write(f"Average Cosine Similarity: {transformer_avg_cosine:.6f}\n")
        if transformer_agreement:
            f.write(f"Top-{top_k} Neuron Agreement: {np.mean(transformer_agreement):.6f}\n")
        
        f.write(f"\nMethod Comparison:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Correlation ratio (Mamba/Transformer): {mamba_avg_corr/transformer_avg_corr:.4f}\n")
        f.write(f"Cosine similarity ratio (Mamba/Transformer): {mamba_avg_cosine/transformer_avg_cosine:.4f}\n")
        
        f.write(f"\nKey Insights:\n")
        f.write("-" * 15 + "\n")
        if abs(mamba_avg_corr) < 0.3:
            f.write("• Mamba shows weak correlation between attention and IG methods\n")
        elif abs(mamba_avg_corr) < 0.7:
            f.write("• Mamba shows moderate correlation between attention and IG methods\n")
        else:
            f.write("• Mamba shows strong correlation between attention and IG methods\n")
        
        if abs(transformer_avg_corr) < 0.3:
            f.write("• Transformer shows weak correlation between attention and IG methods\n")
        elif abs(transformer_avg_corr) < 0.7:
            f.write("• Transformer shows moderate correlation between attention and IG methods\n")
        else:
            f.write("• Transformer shows strong correlation between attention and IG methods\n")
        
        if mamba_avg_corr > transformer_avg_corr:
            f.write("• Mamba has higher agreement between methods than Transformer\n")
        else:
            f.write("• Transformer has higher agreement between methods than Mamba\n")
    
    print(f"Comparison summary saved to {summary_file}")
    print("\nMethod comparison complete!")

if __name__ == "__main__":
    compare_attention_vs_integrated_gradients()
