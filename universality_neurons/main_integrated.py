#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from datetime import datetime
import os

# Ensure plots directory exists
plots_dir = 'plots_integrated'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    print(f"Created plots directory: {plots_dir}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create timestamp for plot filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Analysis timestamp: {timestamp}")

class IntegratedGradientsAnalyzer:
    """
    Analyzes models using Integrated Gradients instead of attention weights.
    This implementation is based on the HiddenMambaAttn approach which
    views Mamba models as attention-driven models.
    """
    
    def __init__(self, model, tokenizer, baseline_input=None):
        self.model = model
        self.tokenizer = tokenizer
        # Ensure model is on the correct device
        if not hasattr(self.model, 'device'):
            self.model.device = next(self.model.parameters()).device
        self.baseline_input = baseline_input or self._create_baseline()
        self.model.eval()
    
    def _create_baseline(self):
        """Create baseline input (usually zeros or special tokens)"""
        # For language models, use padding token as baseline
        baseline_text = self.tokenizer.pad_token or self.tokenizer.eos_token
        baseline_tokens = self.tokenizer(baseline_text, return_tensors="pt")
        # Ensure baseline is on the same device as the model
        baseline_tokens = {k: v.to(self.model.device) for k, v in baseline_tokens.items()}
        return baseline_tokens
    
    def compute_integrated_gradients(self, inputs, target_class=None, steps=50):
        """
        Compute integrated gradients for given inputs.
        
        Args:
            inputs: Input tokens
            target_class: Target class for classification (None for language models)
            steps: Number of interpolation steps
        
        Returns:
            Integrated gradients tensor
        """
        # For language models, we'll use a different approach
        # Instead of interpolating input_ids, we'll use the model's embeddings
        
        # Get the model's embedding layer
        if hasattr(self.model, 'get_input_embeddings'):
            embedding_layer = self.model.get_input_embeddings()
        else:
            # Fallback: use hidden states variance as proxy
            print("Warning: Could not access embedding layer, using hidden states variance as proxy")
            return None
        
        # Store gradients
        gradients = []
        
        # Get baseline and input embeddings
        with torch.no_grad():
            baseline_embeddings = embedding_layer(self.baseline_input['input_ids'])
            input_embeddings = embedding_layer(inputs['input_ids'])
        
        # Interpolate between baseline and input embeddings
        alphas = torch.linspace(0, 1, steps).to(self.model.device)
        
        for alpha in alphas:
            # Interpolate embeddings
            interpolated_embeddings = baseline_embeddings * (1 - alpha) + input_embeddings * alpha
            interpolated_embeddings = interpolated_embeddings.to(self.model.device)
            
            # Compute gradients
            interpolated_embeddings.requires_grad_(True)
            
            with torch.enable_grad():
                # Create a custom forward pass using embeddings
                if hasattr(self.model, 'transformer'):
                    # For GPT-2 style models
                    outputs = self.model.transformer(inputs_embeds=interpolated_embeddings)
                    logits = self.model.lm_head(outputs.last_hidden_state)
                else:
                    # For other models, try to use inputs_embeds
                    try:
                        outputs = self.model(inputs_embeds=interpolated_embeddings)
                        logits = outputs.logits
                    except:
                        # Fallback: use hidden states
                        print("Warning: Could not compute logits, using hidden states")
                        break
                
                # Use the last token's logits
                last_logits = logits[0, -1, :]
                # Target the most likely next token
                target_token = torch.argmax(last_logits)
                target_logit = last_logits[target_token]
                target_logit.backward()
                
                # Get gradients
                grad = interpolated_embeddings.grad.clone()
                gradients.append(grad)
        
        if not gradients:
            print("Warning: No gradients computed, using hidden states variance as proxy")
            return None
        
        # Average gradients and multiply by embedding difference
        avg_gradients = torch.stack(gradients).mean(dim=0)
        embedding_diff = input_embeddings - baseline_embeddings
        
        # Ensure all tensors are on the same device
        avg_gradients = avg_gradients.to(self.model.device)
        embedding_diff = embedding_diff.to(self.model.device)
        
        integrated_gradients = avg_gradients * embedding_diff
        
        return integrated_gradients
    
    def analyze_neuron_importance(self, inputs, layer_idx=0):
        """
        Analyze neuron importance using integrated gradients.
        
        Args:
            inputs: Input tokens
            layer_idx: Layer index to analyze
        
        Returns:
            Dictionary containing neuron importance scores
        """
        # Ensure inputs are on the correct device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Get model outputs and hidden states
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
        
        # Get hidden states for the specified layer
        if layer_idx < len(hidden_states):
            layer_hidden = hidden_states[layer_idx]  # [batch, seq_len, hidden_dim]
        else:
            print(f"Layer {layer_idx} not available. Using last layer.")
            layer_hidden = hidden_states[-1]
        
        # Compute integrated gradients
        integrated_grads = self.compute_integrated_gradients(inputs)
        
        if integrated_grads is not None:
            # Map gradients to hidden states (approximate)
            # For language models, we'll use the gradient magnitude as a proxy
            grad_magnitude = torch.norm(integrated_grads, dim=-1)  # [batch, seq_len]
            
            # Compute neuron importance based on gradient magnitude and hidden state variance
            hidden_variance = torch.var(layer_hidden, dim=1)  # [batch, hidden_dim]
            
            # Ensure all tensors are on the same device
            grad_magnitude = grad_magnitude.to(self.model.device)
            hidden_variance = hidden_variance.to(self.model.device)
            
            # Combine gradient information with hidden state variance
            neuron_importance = hidden_variance * grad_magnitude.mean(dim=1, keepdim=True)
        else:
            # Fallback: use only hidden state variance
            print("Using hidden state variance as fallback for neuron importance")
            hidden_variance = torch.var(layer_hidden, dim=1)  # [batch, hidden_dim]
            hidden_variance = hidden_variance.to(self.model.device)
            neuron_importance = hidden_variance
            grad_magnitude = torch.ones_like(hidden_variance)
        
        return {
            'neuron_importance': neuron_importance.squeeze(),
            'hidden_variance': hidden_variance.squeeze(),
            'gradient_magnitude': grad_magnitude.squeeze(),
            'integrated_gradients': integrated_grads.squeeze() if integrated_grads is not None else None
        }
    
    def compare_with_attention(self, inputs, attention_weights, layer_idx=0):
        """
        Compare integrated gradients with attention weights.
        
        Args:
            inputs: Input tokens
            attention_weights: Attention weights from attention-based analysis
            layer_idx: Layer index
        
        Returns:
            Dictionary containing comparison metrics
        """
        ig_results = self.analyze_neuron_importance(inputs, layer_idx)
        
        # Normalize both to [0, 1] for comparison
        ig_normalized = (ig_results['neuron_importance'] - ig_results['neuron_importance'].min()) / \
                       (ig_results['neuron_importance'].max() - ig_results['neuron_importance'].min())
        
        if attention_weights is not None:
            attention_weights = attention_weights.to(ig_normalized.device)
            attn_normalized = (attention_weights - attention_weights.min()) / \
                            (attention_weights.max() - attention_weights.min())
            
            # Compute correlation
            correlation = torch.corrcoef(torch.stack([ig_normalized, attn_normalized]))[0, 1].item()
            
            # Compute similarity metrics
            cosine_sim = F.cosine_similarity(ig_normalized.unsqueeze(0), attn_normalized.unsqueeze(0)).item()
            
            return {
                'correlation': correlation,
                'cosine_similarity': cosine_sim,
                'ig_scores': ig_normalized,
                'attention_scores': attn_normalized,
                'comparison_metrics': {
                    'correlation': correlation,
                    'cosine_similarity': cosine_sim
                }
            }
        else:
            return {
                'ig_scores': ig_normalized,
                'attention_scores': None,
                'comparison_metrics': None
            }





def run_integrated_gradients_analysis():
    """Main function to run integrated gradients analysis."""
    
    # Load models
    print("\nLoading models...")
    mamba_model_name = "state-spaces/mamba-130m-hf"
    transformer_model_name = "gpt2"
    
    mamba_tokenizer = AutoTokenizer.from_pretrained(mamba_model_name)
    mamba_model = AutoModelForCausalLM.from_pretrained(mamba_model_name).to(device).eval()
    
    transformer_tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
    transformer_model = AutoModelForCausalLM.from_pretrained(transformer_model_name).to(device).eval()
    
    print("Models loaded successfully.")
    
    # Load dataset
    print("\nLoading dataset...")
    try:
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")
        texts = [item["text"] for item in dataset if item["text"].strip() != ""]
        texts = texts[:10]  # Use first 10 texts for analysis
        print(f"Loaded {len(texts)} texts for analysis.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world.",
            "Machine learning models require large datasets.",
            "Natural language processing involves understanding text.",
            "Deep learning has revolutionized computer vision.",
            
        ]
        print(f"Using {len(texts)} sample texts instead.")
    
    # Initialize analyzers
    mamba_ig_analyzer = IntegratedGradientsAnalyzer(mamba_model, mamba_tokenizer)
    transformer_ig_analyzer = IntegratedGradientsAnalyzer(transformer_model, transformer_tokenizer)
    
    print("\n=== Integrated Gradients Analysis ===")
    
    mamba_results = []
    transformer_results = []
    
    # Analyze each text sequentially
    for i, text in enumerate(texts):
        print(f"\nAnalyzing text {i+1}/{len(texts)}: {text[:50]}...")
        
        # Prepare inputs
        mamba_inputs = mamba_tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        transformer_inputs = transformer_tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        
        # Analyze with integrated gradients
        try:
            mamba_ig = mamba_ig_analyzer.analyze_neuron_importance(mamba_inputs, layer_idx=0)
            transformer_ig = transformer_ig_analyzer.analyze_neuron_importance(transformer_inputs, layer_idx=0)
            
            mamba_results.append(mamba_ig)
            transformer_results.append(transformer_ig)
            
        except Exception as e:
            print(f"Error analyzing text {i+1}: {e}")
            continue
    
    if not mamba_results or not transformer_results:
        print("No results obtained. Exiting.")
        return
    
    # Aggregate results
    print("\n=== Aggregating Results ===")
    
    # Average neuron importance across all texts
    mamba_avg_importance = torch.stack([r['neuron_importance'] for r in mamba_results]).mean(dim=0)
    transformer_avg_importance = torch.stack([r['neuron_importance'] for r in transformer_results]).mean(dim=0)
    
    # Compute statistics
    mamba_stats = {
        'mean': mamba_avg_importance.mean().item(),
        'std': mamba_avg_importance.std().item(),
        'min': mamba_avg_importance.min().item(),
        'max': mamba_avg_importance.max().item()
    }
    
    transformer_stats = {
        'mean': transformer_avg_importance.mean().item(),
        'std': transformer_avg_importance.std().item(),
        'min': transformer_avg_importance.min().item(),
        'max': transformer_avg_importance.max().item()
    }
    
    print(f"Mamba IG Statistics: {mamba_stats}")
    print(f"Transformer IG Statistics: {transformer_stats}")
    
    # Create comparison plots
    print("\n=== Creating Comparison Plots ===")
    
    # Plot 1: Integrated Gradients Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Integrated Gradients Analysis - Mamba vs Transformer', fontsize=16)
    
    # Mamba IG scores
    axes[0, 0].bar(range(len(mamba_avg_importance)), mamba_avg_importance.cpu().numpy(), 
                    color='skyblue', alpha=0.7)
    axes[0, 0].set_title(f'Mamba IG Scores (μ={mamba_stats["mean"]:.4f}, σ={mamba_stats["std"]:.4f})')
    axes[0, 0].set_xlabel('Neuron Index')
    axes[0, 0].set_ylabel('IG Importance Score')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Transformer IG scores
    axes[0, 1].bar(range(len(transformer_avg_importance)), transformer_avg_importance.cpu().numpy(), 
                    color='salmon', alpha=0.7)
    axes[0, 1].set_title(f'Transformer IG Scores (μ={transformer_stats["mean"]:.4f}, σ={transformer_stats["std"]:.4f})')
    axes[0, 1].set_xlabel('Neuron Index')
    axes[0, 1].set_ylabel('IG Importance Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Top neurons comparison
    top_k = 20
    top_mamba_indices = torch.topk(mamba_avg_importance, top_k)[1]
    top_transformer_indices = torch.topk(transformer_avg_importance, top_k)[1]
    
    x_pos = np.arange(top_k)
    axes[1, 0].bar(x_pos - 0.2, mamba_avg_importance[top_mamba_indices].cpu().numpy(), 
                    width=0.4, label='Mamba', color='skyblue', alpha=0.8)
    axes[1, 0].bar(x_pos + 0.2, transformer_avg_importance[top_transformer_indices].cpu().numpy(), 
                    width=0.4, label='Transformer', color='salmon', alpha=0.8)
    axes[1, 0].set_title(f'Top {top_k} Neurons Comparison')
    axes[1, 0].set_xlabel('Neuron Rank')
    axes[1, 0].set_ylabel('IG Importance Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Score distribution comparison
    axes[1, 1].hist(mamba_avg_importance.cpu().numpy(), bins=30, alpha=0.7, 
                     label='Mamba', color='skyblue', density=True)
    axes[1, 1].hist(transformer_avg_importance.cpu().numpy(), bins=30, alpha=0.7, 
                     label='Transformer', color='salmon', density=True)
    axes[1, 1].set_title('IG Score Distribution Comparison')
    axes[1, 1].set_xlabel('IG Importance Score')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots_integrated/integrated_gradients_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved IG comparison plot to plots_integrated/integrated_gradients_comparison_{timestamp}.png")
    
    # Save analysis summary
    print("\n=== Saving Analysis Summary ===")
    summary_file = f'plots_integrated/ig_analysis_summary_{timestamp}.txt'
    
    with open(summary_file, 'w') as f:
        f.write(f"Integrated Gradients Analysis Summary - {timestamp}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Device used: {device}\n")
        f.write(f"Number of texts analyzed: {len(texts)}\n")

        f.write(f"Mamba model: {mamba_model_name}\n")
        f.write(f"Transformer model: {transformer_model_name}\n\n")
        
        f.write("Mamba IG Statistics:\n")
        f.write("-" * 25 + "\n")
        for key, value in mamba_stats.items():
            f.write(f"{key.capitalize()}: {value:.6f}\n")
        
        f.write("\nTransformer IG Statistics:\n")
        f.write("-" * 30 + "\n")
        for key, value in transformer_stats.items():
            f.write(f"{key.capitalize()}: {value:.6f}\n")
        
        f.write(f"\nModel Comparison:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Mean ratio (Mamba/Transformer): {mamba_stats['mean']/transformer_stats['mean']:.4f}\n")
        f.write(f"Std ratio (Mamba/Transformer): {mamba_stats['std']/transformer_stats['std']:.4f}\n")
        f.write(f"Range ratio (Mamba/Transformer): {(mamba_stats['max']-mamba_stats['min'])/(transformer_stats['max']-transformer_stats['min']):.4f}\n")
        
        f.write(f"\nTop 10 Mamba IG Neurons:\n")
        f.write("-" * 30 + "\n")
        top_mamba_neurons = torch.topk(mamba_avg_importance, 10)
        for i, (score, idx) in enumerate(zip(top_mamba_neurons[0], top_mamba_neurons[1])):
            f.write(f"{i+1}. Neuron {idx.item()}: {score.item():.6f}\n")
        
        f.write(f"\nTop 10 Transformer IG Neurons:\n")
        f.write("-" * 35 + "\n")
        top_transformer_neurons = torch.topk(transformer_avg_importance, 10)
        for i, (score, idx) in enumerate(zip(top_transformer_neurons[0], top_transformer_neurons[1])):
            f.write(f"{i+1}. Neuron {idx.item()}: {score.item():.6f}\n")
    
    print(f"Analysis summary saved to {summary_file}")
    print("\nIntegrated Gradients analysis complete!")

if __name__ == "__main__":
    run_integrated_gradients_analysis()
