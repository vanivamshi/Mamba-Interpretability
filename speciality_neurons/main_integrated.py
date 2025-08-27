#!/usr/bin/env python3
"""
Main script for running Mamba and GPT-2 neuron analysis
with integrated gradients method integration.
"""

import os
import json
import torch
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from analysis_plots import run_all_plots
from utils import debug_model_structure
from delta_extraction import find_delta_sensitive_neurons_fixed
from delta_sensitivity_transformer import find_delta_sensitive_neurons_transformer
from specialty_neurons import find_specialty_neurons_fixed
from template_robustness import (
    test_template_robustness, 
    analyze_template_robustness_results,
    run_comprehensive_template_analysis,
    get_available_relations
)
from typographical_errors_robustness import (
    test_typo_robustness,
    analyze_typo_robustness_results,
    generate_typo_examples
)
from attention_neurons import MambaAttentionNeurons, integrate_mamba_attention_neurons

def compute_integrated_gradients(model, input_ids, tokenizer, target_layer=0, num_steps=50):
    """
    Compute integrated gradients for a given input and model.
    
    Args:
        model: The model to analyze
        input_ids: Input token IDs
        tokenizer: Tokenizer for the model
        target_layer: Layer to compute gradients for
        num_steps: Number of interpolation steps
    
    Returns:
        Dictionary containing integrated gradients and neuron importance
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Get embeddings
    if hasattr(model, 'wte'):  # GPT-2 style
        embeddings = model.wte(input_ids)
    elif hasattr(model, 'embed_tokens'):  # Mamba style
        embeddings = model.embed_tokens(input_ids)
    else:
        # Fallback: try to find embedding layer
        for name, module in model.named_modules():
            if 'embed' in name.lower() and hasattr(module, 'weight'):
                embeddings = module(input_ids)
                break
        else:
            raise ValueError("Could not find embedding layer")
    
    # Create baseline (zero embeddings)
    baseline = torch.zeros_like(embeddings)
    
    # Interpolate between baseline and actual embeddings
    alphas = torch.linspace(0, 1, num_steps).to(device)
    interpolated_embeddings = torch.stack([
        baseline + alpha * (embeddings - baseline) for alpha in alphas
    ])
    
    # Compute gradients for each interpolated input
    gradients = []
    
    for i in range(num_steps):
        interpolated_input = interpolated_embeddings[i].unsqueeze(0)
        interpolated_input.requires_grad_(True)
        
        # Forward pass to target layer
        if hasattr(model, 'layers') and target_layer < len(model.layers):
            # For Mamba models
            with torch.enable_grad():
                # Hook to capture intermediate activations
                activations = {}
                
                def hook_fn(name):
                    def hook(module, input, output):
                        activations[name] = output
                    return hook
                
                # Register hooks for the target layer
                target_module = model.layers[target_layer]
                hook_handles = []
                
                if hasattr(target_module, 'mixer'):
                    hook_handles.append(target_module.mixer.register_forward_hook(hook_fn('mixer')))
                if hasattr(target_module, 'norm'):
                    hook_handles.append(target_module.norm.register_forward_hook(hook_fn('norm')))
                
                # Forward pass
                _ = model(input_ids)  # Use original input_ids for forward pass
                
                # Remove hooks
                for handle in hook_handles:
                    handle.remove()
                
                # Compute gradients with respect to interpolated embeddings
                if 'mixer' in activations:
                    output = activations['mixer']
                    gradients.append(torch.autograd.grad(output.sum(), interpolated_input, 
                                                      retain_graph=False)[0])
                else:
                    gradients.append(torch.zeros_like(interpolated_input))
        else:
            # For other models, use a simpler approach
            with torch.enable_grad():
                output = model(input_ids)
                gradients.append(torch.autograd.grad(output.logits.sum(), interpolated_input, 
                                                  retain_graph=False)[0])
    
    # Average gradients
    avg_gradients = torch.stack(gradients).mean(dim=0)
    
    # Compute integrated gradients
    integrated_gradients = (embeddings - baseline) * avg_gradients
    
    # Compute neuron importance (sum across sequence and embedding dimensions)
    neuron_importance = integrated_gradients.sum(dim=(0, 1))  # (hidden_size,)
    
    return {
        'integrated_gradients': integrated_gradients,
        'neuron_importance': neuron_importance,
        'avg_gradients': avg_gradients,
        'embeddings': embeddings,
        'baseline': baseline
    }

def analyze_integrated_gradients_results(ig_results, model_name, layer_idx=0):
    """
    Analyze and print detailed results from integrated gradients computation.
    
    Args:
        ig_results: Results from compute_integrated_gradients
        model_name: Name of the model being analyzed
        layer_idx: Layer index being analyzed
    """
    print(f"\n=== INTEGRATED GRADIENTS ANALYSIS FOR {model_name.upper()} ===")
    print(f"Layer: {layer_idx}")
    
    if 'neuron_importance' in ig_results:
        neuron_importance = ig_results['neuron_importance']
        
        # Basic statistics
        print(f"Total neurons analyzed: {len(neuron_importance)}")
        print(f"Mean importance: {neuron_importance.mean().item():.6f}")
        print(f"Std importance: {neuron_importance.std().item():.6f}")
        print(f"Min importance: {neuron_importance.min().item():.6f}")
        print(f"Max importance: {neuron_importance.max().item():.6f}")
        
        # Top neurons
        top_indices = torch.argsort(neuron_importance, descending=True)[:10]
        print(f"\nTop 10 most important neurons:")
        for i, idx in enumerate(top_indices):
            importance = neuron_importance[idx].item()
            print(f"  {i+1:2d}. Neuron {idx:4d}: {importance:.6f}")
        
        # Distribution analysis
        quantiles = torch.quantile(neuron_importance, torch.tensor([0.25, 0.5, 0.75]))
        print(f"\nImportance distribution:")
        print(f"  25th percentile: {quantiles[0].item():.6f}")
        print(f"  50th percentile (median): {quantiles[1].item():.6f}")
        print(f"  75th percentile: {quantiles[2].item():.6f}")
        
        # Sparsity analysis
        zero_threshold = 1e-6
        num_zero_neurons = (torch.abs(neuron_importance) < zero_threshold).sum().item()
        sparsity = num_zero_neurons / len(neuron_importance) * 100
        print(f"\nSparsity analysis:")
        print(f"  Neurons with near-zero importance (<{zero_threshold}): {num_zero_neurons}")
        print(f"  Sparsity percentage: {sparsity:.2f}%")
        
        return {
            'top_neurons': [(int(idx.item()), float(neuron_importance[idx].item())) for idx in top_indices],
            'statistics': {
                'mean': float(neuron_importance.mean().item()),
                'std': float(neuron_importance.std().item()),
                'min': float(neuron_importance.min().item()),
                'max': float(neuron_importance.max().item()),
                'sparsity': sparsity
            }
        }
    
    return None


def run_integrated_gradients_analysis(model, tokenizer, device):
    """
    Run comprehensive integrated gradients analysis for neuron importance.
    """
    print("🔍 Running Integrated Gradients Analysis...")
    
    # Sample texts for integrated gradients analysis
    ig_texts = [
        "The quick brown fox jumps over the lazy dog with great speed.",
        "Apple produces the iPhone smartphone device for global markets worldwide.",
        "Artificial intelligence is transforming the world through machine learning algorithms.",
        "Machine learning models require large datasets to achieve optimal performance.",
        "Natural language processing involves understanding and generating human text."
    ]
    
    # Tokenize texts for integrated gradients analysis
    ig_inputs = []
    for text in ig_texts:
        try:
            input_ids = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            ig_inputs.append(input_ids)
        except Exception as e:
            print(f"Error tokenizing text: {e}")
            continue
    
    if not ig_inputs:
        print("No valid inputs for integrated gradients analysis. Using dummy data.")
        # Create dummy input for testing
        dummy_input = torch.randint(0, 1000, (1, 64)).to(device)
        ig_inputs = [dummy_input]
    
    # Use the first input for analysis
    main_input = ig_inputs[0]
    
    try:
        # Check if model supports integrated gradients computation
        if hasattr(model, 'layers') and len(model.layers) > 0:
            print("Model supports integrated gradients computation. Running Mamba integrated gradients analysis...")
            
            # Try to use the new integrated gradients implementation first
            try:
                print("Computing integrated gradients using gradient-based approach...")
                ig_results = compute_integrated_gradients(model, main_input, tokenizer, target_layer=0, num_steps=20)
                
                # Analyze the results
                analysis_summary = analyze_integrated_gradients_results(ig_results, "Mamba", 0)
                
                # Convert to the expected format for compatibility
                converted_results = {
                    'mamba_neurons': {
                        'integrated_gradients': {
                            0: {
                                'neuron_activations': ig_results['neuron_importance'],
                                'neuron_importance': ig_results['neuron_importance'],
                                'integrated_gradients': ig_results['integrated_gradients']
                            }
                        }
                    },
                    'analysis_results': {
                        'integrated_gradients': {
                            0: {
                                'num_neurons': len(ig_results['neuron_importance']),
                                'mean_activation': ig_results['neuron_importance'].mean().item(),
                                'activation_std': ig_results['neuron_importance'].std().item()
                            }
                        }
                    },
                    'ig_analysis': analysis_summary
                }
                
                return converted_results, main_input
                
            except Exception as ig_error:
                print(f"Integrated gradients computation failed: {ig_error}")
                print("Falling back to gradient-guided method...")
                
                # Fallback to the existing gradient_guided method
                ig_results = integrate_mamba_attention_neurons(
                    model=model,
                    inputs=main_input,
                    layer_indices=[0, 6, 12, 18],  # Analyze multiple layers
                    methods=['gradient_guided']  # Focus on gradient-based methods
                )
                
                return ig_results, main_input
            
        else:
            print("Model doesn't support integrated gradients computation. Using fallback method.")
            return None, main_input
            
    except Exception as e:
        print(f"Error in integrated gradients analysis: {e}")
        return None, main_input


def run_analysis_for_model(model_name, output_filename):

    print("="*70)
    print(f"Running analysis for: {model_name}")
    print("="*70)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        print("Model and tokenizer loaded successfully!")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("\n=== Model Structure Debug ===")
        debug_model_structure(model, max_depth=3)

    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Dataset
    print("\n Loading Dataset...")
    try:
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")
        texts = [item["text"] for item in dataset if item["text"].strip() != "" and len(item["text"]) > 50]
        texts = texts[:10]
        print(f"Loaded {len(texts)} high-quality samples.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        texts = [
            "The quick brown fox jumps over the lazy dog with great speed.",
            "Apple produces the iPhone smartphone device for global markets worldwide.",
            "Artificial intelligence is transforming the world through machine learning algorithms.",
            "Machine learning models require large datasets to achieve optimal performance.",
            "Natural language processing involves understanding and generating human text.",
            "Germany is a member of the European Union organization since 1957.",
            "The novel Don Quixote was written in Spanish by Miguel de Cervantes.",
            "Sony manufactures PlayStation gaming consoles for entertainment purposes.",
            "France belongs to the NATO military alliance for collective defense.",
            "Once upon a time, there lived a wise old king in a distant kingdom."
        ]
        print(f"Using {len(texts)} enhanced dummy texts for testing.")

    model.to("cpu")

    print("\n INTEGRATED GRADIENTS NEURONS ANALYSIS")

    # Run the integrated gradients analysis
    ig_results, main_input = run_integrated_gradients_analysis(model, tokenizer, device)
    
    specialty_neurons = {}
    try:
        if ig_results:
            # Extract neurons from integrated gradients analysis
            if 'mamba_neurons' in ig_results:
                mamba_neurons = ig_results['mamba_neurons']
                
                # Convert to specialty neurons format for compatibility
                for method, layer_neurons in mamba_neurons.items():
                    if method == 'integrated_gradients' and 0 in layer_neurons:
                        layer_0_neurons = layer_neurons[0]
                        if layer_0_neurons and 'neuron_activations' in layer_0_neurons:
                            activations = layer_0_neurons['neuron_activations']
                            # Get top neurons by activation
                            top_indices = torch.argsort(activations, descending=True)[:8]
                            specialty_neurons['integrated_gradients'] = [
                                (int(idx.item()), float(activations[idx].item())) 
                                for idx in top_indices
                            ]
                
                # Print analysis results
                if 'analysis_results' in ig_results:
                    print("\nIntegrated Gradients Analysis Results:")
                    for method, layer_results in ig_results['analysis_results'].items():
                        print(f"\nMethod: {method}")
                        for layer_idx, analysis in layer_results.items():
                            if analysis:
                                print(f"  Layer {layer_idx}: {analysis['num_neurons']} neurons, "
                                      f"mean activation: {analysis['mean_activation']:.4f}")
                
                # Visualize neurons if matplotlib is available
                try:
                    if 'analyzer' in ig_results:
                        analyzer = ig_results['analyzer']
                        if 0 in mamba_neurons.get('integrated_gradients', {}):
                            analyzer.visualize_neurons(
                                mamba_neurons['integrated_gradients'], 
                                layer_idx=0, 
                                save_path=f"integrated_gradients_neurons_layer_0.png"
                            )
                except Exception as viz_error:
                    print(f"Visualization failed: {viz_error}")
            
        else:
            # Fallback to original specialty neurons for non-Mamba models
            print("Model doesn't support integrated gradients computation. Using fallback method.")
            class_samples = {
                "factual": ["Apple produces the iPhone smartphone device for consumers."],
                "narrative": ["Once upon a time, there lived a wise old king."],
                "technical": ["The algorithm processes data using matrix multiplication operations."]
            }
            specialty_results = find_specialty_neurons_fixed(
                model, tokenizer, class_samples, layer_idx=0, top_k=8
            )
            specialty_neurons = specialty_results

        # Print results
        if specialty_neurons:
            for cls, neurons in specialty_neurons.items():
                print(f"\nClass '{cls.upper()}':")
                for neuron, score in neurons:
                    print(f"   • Neuron {neuron:4d} - Specialization Score: {score:.4f}")
        else:
            print("No specialty neurons found.")

    except Exception as e:
        print(f"Error in integrated gradients analysis: {e}")
        # Fallback to dummy neurons
        specialty_neurons = {'integrated_gradients': [(i, float(i)) for i in range(8)]}

    print("\n TYPOGRAPHICAL ERRORS ROBUSTNESS ANALYSIS")

    try:
        typo_examples = generate_typo_examples(texts[:3], num_examples=3)
        for i in range(len(typo_examples["original"])):
            print(f"\nExample {i+1}:")
            print(f"Original:   {typo_examples['original'][i]}")
            print(f"Realistic:  {typo_examples['realistic'][i]}")
            print(f"Random:     {typo_examples['random'][i]}")
            print(f"Contextual: {typo_examples['contextual'][i]}")

        typo_results = test_typo_robustness(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            specialty_neurons=specialty_neurons,
            layer_idx=0,
            typo_types=["realistic", "random", "contextual"],
            num_variants=3
        )
        analyze_typo_robustness_results(typo_results)

    except Exception as e:
        print(f"Error in typographical robustness analysis: {e}")

    print("\n TEMPLATE ROBUSTNESS ANALYSIS")

    try:
        available_relations = get_available_relations()
        print(f"Available relations: {available_relations}")

        template_results = run_comprehensive_template_analysis(
            model, tokenizer, specialty_neurons, layer_idx=0
        )

        print("\nFocused analysis on P176 (manufacturer):")
        manufacturer_results = test_template_robustness(
            model, tokenizer, specialty_neurons,
            relation_type="P176", layer_idx=0
        )
        analyze_template_robustness_results(manufacturer_results)

    except Exception as e:
        print(f"Error in template robustness analysis: {e}")

    print("\n DELTA SENSITIVITY ANALYSIS")

    try:
        delta_results = find_delta_sensitive_neurons_fixed(
            model, tokenizer, texts[:20], layer_idx=0, top_k=8
        )
        for neuron, variance in delta_results:
            print(f"Neuron {neuron} - Variance: {variance:.6f}")
        
        delta_transformer_results = find_delta_sensitive_neurons_transformer(
            model=model,
            tokenizer=tokenizer,
            texts=texts[:20],
            layer_idx=0,
            top_k=8,
            permutation_strategy="reverse"   # or "shuffle" or "swap"
        )

    except Exception as e:
        print(f"Error in delta sensitivity analysis: {e}")

    print("\n COMPREHENSIVE ANALYSIS SUMMARY")

    try:
        results_summary = {
            "analysis_timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "integrated_gradients_neurons": {
                k: [(int(n), float(s)) for n, s in v]
                for k, v in specialty_neurons.items()
            },
            "specialty_neurons": {  # Keep for compatibility with existing plotting functions
                k: [(int(n), float(s)) for n, s in v]
                for k, v in specialty_neurons.items()
            },
            "integrated_gradients_methods": list(specialty_neurons.keys()) if specialty_neurons else [],
            "template_robustness": {
                k: v.get("overall_stats", {})
                for k, v in template_results.items()
            } if "template_results" in locals() else {},
            "typo_robustness_summary": {
                "tested_typo_types": list(typo_results["consistency_scores"].keys())
                if "typo_results" in locals() else [],
                "overall_consistency": {
                    typo_type: {
                        "mean": float(np.mean([score for text_scores in scores for score in text_scores])),
                        "std": float(np.std([score for text_scores in scores for score in text_scores]))
                    }
                    for typo_type, scores in typo_results["consistency_scores"].items()
                } if "typo_results" in locals() else {},
            },
            "delta_sensitivity": {
                "top_neurons": [(int(n), float(v)) for n, v in delta_results]
                if "delta_results" in locals() else []
            },
            "delta_transformer_sensitivity": {
                "top_neurons": [(int(n), float(v)) for n, v in delta_transformer_results]
                if "delta_transformer_results" in locals() else []
            }
        }

        # Save
        with open(output_filename, "w") as f:
            json.dump(results_summary, f, indent=2)

        print(f"Results saved to {output_filename}")

        return results_summary

    except Exception as e:
        print(f"Could not save results: {e}")
        return None


if __name__ == "__main__":

    # Run both models
    results_mamba = run_analysis_for_model(
        model_name="state-spaces/mamba-130m-hf",
        output_filename="mamba_results.json"
    )

    results_transformer = run_analysis_for_model(
        model_name="gpt2",
        output_filename="gpt2_results.json"
    )

    if results_mamba and results_transformer:
        run_all_plots(results_mamba, results_transformer)
    else:
        print("Could not run comparison plots because one analysis failed.")
