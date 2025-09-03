#!/usr/bin/env python3
"""
Main script for running Mamba and GPT-2 neuron analysis
with attention weight method integration.
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


def run_attention_weight_analysis(model, tokenizer, device):
    """
    Run comprehensive attention weight analysis using the MambaAttentionNeurons approach.
    """
    print("🔍 Running Attention Weight Analysis...")
    
    # Sample texts for attention analysis
    attention_texts = [
        "The quick brown fox jumps over the lazy dog with great speed.",
        "Apple produces the iPhone smartphone device for global markets worldwide.",
        "Artificial intelligence is transforming the world through machine learning algorithms.",
        "Machine learning models require large datasets to achieve optimal performance.",
        "Natural language processing involves understanding and generating human text."
    ]
    
    # Tokenize texts for attention analysis
    attention_inputs = []
    for text in attention_texts:
        try:
            input_ids = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            attention_inputs.append(input_ids)
        except Exception as e:
            print(f"Error tokenizing text: {e}")
            continue
    
    if not attention_inputs:
        print("No valid inputs for attention analysis. Using dummy data.")
        # Create dummy input for testing
        dummy_input = torch.randint(0, 1000, (1, 64)).to(device)
        attention_inputs = [dummy_input]
    
    # Use the first input for analysis
    main_input = attention_inputs[0]
    
    try:
        # Check if model supports attention computation
        if hasattr(model, 'layers') and len(model.layers) > 0:
            print("Model supports attention computation. Running Mamba attention analysis...")
            
            # Run the attention analysis
            attention_results = integrate_mamba_attention_neurons(
                model=model,
                inputs=main_input,
                layer_indices=[0, 6, 12, 18],  # Analyze multiple layers
                methods=['attention_weighted', 'gradient_guided', 'rollout']
            )
            
            return attention_results, main_input
            
        else:
            print("Model doesn't support attention computation. Using fallback method.")
            return None, main_input
            
    except Exception as e:
        print(f"Error in attention weight analysis: {e}")
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

    print("\n ATTENTION WEIGHT NEURONS ANALYSIS")

    # Run the attention weight analysis
    attention_results, main_input = run_attention_weight_analysis(model, tokenizer, device)
    
    specialty_neurons = {}
    try:
        if attention_results:
            # Extract neurons from attention analysis
            if 'mamba_neurons' in attention_results:
                mamba_neurons = attention_results['mamba_neurons']
                
                # Convert to specialty neurons format for compatibility
                for method, layer_neurons in mamba_neurons.items():
                    if method == 'attention_weighted' and 0 in layer_neurons:
                        layer_0_neurons = layer_neurons[0]
                        if layer_0_neurons and 'neuron_activations' in layer_0_neurons:
                            activations = layer_0_neurons['neuron_activations']
                            # Get top neurons by activation
                            top_indices = torch.argsort(activations, descending=True)[:8]
                            specialty_neurons['attention_weighted'] = [
                                (int(idx.item()), float(activations[idx].item())) 
                                for idx in top_indices
                            ]
                
                # Print analysis results
                if 'analysis_results' in attention_results:
                    print("\nAttention Weight Analysis Results:")
                    for method, layer_results in attention_results['analysis_results'].items():
                        print(f"\nMethod: {method}")
                        for layer_idx, analysis in layer_results.items():
                            if analysis:
                                print(f"  Layer {layer_idx}: {analysis['num_neurons']} neurons, "
                                      f"mean activation: {analysis['mean_activation']:.4f}")
                
                # Visualize neurons if matplotlib is available
                try:
                    if 'analyzer' in attention_results:
                        analyzer = attention_results['analyzer']
                        if 0 in mamba_neurons.get('attention_weighted', {}):
                            analyzer.visualize_neurons(
                                mamba_neurons['attention_weighted'], 
                                layer_idx=0, 
                                save_path=f"attention_neurons_layer_0.png"
                            )
                except Exception as viz_error:
                    print(f"Visualization failed: {viz_error}")
            
        else:
            # Fallback to original specialty neurons for non-Mamba models
            print("Model doesn't support attention computation. Using fallback method.")
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
        print(f"Error in attention weight analysis: {e}")
        # Fallback to dummy neurons
        specialty_neurons = {'attention_weighted': [(i, float(i)) for i in range(8)]}

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
            "attention_weight_neurons": {
                k: [(int(n), float(s)) for n, s in v]
                for k, v in specialty_neurons.items()
            },
            "specialty_neurons": {  # Keep for compatibility with existing plotting functions
                k: [(int(n), float(s)) for n, s in v]
                for k, v in specialty_neurons.items()
            },
            "attention_analysis_methods": list(specialty_neurons.keys()) if specialty_neurons else [],
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
                "activation_counts": {
                    typo_type: {
                        "mean": float(np.mean([count for text_counts in counts for count in text_counts])),
                        "std": float(np.std([count for text_counts in counts for count in text_counts]))
                    }
                    for typo_type, counts in typo_results["activation_counts"].items()
                } if "typo_results" in locals() and "activation_counts" in typo_results else {},
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
