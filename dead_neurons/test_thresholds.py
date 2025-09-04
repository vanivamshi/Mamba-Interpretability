#!/usr/bin/env python3
"""
Quick script to test different thresholds for dead neuron detection.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from neuron_characterization import plot_neuron_activation_distribution
import argparse

def test_thresholds(model_name="state-spaces/mamba-130m-hf", layer_idx=1, num_texts=100):
    """Test different thresholds to find optimal dead neuron detection."""
    
    print(f"Testing thresholds for {model_name}, layer {layer_idx}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load some sample texts
    try:
        from datasets import load_dataset
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")
        texts = [item["text"] for item in dataset if item["text"].strip()]
        texts = [text for text in texts if len(text.split()) > 10 and not text.startswith("=")]
        texts = texts[:num_texts]
    except:
        texts = [
            "Artificial intelligence is transforming industries.",
            "The quick brown fox jumps over the lazy dog.",
            "Transformer models have revolutionized NLP tasks.",
            "Quantum computing promises exponential speedup.",
            "She loves chocolate. She hates chocolate."
        ] * 20  # Repeat to get more texts
    
    print(f"Using {len(texts)} texts for analysis")
    
    # Test different thresholds
    thresholds_to_test = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    
    for threshold in thresholds_to_test:
        print(f"\n{'='*50}")
        print(f"Testing threshold: {threshold}")
        print(f"{'='*50}")
        
        # Generate activation analysis with this threshold
        max_acts, activation_freq = plot_neuron_activation_distribution(
            model, tokenizer, texts, layer_idx, 
            save_dir="plots", test_threshold=threshold
        )
        
        # Count dead neurons with this threshold
        dead_count = np.sum(max_acts < threshold)
        total_neurons = len(max_acts)
        dead_percentage = (dead_count / total_neurons) * 100
        
        print(f"\n📊 SUMMARY for threshold {threshold}:")
        print(f"   Dead neurons: {dead_count}/{total_neurons} ({dead_percentage:.1f}%)")
        
        if 5 <= dead_percentage <= 20:
            print(f"   ✅ GOOD: This threshold gives a reasonable number of dead neurons")
        elif dead_percentage < 5:
            print(f"   ⚠️  LOW: Too few dead neurons detected")
        else:
            print(f"   ⚠️  HIGH: Too many dead neurons detected")
        
        # Ask user if they want to continue
        if threshold != thresholds_to_test[-1]:
            response = input(f"\nContinue to next threshold? (y/n): ")
            if response.lower() != 'y':
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="state-spaces/mamba-130m-hf", help='Model name')
    parser.add_argument('--layer', type=int, default=1, help='Layer index')
    parser.add_argument('--texts', type=int, default=100, help='Number of texts to use')
    
    args = parser.parse_args()
    test_thresholds(args.model, args.layer, args.texts) 