# Run main.py. then run - python3 3_run_comprehensive_ablation.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
import matplotlib.pyplot as plt
import numpy as np

# Import the main analysis functions and other helpers from your main.py
from main import run_comprehensive_analysis, plot_ablation_results, PLOTS_DIR

def load_wikitext_data():
    """
    Loads a small portion of the WikiText-2 dataset.
    """
    print("Loading WikiText-2 dataset...")
    try:
        # Load a small portion of the 'train' split for a manageable run time
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train[:1000]')
        # Extract the 'text' field and filter out empty strings
        texts = [d['text'] for d in dataset if d['text'].strip()]
        print(f"Loaded {len(texts)} texts from WikiText-2.")
        return texts
    except Exception as e:
        print(f"Failed to load dataset. Please ensure `datasets` library is installed. Error: {e}")
        return None

def load_real_mamba_model(model_name: str = "state-spaces/mamba-130m-hf"):
    """
    Loads a pre-trained Mamba model and its tokenizer from Hugging Face.
    """
    print(f"Loading Mamba model: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print("Model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load the model or tokenizer. Please ensure an internet connection. Error: {e}")
        return None, None

if __name__ == "__main__":
    # Define models to analyze
    mamba_model_name = "state-spaces/mamba-130m-hf"
    
    # Load the model, tokenizer, and data
    model, tokenizer = load_real_mamba_model(mamba_model_name)
    if model is None:
        exit()
        
    texts = load_wikitext_data()
    if texts is None:
        exit()

    # Define parameters for the analysis
    layer_to_analyze = 0
    top_k_delta_neurons = 10

    # Ensure the plots directory exists
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    # Run the comprehensive analysis which includes identifying and pruning dead, positional,
    # and delta-sensitive neurons, and checking performance. This will also generate
    # the requested plots.
    print("\nStarting comprehensive analysis with perplexity check...")
    results, _ = run_comprehensive_analysis(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        layer_idx=layer_to_analyze,
        top_k=top_k_delta_neurons,
        model_name=mamba_model_name
    )
    
    # Print the specific perplexity results for each type of ablation
    if "ablation_study" in results['analysis_results']:
        ablation_results = results['analysis_results']['ablation_study']
        print("\n--- Ablation Results Summary ---")
        if 'baseline' in ablation_results:
            print(f"Baseline Perplexity: {ablation_results['baseline']:.3f}")
        if 'dead_ablated' in ablation_results:
            print(f"Perplexity after pruning dead neurons: {ablation_results['dead_ablated']:.3f}")
        if 'positional_ablated' in ablation_results:
            print(f"Perplexity after pruning positional neurons: {ablation_results['positional_ablated']:.3f}")
        #if 'delta_ablated' in ablation_results:
        #    print(f"Perplexity after pruning delta-sensitive neurons: {ablation_results['delta_ablated']:.3f}")
        if 'dead_and_positional_ablated' in ablation_results:
            print(f"Perplexity after pruning combined dead and positional neurons: {ablation_results['dead_and_positional_ablated']:.3f}")
