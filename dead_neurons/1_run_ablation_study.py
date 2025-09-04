# Run main.py. then run - python3 1_run_ablation_study.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
import matplotlib.pyplot as plt
import numpy as np

# Import the main analysis function from main.py
from main import run_comprehensive_analysis


def load_wikitext_data():
    """
    Loads a small portion of the WikiText-2 dataset.
    """
    print("Loading WikiText-2 dataset...")
    try:
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train[:1000]')
        texts = [d['text'] for d in dataset if d['text'].strip()]
        print(f"Loaded {len(texts)} texts from WikiText-2.")
        return texts
    except Exception as e:
        print(f"Failed to load dataset. Please ensure `datasets` library is installed. Error: {e}")
        return None


def load_real_mamba_model(model_name: str = "state-spaces/mamba-130m-hf"):
    """
    Loads a pre-trained model (Mamba or GPT-like) and its tokenizer.
    """
    print(f"Loading model: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print("Model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load the model or tokenizer. Please ensure an internet connection. Error: {e}")
        return None, None


def get_layer_container(model):
    """
    Detects the correct layer container inside the model (works for GPT + Mamba).
    Returns (num_layers, container).
    """
    # GPT-style (e.g., GPT2, OPT, LLaMA)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h), model.transformer.h

    # Some models put layers directly under `model.layers`
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers), model.model.layers

    # Mamba HF usually uses backbone.layers
    if hasattr(model, "backbone") and hasattr(model.backbone, "layers"):
        return len(model.backbone.layers), model.backbone.layers

    # Direct .layers
    if hasattr(model, "layers"):
        return len(model.layers), model.layers

    # As fallback, search for ModuleList
    for name, module in model.named_children():
        if isinstance(module, torch.nn.ModuleList):
            return len(module), module

    raise AttributeError("Could not detect the layer container in this model.")


if __name__ == "__main__":
    mamba_model_name = "state-spaces/mamba-130m-hf"

    # Load model + tokenizer
    model, tokenizer = load_real_mamba_model(mamba_model_name)
    if model is None:
        exit()

    texts = load_wikitext_data()
    if texts is None:
        exit()

    # Detect number of layers
    num_layers, layer_container = get_layer_container(model)
    print(f"\nModel has {num_layers} layers in {type(layer_container)}. Running ablation study...")

    top_k_delta_neurons = 10
    all_results = {}

    # Run per-layer analysis
    for layer_idx in range(num_layers):
        print(f"\n=== Analyzing Layer {layer_idx} ===")
        results, delta_neurons = run_comprehensive_analysis(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            layer_idx=layer_idx,
            top_k=top_k_delta_neurons,
            model_name=mamba_model_name
        )
        all_results[layer_idx] = results

        if "ablation_study" in results['analysis_results']:
            ablation_results = results['analysis_results']['ablation_study']
            baseline = ablation_results.get("baseline")
            dead_ablated = ablation_results.get("dead_ablated")

            if baseline is not None:
                print(f"Baseline Perplexity: {baseline:.3f}")
            if dead_ablated is not None:
                print(f"Perplexity after pruning dead neurons: {dead_ablated:.3f}")

    # Print summary
    print("\n==== Summary of Results ====")
    for layer_idx, results in all_results.items():
        ablation = results['analysis_results'].get("ablation_study", {})
        baseline = ablation.get("baseline", None)
        dead_ablated = ablation.get("dead_ablated", None)
        baseline_str = f"{baseline:.3f}" if baseline is not None else "N/A"
        ablated_str = f"{dead_ablated:.3f}" if dead_ablated is not None else "N/A"
        print(f"Layer {layer_idx}: Baseline={baseline_str}, Dead Ablated={ablated_str}")
