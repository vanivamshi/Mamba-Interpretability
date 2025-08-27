# Run main.py. then run - python3 2_analyze_dead_neurons.py

# 2_analyze_dead_neurons.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import os
import json
import datetime

# Import updated function
from neuron_characterization import find_dead_neurons
from visualization_module import plot_dead_neuron_distribution


def load_and_analyze_model(model_name: str, text: str, layer_prefix: str, device: str = "cuda"):
    """
    Loads a model, tokenizes text, and identifies dead neurons for each layer.

    Args:
        model_name (str): The name of the model to load from Hugging Face.
        text (str): The text to be used for activation analysis.
        layer_prefix (str): The common prefix of the layer names to analyze.
        device (str): The device to run the model on ('cuda' or 'cpu').

    Returns:
        dict: A dictionary with layer numbers as keys and the percentage of dead neurons as values.
    """
    print(f"\n--- Analyzing {model_name} ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    except Exception as e:
        print(f"Failed to load model {model_name}. Error: {e}")
        return None

    # Split input text into smaller chunks
    text_chunks = [text[i:i+300] for i in range(0, len(text), 300)]

    layers = model.config.n_layer if hasattr(model.config, "n_layer") else len(model.transformer.h)
    hidden_size = model.config.hidden_size
    dead_neuron_data = {}

    for layer_idx in range(layers):
        dead_neurons, _ = find_dead_neurons(model, tokenizer, text_chunks, layer_idx=layer_idx)
        if dead_neurons is not None:
            dead_ratio = len(dead_neurons) / hidden_size
            dead_neuron_data[layer_idx] = dead_ratio
            print(f"Layer {layer_idx}: {dead_ratio:.2%} dead neurons")

    return dead_neuron_data


if __name__ == "__main__":
    models_to_analyze = {
        "Mamba-130M": {
            "name": "state-spaces/mamba-130m-hf",
            "layer_prefix": "layers."
        },
        "Mamba-370M": {
            "name": "state-spaces/mamba-370m-hf",
            "layer_prefix": "layers."
        },
        "Mamba-790M": {
            "name": "state-spaces/mamba-790m-hf",
            "layer_prefix": "layers."
        },
        "Mamba-1.4B": {
            "name": "state-spaces/mamba-1.4b-hf",
            "layer_prefix": "layers."
        },
        "Mamba-2.8B": {
            "name": "state-spaces/mamba-2.8b-hf",
            "layer_prefix": "layers."
        },
        "GPT-2": {
            "name": "gpt2",
            "layer_prefix": "h."
        }
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    sample_text = (
        "In the realm of deep learning, models like Transformers and State Space Models (SSMs) "
        "have revolutionized natural language processing. While Transformers excel at capturing "
        "global dependencies with their self-attention mechanism, models like Mamba, a selective "
        "SSM, offer a compelling alternative by providing linear-time complexity and efficient "
        "hardware utilization. This allows Mamba to handle very long sequences with high throughput. "
        "However, a key area of analysis is understanding the internal behavior of these models, "
        "such as the distribution of dead neurons across different layers and architectures."
    )

    all_models_data = {}
    for model_label, model_info in models_to_analyze.items():
        dead_neurons_data = load_and_analyze_model(
            model_info["name"],
            sample_text,
            model_info["layer_prefix"],
            device
        )
        if dead_neurons_data:
            all_models_data[model_label] = dead_neurons_data

    if all_models_data:
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)

        plot_path = os.path.join(plots_dir, "dead_neuron_distribution.png")
        plot_dead_neuron_distribution(all_models_data, save_path=plot_path)
        print(f"\nPlot saved to '{plot_path}'.")
        
        # Save data to log file for offline plotting
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"dead_neurons_analysis_{timestamp}.json")
        
        log_data = {
            "timestamp": timestamp,
            "device": device,
            "sample_text_length": len(sample_text),
            "models_data": all_models_data
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        print(f"✅ Data logged to '{log_file}' for offline plotting.")
        
    else:
        print("No model data to plot. Please check for errors during model loading and analysis.")
