import torch
import numpy as np
from utils import get_model_layers
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def get_top_activations(model, tokenizer, texts, neuron_index, layer_idx=0, top_k=20, max_len=128):
    """
    Extract top-k highest activations for a specific neuron across many texts.
    Returns a list of tuples: (activation value, token, decoded word, decoded context, full_text)
    """
    model.eval()
    # Ensure model is on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    layers = get_model_layers(model)
    layer = layers[layer_idx]

    results = []
    activations = []

    def hook(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        activations.append(hs[..., neuron_index].detach().cpu().numpy())

    handle = layer.register_forward_hook(hook)

    try:
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
            # Move inputs to the same device as model
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                _ = model(**inputs)
            act = activations.pop()  # shape: [1, seq_len]
            token_ids = inputs['input_ids'][0].tolist()
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            values = act[0]

            for idx, val in enumerate(values):
                token = tokens[idx]
                word = token[1:] if token.startswith('Ġ') else token
                context_ids = token_ids[max(0, idx - 2): idx + 3]
                decoded_context = tokenizer.decode(context_ids)
                results.append((val, token, word, decoded_context.strip(), text))
    finally:
        handle.remove()

    return sorted(results, key=lambda x: -x[0])[:top_k]

def visualize_top_activations(results, neuron_index, layer_idx=0):
    """
    Visualize top token activations.
    """
    print(f"Top activations for Neuron {neuron_index} (Layer {layer_idx}):")
    for i, (val, token, word, context, full_text) in enumerate(results):
        print(f"{i+1:2d}. Token: '{token}' | Word: '{word}' | Activation: {val:.4f} | Context: ...{context}...")

    words = [r[2] for r in results]
    values = [r[0] for r in results]

    plt.figure(figsize=(10, 4))
    sns.barplot(x=values, y=words, orient="h")
    plt.title(f"Top Activations for Neuron {neuron_index} (Layer {layer_idx})")
    plt.xlabel("Activation Value")
    plt.ylabel("Decoded Word")
    plt.tight_layout()
    
    # Save the plot to images folder
    import os
    os.makedirs("images", exist_ok=True)
    plt.savefig(f"images/feature_activations_neuron_{neuron_index}.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved feature activations plot to images/feature_activations_neuron_{neuron_index}.png")
    # plt.show()  # Not needed with non-interactive backend

# Example usage:
# from transformers import AutoModelForCausalLM, AutoTokenizer
# model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
# tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
# texts = load_analysis_texts()  # or your own corpus
# results = get_top_activations(model, tokenizer, texts, neuron_index=560)
# visualize_top_activations(results, neuron_index=560)
