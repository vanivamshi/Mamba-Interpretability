import torch
import numpy as np
from utils import get_model_layers

def find_projection_dominant_neurons_fixed(model, layer_idx=0, top_k=10):
    """
    Find neurons with dominant projection weights
    Works for both Mamba and standard Transformers (e.g. GPT-2).
    """
    try:
        layers = get_model_layers(model)
        if layers is None or layer_idx >= len(layers):
            print("Could not access model layers. Using dummy values.")
            return [(i, float(i + 1)) for i in range(top_k)]

        layer = layers[layer_idx]

        projection_weights = None

        # Search for projections in known places
        possible_projections = [
            # Mamba
            lambda l: l.mixer.x_proj.weight
                if hasattr(l, 'mixer') and hasattr(l.mixer, 'x_proj')
                and hasattr(l.mixer.x_proj, 'weight')
                else None,
            lambda l: l.mixer.in_proj.weight
                if hasattr(l, 'mixer') and hasattr(l.mixer, 'in_proj')
                and hasattr(l.mixer.in_proj, 'weight')
                else None,
            lambda l: l.mixer.dt_proj.weight
                if hasattr(l, 'mixer') and hasattr(l.mixer, 'dt_proj')
                and hasattr(l.mixer.dt_proj, 'weight')
                else None,
            lambda l: l.in_proj.weight
                if hasattr(l, 'in_proj') and hasattr(l.in_proj, 'weight')
                else None,
            lambda l: l.linear.weight
                if hasattr(l, 'linear') and hasattr(l.linear, 'weight')
                else None,
            # Transformers (e.g. GPT-2)
            lambda l: l.attn.c_proj.weight
                if hasattr(l, 'attn') and hasattr(l.attn, 'c_proj')
                and hasattr(l.attn.c_proj, 'weight')
                else None,
            lambda l: l.mlp.c_proj.weight
                if hasattr(l, 'mlp') and hasattr(l.mlp, 'c_proj')
                and hasattr(l.mlp.c_proj, 'weight')
                else None,
        ]

        for proj_fn in possible_projections:
            try:
                weights = proj_fn(layer)
                if weights is not None:
                    projection_weights = weights.detach().cpu().numpy()
                    break
            except AttributeError:
                continue

        if projection_weights is None:
            print("Could not find projection weights. Using dummy values.")
            return [(i, float(i + 1)) for i in range(top_k)]

        # Compute magnitude per neuron
        if projection_weights.ndim == 2:
            # Shape: (output_dim, input_dim)
            magnitudes = np.linalg.norm(projection_weights, axis=0)
        else:
            magnitudes = np.abs(projection_weights)

        # Check hidden size for this layer
        dummy_input = torch.randint(0, 100, (1, 10))
        with torch.no_grad():
            output = model(dummy_input, output_hidden_states=True)
            hidden_size = output.hidden_states[layer_idx].shape[-1]

        valid_indices = [i for i in range(len(magnitudes)) if i < hidden_size]
        magnitudes = magnitudes[valid_indices]

        if len(magnitudes) < top_k:
            top_k = len(magnitudes)

        top_dims = np.argsort(magnitudes)[-top_k:]
        return [(int(i), float(magnitudes[i])) for i in top_dims[::-1]]

    except Exception as e:
        print(f"Error in find_projection_dominant_neurons_fixed: {e}")
        return [(i, float(i + 1)) for i in range(top_k)]


def functional_importance_projection_neurons(model, tokenizer, prompt, layer_idx=0, top_k=5):
    """Get neurons that assess functional contribution to the projection-weighted neurons."""
    try:
        layers = get_model_layers(model)
        if layers is None or layer_idx >= len(layers):
            print("Could not access model layers. Using dummy values.")
            return [(i, float(i + 1)) for i in range(top_k)]

        layer = layers[layer_idx]

        # Try different possible paths to find projection weights
        projection_weights = None
        possible_projections = [
            lambda l: l.mixer.x_proj.weight if hasattr(l, 'mixer') and hasattr(l.mixer, 'x_proj') and hasattr(l.mixer.x_proj, 'weight') else None,
            lambda l: l.mixer.in_proj.weight if hasattr(l, 'mixer') and hasattr(l.mixer, 'in_proj') and hasattr(l.mixer.in_proj, 'weight') else None,
            lambda l: l.mixer.dt_proj.weight if hasattr(l, 'mixer') and hasattr(l.mixer, 'dt_proj') and hasattr(l.mixer.dt_proj, 'weight') else None,
            lambda l: l.in_proj.weight if hasattr(l, 'in_proj') and hasattr(l.in_proj, 'weight') else None,
            lambda l: l.linear.weight if hasattr(l, 'linear') and hasattr(l.linear, 'weight') else None,
        ]

        for proj_fn in possible_projections:
            try:
                weights = proj_fn(layer)
                if weights is not None:
                    projection_weights = weights.detach().cpu().numpy()  # shape: (output_dim, input_dim)
                    break
            except AttributeError:
                continue

        if projection_weights is None:
            print("Could not find projection weights. Using dummy values.")
            return [(i, float(i + 1)) for i in range(top_k)]

        # Tokenize and run the model
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Get mean activation from the layer
        layer_activations = outputs.hidden_states[layer_idx]  # [1, seq_len, hidden_dim]
        avg_activation = layer_activations.mean(dim=1).squeeze(0).cpu().numpy()  # [hidden_dim]

        # Resize avg_activation if necessary
        in_dim = projection_weights.shape[1]
        act_dim = avg_activation.shape[0]

        if in_dim % act_dim != 0:
            raise ValueError(f"Cannot resize activation vector of dim {act_dim} to match projection weight input dim {in_dim}")

        repeats = in_dim // act_dim
        avg_activation_resized = np.tile(avg_activation, repeats)  # [in_dim]

        # Element-wise product and row-wise norm
        functional_scores = projection_weights * avg_activation_resized[np.newaxis, :]  # [output_dim, in_dim]
        functional_magnitudes = np.linalg.norm(functional_scores, axis=1)  # [output_dim]

        if len(functional_magnitudes) < top_k:
            top_k = len(functional_magnitudes)

        top_dims = np.argsort(functional_magnitudes)[-top_k:]
        return [(int(i), float(functional_magnitudes[i])) for i in top_dims[::-1]]

    except Exception as e:
        print(f"Error in functional_importance_projection_neurons: {e}")
        return [(i, float(i + 1)) for i in range(top_k)]


def ablate_neurons_hook(neuron_indices):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            output[:, :, neuron_indices] = 0  # (batch, seq, hidden)
        elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
            output = list(output)
            output[0][:, :, neuron_indices] = 0
            output = tuple(output)
        return output
    return hook

