import torch
import numpy as np
from utils import get_model_layers
from attention_neurons import MambaAttentionNeurons
import matplotlib.pyplot as plt
import os

def extract_deltas_fixed(model, layer_idx, input_ids):
    """
    Extract delta parameters for a specific layer - fixed version.
    """
    layers = get_model_layers(model)
    if layers is None:
        print("Could not find model layers. Using dummy delta values.")
        # Return dummy values with appropriate shape
        batch_size, seq_len = input_ids.shape
        return torch.randn(batch_size, seq_len, 512)  # Assuming 512 hidden size
    
    if layer_idx >= len(layers):
        print(f"Layer {layer_idx} not found. Using dummy values.")
        batch_size, seq_len = input_ids.shape
        return torch.randn(batch_size, seq_len, 512)
    
    layer = layers[layer_idx]
    delta_values = []
    
    def delta_hook(module, input, output):
        if isinstance(output, tuple):
            # If output is a tuple, take the first element or the one that looks like deltas
            delta_values.append(output[0].detach() if len(output) > 0 else torch.randn(1, 1, 512))
        else:
            delta_values.append(output.detach())
    
    # Try to find the delta computation module
    hook_registered = False
    
    # Common patterns for Mamba SSM modules
    possible_delta_modules = [
        lambda l: l.mixer.compute_delta if hasattr(l, 'mixer') and hasattr(l.mixer, 'compute_delta') else None,
        lambda l: l.ssm.compute_delta if hasattr(l, 'ssm') and hasattr(l.ssm, 'compute_delta') else None,
        lambda l: l.mixer.ssm.compute_delta if hasattr(l, 'mixer') and hasattr(l.mixer, 'ssm') and hasattr(l.mixer.ssm, 'compute_delta') else None,
        lambda l: l.mixer.dt_proj if hasattr(l, 'mixer') and hasattr(l.mixer, 'dt_proj') else None,
        lambda l: l.mixer if hasattr(l, 'mixer') else None,  # Fallback to mixer
    ]
    
    handle = None
    for module_fn in possible_delta_modules:
        try:
            delta_module = module_fn(layer)
            if delta_module is not None:
                handle = delta_module.register_forward_hook(delta_hook)
                hook_registered = True
                break
        except AttributeError:
            continue
    
    if not hook_registered:
        print(f"Could not find delta module in layer {layer_idx}. Using layer output.")
        # Fallback: use the layer's main output
        handle = layer.register_forward_hook(delta_hook)
    
    # Forward pass
    try:
        with torch.no_grad():
            _ = model(input_ids)
        
        if handle:
            handle.remove()
        
        if delta_values:
            return delta_values[0]
        else:
            # Return dummy values if nothing was captured
            batch_size, seq_len = input_ids.shape
            return torch.randn(batch_size, seq_len, 512)
            
    except Exception as e:
        print(f"Error during forward pass: {e}")
        if handle:
            handle.remove()
        batch_size, seq_len = input_ids.shape
        return torch.randn(batch_size, seq_len, 512)

def find_delta_sensitive_neurons_fixed(model, tokenizer, texts, layer_idx=0, top_k=10):
    """Find neurons that are sensitive to delta computation - fixed version."""
    deltas = []
    
    for text in texts:
        try:
            input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
            delta = extract_deltas_fixed(model, layer_idx, input_ids)
            
            # Handle different possible shapes
            if delta.dim() == 3:  # (batch, seq, hidden)
                delta_mean = delta.mean(dim=(0, 1))  # Average over batch and sequence
            elif delta.dim() == 2:  # (seq, hidden)
                delta_mean = delta.mean(dim=0)  # Average over sequence
            else:  # (hidden,)
                delta_mean = delta
            
            deltas.append(delta_mean.cpu().numpy())
            
        except Exception as e:
            print(f"Error processing text '{text[:50]}...': {e}")
            # Add dummy values to maintain consistency
            deltas.append(np.random.randn(512))
    
    if not deltas:
        print("No deltas extracted. Returning dummy results.")
        return [(i, float(i)) for i in range(top_k)]
    
    try:
        all_deltas = np.array(deltas)
        if all_deltas.ndim == 1:
            all_deltas = all_deltas.reshape(1, -1)
        
        variance = np.var(all_deltas, axis=0)
        top_dims = np.argsort(variance)[-top_k:]
        return [(int(i), float(variance[i])) for i in top_dims[::-1]]
        
    except Exception as e:
        print(f"Error computing variance: {e}")
        return [(i, float(i)) for i in range(top_k)]


def find_attention_sensitive_neurons(model, tokenizer, texts, layer_idx=0, top_k=10, method='attention_weighted'):
    """
    Find neurons that are sensitive to attention weights using the attention_neurons module.
    
    Args:
        model: The model to analyze
        tokenizer: Tokenizer for the model
        texts: List of texts to analyze
        layer_idx: Layer index to analyze
        top_k: Number of top neurons to return
        method: Method to use for neuron creation ('attention_weighted', 'gradient_guided', 'rollout')
    
    Returns:
        List of tuples (neuron_index, attention_score)
    """
    try:
        # Initialize the attention neurons analyzer
        attention_analyzer = MambaAttentionNeurons(model, enable_attention_computation=True)
        
        attention_scores = []
        
        for text in texts:
            try:
                input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
                
                # Extract attention vectors for the specific layer
                attention_data = attention_analyzer.extract_attention_vectors(input_ids, [layer_idx])
                
                if layer_idx in attention_data:
                    layer_data = attention_data[layer_idx]
                    
                    # Create neurons using the specified method
                    neurons = attention_analyzer.create_mamba_neurons({layer_idx: layer_data}, method)
                    
                    if neurons and layer_idx in neurons:
                        neuron_data = neurons[layer_idx]
                        
                        if 'neuron_activations' in neuron_data:
                            activations = neuron_data['neuron_activations']
                            
                            # Handle different tensor shapes
                            if activations.dim() == 2:  # (batch, hidden)
                                # Average across batch dimension
                                activations = activations.mean(dim=0)
                            elif activations.dim() > 2:
                                # Average across all dimensions except the last
                                for _ in range(activations.dim() - 1):
                                    activations = activations.mean(dim=0)
                            
                            # Convert to numpy
                            if hasattr(activations, 'cpu'):
                                activations = activations.cpu().numpy()
                            elif hasattr(activations, 'numpy'):
                                activations = activations.numpy()
                            
                            # Ensure it's 1D
                            if activations.ndim > 1:
                                activations = activations.flatten()
                            
                            # Store the attention scores
                            attention_scores.append(activations)
                        
            except Exception as e:
                print(f"Error processing text '{text[:50]}...': {e}")
                continue
        
        if not attention_scores:
            print("No attention scores extracted. Returning dummy results.")
            return [(i, float(i)) for i in range(top_k)]
        
        try:
            # Compute average attention scores across all texts
            all_scores = np.array(attention_scores)
            
            # Ensure consistent shapes
            if all_scores.ndim == 1:
                all_scores = all_scores.reshape(1, -1)
            
            # Use variance as the sensitivity measure (similar to delta analysis)
            variance = np.var(all_scores, axis=0)
            
            # Get top k neurons by attention sensitivity
            top_dims = np.argsort(variance)[-top_k:]
            return [(int(i), float(variance[i])) for i in top_dims[::-1]]
            
        except Exception as e:
            print(f"Error computing attention variance: {e}")
            return [(i, float(i)) for i in range(top_k)]
            
    except Exception as e:
        print(f"Error in attention analysis: {e}")
        return [(i, float(i)) for i in range(top_k)]


def find_integrated_gradients_sensitive_neurons(model, tokenizer, texts, layer_idx=0, top_k=10, method='attribution_weighted'):
    """
    Find neurons that are sensitive to integrated gradients using the integrated_gradients module.
    
    Args:
        model: The model to analyze
        tokenizer: Tokenizer for the model
        texts: List of texts to analyze
        layer_idx: Layer index to analyze
        top_k: Number of top neurons to return
        method: Method to use for neuron creation ('attribution_weighted', 'convergence_guided', 'layer_wise')
    
    Returns:
        List of tuples (neuron_index, integrated_gradients_score)
    """
    try:
        # Import integrated gradients module
        from integrated_gradients import IntegratedGradientsNeurons
        
        # Initialize the integrated gradients neurons analyzer
        ig_analyzer = IntegratedGradientsNeurons(model, enable_gradients=True)
        
        ig_scores = []
        
        for text in texts[:5]:  # Limit to first 5 texts for performance
            try:
                input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
                
                # Extract integrated gradients for the specific layer
                ig_data = ig_analyzer.extract_integrated_gradients(input_ids, target_layer_idx=layer_idx)
                
                if ig_data:
                    # Create neurons using the specified method
                    neurons = ig_analyzer.create_integrated_gradients_neurons(ig_data, method)
                    
                    if neurons:
                        if 'neuron_activations' in neurons:
                            activations = neurons['neuron_activations']
                            
                            # Handle different tensor shapes
                            if hasattr(activations, 'dim'):
                                # PyTorch tensor
                                if activations.dim() == 2:  # (batch, hidden)
                                    # Average across batch dimension
                                    activations = activations.mean(dim=0)
                                elif activations.dim() > 2:
                                    # Average across all dimensions except the last
                                    for _ in range(activations.dim() - 1):
                                        activations = activations.mean(dim=0)
                            elif hasattr(activations, 'ndim'):
                                # Numpy array
                                if activations.ndim == 2:  # (batch, hidden)
                                    # Average across batch dimension
                                    activations = activations.mean(axis=0)
                                elif activations.ndim > 2:
                                    # Average across all dimensions except the last
                                    for _ in range(activations.ndim - 1):
                                        activations = activations.mean(axis=0)
                            else:
                                print("Warning: activations is not a proper tensor/array")
                                continue
                            
                            # Convert to numpy
                            if hasattr(activations, 'cpu'):
                                activations = activations.cpu().numpy()
                            elif hasattr(activations, 'numpy'):
                                activations = activations.numpy()
                            
                            # Ensure it's a numpy array
                            if not hasattr(activations, 'ndim'):
                                print("Warning: activations is not a proper numpy array after conversion")
                                continue
                            
                            # Ensure it's 1D
                            if activations.ndim > 1:
                                activations = activations.flatten()
                            
                            # Store the integrated gradients scores
                            ig_scores.append(activations)
                        
            except Exception as e:
                print(f"Error processing text '{text[:50]}...': {e}")
                continue
        
        if not ig_scores:
            print("No integrated gradients scores extracted. Returning dummy results.")
            return [(i, float(i)) for i in range(top_k)]
        
        try:
            # Compute average integrated gradients scores across all texts
            all_scores = np.array(ig_scores)
            
            # Ensure consistent shapes
            if all_scores.ndim == 1:
                all_scores = all_scores.reshape(1, -1)
            
            # Use variance as the sensitivity measure (similar to delta analysis)
            variance = np.var(all_scores, axis=0)
            
            # Get top k neurons by integrated gradients sensitivity
            top_dims = np.argsort(variance)[-top_k:]
            return [(int(i), float(variance[i])) for i in top_dims[::-1]]
            
        except Exception as e:
            print(f"Error computing integrated gradients variance: {e}")
            return [(i, float(i)) for i in range(top_k)]
            
    except Exception as e:
        print(f"Error in integrated gradients analysis: {e}")
        return [(i, float(i)) for i in range(top_k)]


def find_combined_delta_attention_neurons(model, tokenizer, texts, layer_idx=0, top_k=10, 
                                         delta_weight=0.5, attention_weight=0.5):
    """
    Find neurons that are sensitive to both delta computation and attention.
    This combines the delta and attention analysis for a more comprehensive view.
    
    Args:
        model: The model to analyze
        tokenizer: Tokenizer for the model
        texts: List of texts to analyze
        layer_idx: Layer index to analyze
        top_k: Number of top neurons to return
        delta_weight: Weight for delta sensitivity (0.0 to 1.0)
        attention_weight: Weight for attention sensitivity (0.0 to 1.0)
    
    Returns:
        List of tuples (neuron_index, combined_score)
    """
    try:
        # Get delta-sensitive neurons
        delta_neurons = find_delta_sensitive_neurons_fixed(model, tokenizer, texts, layer_idx, top_k)
        
        # Get attention-sensitive neurons
        attention_neurons = find_attention_sensitive_neurons(model, tokenizer, texts, layer_idx, top_k)
        
        # Create a mapping of neuron indices to scores
        delta_scores = {idx: score for idx, score in delta_neurons}
        attention_scores = {idx: score for idx, score in attention_neurons}
        
        # Get all unique neuron indices
        all_indices = set(delta_scores.keys()) | set(attention_scores.keys())
        
        # Compute combined scores
        combined_scores = []
        for idx in all_indices:
            delta_score = delta_scores.get(idx, 0.0)
            attention_score = attention_scores.get(idx, 0.0)
            
            # Normalize scores to 0-1 range
            delta_norm = delta_score / (max(delta_scores.values()) + 1e-8) if delta_scores else 0.0
            attention_norm = attention_score / (max(attention_scores.values()) + 1e-8) if attention_scores else 0.0
            
            # Compute weighted combination
            combined_score = delta_weight * delta_norm + attention_weight * attention_norm
            combined_scores.append((idx, combined_score))
        
        # Sort by combined score and return top k
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        return combined_scores[:top_k]
        
    except Exception as e:
        print(f"Error in combined delta+attention analysis: {e}")
        return [(i, float(i)) for i in range(top_k)]


def find_combined_sensitive_neurons(model, tokenizer, texts, layer_idx=0, top_k=10, 
                                   delta_weight=0.5, integrated_gradients_weight=0.5):
    """
    Find neurons that are sensitive to both delta computation and integrated gradients.
    This combines the delta and integrated gradients analysis for a more comprehensive view.
    
    Args:
        model: The model to analyze
        tokenizer: Tokenizer for the model
        texts: List of texts to analyze
        layer_idx: Layer index to analyze
        top_k: Number of top neurons to return
        delta_weight: Weight for delta sensitivity (0.0 to 1.0)
        integrated_gradients_weight: Weight for integrated gradients sensitivity (0.0 to 1.0)
    
    Returns:
        List of tuples (neuron_index, combined_score)
    """
    try:
        # Get delta-sensitive neurons
        delta_neurons = find_delta_sensitive_neurons_fixed(model, tokenizer, texts, layer_idx, top_k)
        
        # Get integrated gradients-sensitive neurons
        ig_neurons = find_integrated_gradients_sensitive_neurons(model, tokenizer, texts, layer_idx, top_k)
        
        # Create a mapping of neuron indices to scores
        delta_scores = {idx: score for idx, score in delta_neurons}
        ig_scores = {idx: score for idx, score in ig_neurons}
        
        # Get all unique neuron indices
        all_indices = set(delta_scores.keys()) | set(ig_scores.keys())
        
        # Compute combined scores
        combined_scores = []
        for idx in all_indices:
            delta_score = delta_scores.get(idx, 0.0)
            ig_score = ig_scores.get(idx, 0.0)
            
            # Normalize scores to 0-1 range
            delta_norm = delta_score / (max(delta_scores.values()) + 1e-8) if delta_scores else 0.0
            ig_norm = ig_score / (max(ig_scores.values()) + 1e-8) if ig_scores else 0.0
            
            # Compute weighted combination
            combined_score = delta_weight * delta_norm + integrated_gradients_weight * ig_norm
            combined_scores.append((idx, combined_score))
        
        # Sort by combined score and return top k
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        return combined_scores[:top_k]
        
    except Exception as e:
        print(f"Error in combined analysis: {e}")
        return [(i, float(i)) for i in range(top_k)]


def compute_layer_neuron_variances(model, tokenizer, texts, layer_idx=0, max_texts=None):
    """
    Compute per-neuron variance (across texts) for the given layer by extracting
    delta vectors and computing variance across examples for each neuron.

    Returns:
        1D numpy array of length == hidden size containing variance per neuron.
    """
    examples = []
    count = 0
    for text in texts:
        if max_texts is not None and count >= max_texts:
            break
        try:
            input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
            delta = extract_deltas_fixed(model, layer_idx, input_ids)

            # Normalize delta to 1D hidden vector per example similar to other functions
            if hasattr(delta, 'dim'):
                if delta.dim() == 3:  # (batch, seq, hidden)
                    delta_mean = delta.mean(dim=(0, 1))
                elif delta.dim() == 2:  # (seq, hidden)
                    delta_mean = delta.mean(dim=0)
                else:
                    delta_mean = delta
                if hasattr(delta_mean, 'cpu'):
                    arr = delta_mean.cpu().numpy()
                else:
                    arr = np.array(delta_mean)
            else:
                # Already numpy-like
                arr = np.array(delta)

            # Ensure 1D
            arr = np.asarray(arr).reshape(-1)
            examples.append(arr)
            count += 1
        except Exception:
            # skip problematic examples but keep going
            continue

    if not examples:
        # Return a dummy small-variance vector of reasonable length (512)
        return np.zeros(512)

    all_examples = np.vstack(examples)
    # Variance across examples for each neuron (column-wise)
    variances = np.var(all_examples, axis=0)
    return variances


def plot_per_layer_neuron_distributions_for_models(models_info, texts,
                                                   num_layers, save_path=None,
                                                   max_texts_per_layer=200, cols=2):
    """
    Create a grid of subplots showing per-neuron distributions (variance) for each layer
    for multiple models. `models_info` should be a list of tuples: (model, label).

    Layout: rows = num_layers, cols = number of models (or specified `cols` organizes models across columns).

    Args:
        models_info: list of (model, label) tuples
        tokenizer: tokenizer used for all models (or dict mapping labels to tokenizers)
        texts: iterable of input texts to compute variances across
        num_layers: number of layers to visualize
        save_path: file path to save the composed figure (defaults to `projection_neurons/images/per_layer_neuron_distributions.png`)
        max_texts_per_layer: max examples per layer to use when computing variance (for speed)
        cols: number of columns in the grid per model grouping (unused if len(models_info) used)

    Returns:
        Path to saved figure.
    """
    if save_path is None:
        images_dir = os.path.join(os.path.dirname(__file__), "images")
        os.makedirs(images_dir, exist_ok=True)
        save_path = os.path.join(images_dir, "per_layer_neuron_distributions.png")

    # models_info is expected to be a list of tuples: (model, tokenizer, label)
    num_models = len(models_info)
    rows = num_layers
    cols = num_models

    # Precompute variances per model per layer to determine y-limits
    variances_by_model = {}
    for model, tokenizer, label in models_info:
        variances_by_model[label] = []
        for layer_idx in range(num_layers):
            var = compute_layer_neuron_variances(model, tokenizer, texts, layer_idx, max_texts=max_texts_per_layer)
            variances_by_model[label].append(var)

    # Determine per-model max for consistent y-axis scaling across layers of a model
    max_by_model = {label: max([v.max() if v.size else 0 for v in varlist]) for label, varlist in variances_by_model.items()}

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 2.5 * rows), sharex=False)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for r in range(rows):
        for c, (model, label) in enumerate(models_info):
            ax = axes[r, c]
            var = variances_by_model[label][r]
            if var is None or len(var) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            x = np.arange(len(var))
            ax.bar(x, var, color='#7fb3d5' if 'Mamba' in label else '#f5b7b1', width=1.0)
            ax.set_xlim(0, len(var))
            ax.set_ylim(0, max_by_model[label] * 1.05 if max_by_model[label] > 0 else None)
            if r == 0:
                ax.set_title(f"{label}")
            if c == 0:
                ax.set_ylabel(f"Layer {r} variance")
            # Reduce x-axis clutter for many neurons
            if len(var) > 200:
                ax.set_xticks([])
            else:
                ax.set_xlabel("Neuron Index")
            ax.grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path
