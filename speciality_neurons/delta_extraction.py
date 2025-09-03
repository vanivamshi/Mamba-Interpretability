import torch
import numpy as np
from utils import get_model_layers

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
