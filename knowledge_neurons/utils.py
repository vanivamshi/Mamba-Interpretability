import torch
import numpy as np

def debug_model_structure(model, max_depth=3, current_depth=0, prefix=""):
    """Debug function to understand the model structure"""
    if current_depth >= max_depth:
        return
    
    for name, module in model.named_children():
        print(f"{prefix}{name}: {type(module).__name__}")
        if current_depth < max_depth - 1:
            debug_model_structure(module, max_depth, current_depth + 1, prefix + "  ")

def get_model_layers(model):
    """Get the layers from the model, handling different possible structures"""
    # Try different possible paths to access layers
    possible_paths = [
        lambda m: m.backbone.layers,  # Standard Mamba structure
        lambda m: m.model.layers,     # Alternative structure
        lambda m: m.layers,           # Direct access
        lambda m: m.transformer.h,    # Transformer-like structure
        lambda m: m.transformer.layers,
    ]
    
    for path_fn in possible_paths:
        try:
            layers = path_fn(model)
            if layers is not None:
                return layers
        except AttributeError:
            continue
    
    # If none work, return None and we'll handle it
    return None
