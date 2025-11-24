import torch
import numpy as np
from typing import Tuple, Literal, Optional

HookKind = Literal["forward", "pre"]

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
    # If none work, return None and we'll handle it
    return None

def get_activation_hook_target(model, layer_idx: int) -> Tuple[torch.nn.Module, HookKind, Optional[int]]:
    """
    Return (module_to_hook, hook_kind, neuron_dim_if_known).
    For Transformers, we hook *pre* c_proj to capture post-activation MLP features.
    For Mamba, we hook mixer/ssm (forward).
    Fallback: hook the whole layer (forward).
    """
    layers = get_model_layers(model)
    if layers is None or not (0 <= layer_idx < len(layers)):
        return model, "forward", None  # will be handled upstream

    layer = layers[layer_idx]

    # ---- MAMBA family: prefer mixer.ssm, else mixer ----
    if hasattr(layer, "mixer") and hasattr(layer.mixer, "ssm"):
        return layer.mixer.ssm, "forward", getattr(model.config, "hidden_size", None)
    if hasattr(layer, "mixer"):
        return layer.mixer, "forward", getattr(model.config, "hidden_size", None)

    # ---- TRANSFORMERS: take MLP pre-proj (post-activation) ----
    if hasattr(layer, "mlp"):
        mlp = layer.mlp
        if hasattr(mlp, "c_proj"):  # GPT-2 block
            # pre-hook on c_proj lets us see the input to c_proj (i.e., post-activation tensor)
            inner = getattr(model.config, "n_inner", None)
            if inner is None and hasattr(mlp, "c_fc") and hasattr(mlp.c_fc, "out_features"):
                inner = mlp.c_fc.out_features
            return mlp.c_proj, "pre", inner

    # ---- Fallback: layer output ----
    return layer, "forward", getattr(model.config, "hidden_size", None)
