# utils.py
import torch
import numpy as np
from typing import Tuple, Literal, Optional

def debug_model_structure(model, max_depth=3, current_depth=0, prefix=""):
    if current_depth >= max_depth:
        return
    for name, module in model.named_children():
        print(f"{prefix}{name}: {type(module).__name__}")
        if current_depth < max_depth - 1:
            debug_model_structure(module, max_depth, current_depth + 1, prefix + "  ")

def get_model_layers(model):
    """
    Return the list/ModuleList of top-level layers (Transformer blocks or Mamba blocks).
    Covers common HF layouts.
    """
    candidates = [
        lambda m: m.backbone.layers,      # Mamba HF ports
        lambda m: m.model.layers,         # LLaMA-style
        lambda m: m.layers,               # direct
        lambda m: m.transformer.h,        # GPT-2, GPT-J
        lambda m: m.transformer.layers,   # some variants
    ]
    for fn in candidates:
        try:
            layers = fn(model)
            if layers is not None:
                return layers
        except AttributeError:
            pass
    return None

HookKind = Literal["forward", "pre"]

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
