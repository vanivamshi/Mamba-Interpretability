# Run as: python3 7_ngram_new.py
"""
Simplified Extended N-gram Analysis for Mamba models
Based on "In-Context Language Learning: Architectures and Algorithms"

This is a simplified version that extends the original 4_ngram_analysis.py
to study head behaviors and n-gram circuits in Mamba models.

Patched: now saves raw per-layer head distributions into JSON for debugging.

CONFIGURATION:
- Set LAYERS_TO_ANALYZE = None to analyze all layers (slower but complete)
- Set LAYERS_TO_ANALYZE = [1, 2, 3] to analyze only specific layers (faster)
- Set LAYERS_TO_ANALYZE = [0, 1, 2] to analyze first 3 layers
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import json
import datetime
from utils import get_model_layers
from neuron_characterization import find_dead_neurons
import seaborn as sns
from typing import Dict, List, Tuple, Set, Optional
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import torch.nn.functional as F
import gc
import copy
warnings.filterwarnings('ignore')



# -------------------------
# Global threshold settings
# -------------------------
GPT2_ACTIVATION_THRESHOLD = 0.7
OTHER_MODELS_ACTIVATION_THRESHOLD = 0.6
MAMBA_ACTIVATION_THRESHOLD = 0.6

# Global defaults (tweak these)
MAX_SEQUENCE_LENGTH = 128
MAX_TEXTS_PER_LAYER = 100
MEMORY_CLEANUP_FREQUENCY = 50
ATTENTION_MASS_DEFAULT_THRESHOLD = 0.1

# Global variable for kernel interpolation results
kernel_interpolation_results = None


# Quick single-layer test snippet (run before full loop)
def test_gpt2_attribution(model, tokenizer, device=None):
    """
    Test snippet to verify gradient attribution and validation functions work correctly.
    Run this before the full analysis to check if both functions return nonzero numbers.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()
    
    texts = ["The cat sat on the mat.", "The quick brown fox jumps over the lazy dog."]
    inputs = tokenizer(texts[0], return_tensors="pt", truncation=True, max_length=64).to(device)
    
    # choose layer to test
    layer_idx = 0
    print(">>> Running gradient attribution test for GPT-2 layer", layer_idx)
    imp = compute_gradient_attribution_gpt2(model, tokenizer, inputs, layer_idx, debug=True)
    print("Gradient importance:", imp)
    
    print(">>> Running ablation validation test")
    val = validate_head_specialization_with_ablation_gpt2(model, tokenizer, texts, layer_idx, head_neurons=[0,1,2], device=device, debug=True)
    print("Validation result:", val)
    
    # Extra checks if still getting zeros
    print(">>> Extra diagnostic checks:")
    print("model.lm_head exists:", hasattr(model, "lm_head"))
    if hasattr(model, "lm_head"):
        print("lm_head dtype:", next(model.lm_head.parameters()).dtype)
    
    # Run sanity check
    print(">>> Running sanity check:")
    sanity_check_forward(model, tokenizer, texts, layer_idx, device)
    
    return imp, val


# -------------------------
# Model zoo
# -------------------------
models_to_analyze = {
    "Mamba-130M": "state-spaces/mamba-130m-hf",
     "Mamba-370M": "state-spaces/mamba-370m-hf",
     "Mamba-790M": "state-spaces/mamba-790m-hf",
     "Mamba-1.4B": "state-spaces/mamba-1.4b-hf",
     "GPT-2": "gpt2",
}

# -------------------------
# Analysis configuration
# -------------------------
# Set to None to analyze all layers, or specify a list like [1, 2, 3] for specific layers
# Different configurations for different model types
LAYERS_TO_ANALYZE = {
    "GPT-2": [0, 5, 10, 11],  # Analyze layers 0, 5, and 10 for GPT-2
    "Mamba-370M": [0, 8, 15, 23],  # Analyze layers 0, 8, and 15 for Mamba-1.4B
    "default": [0, 10]  # Default fallback
}

# Memory optimization settings
MAX_SEQUENCE_LENGTH = 128  # Use same as 4_ngram_analysis.py
BATCH_SIZE = 1  # Process one text at a time
MEMORY_CLEANUP_FREQUENCY = 5  # Less frequent cleanup like 4_ngram_analysis.py
MAX_TEXTS_PER_LAYER = 50  # Use more texts like 4_ngram_analysis.py
SAFE_SEQUENCE_LENGTH = 64  # Reasonable fallback

# Quick configuration examples:
# LAYERS_TO_ANALYZE = None        # All layers (slow but complete)
# LAYERS_TO_ANALYZE = [0, 5, 10]  # Specific layers (fast)

# Mamba models are enabled by default (same as 4_ngram_analysis.py and main.py)

def clear_memory():
    """Clear GPU memory and run garbage collection with error handling."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except RuntimeError as e:
        if "device-side assert" in str(e):
            print(f"⚠️  CUDA device-side assert detected, skipping cache clear: {e}")
        else:
            print(f"⚠️  CUDA error during cache clear: {e}")
    except Exception as e:
        print(f"⚠️  Unexpected error during cache clear: {e}")
    
    gc.collect()

def get_memory_usage():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        return allocated, reserved
    return 0, 0

def safe_tokenize(text, tokenizer, max_length=None):
    """Safely tokenize text with proper length handling."""
    if max_length is None:
        max_length = MAX_SEQUENCE_LENGTH
    
    # Simple approach like 4_ngram_analysis.py
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
    return inputs

def try_import_hidden_mamba_attn():
    """Try to import HiddenMambaAttn helper for computing hidden attention."""
    try:
        # Try different possible import paths
        try:
            from hidden_mamba_attn import compute_hidden_attention
            return compute_hidden_attention
        except ImportError:
            pass
        
        try:
            from mamba_attention import compute_hidden_attention
            return compute_hidden_attention
        except ImportError:
            pass
            
        # If you have the HiddenMambaAttn repo in your PYTHONPATH
        try:
            import sys
            sys.path.append('/path/to/HiddenMambaAttn')  # Update this path
            from hidden_mamba_attn import compute_hidden_attention
            return compute_hidden_attention
        except ImportError:
            pass
            
        return None
    except Exception as e:
        print(f"Warning: Could not import HiddenMambaAttn: {e}")
        return None

def get_transformer_attention_weights(model, tokenizer, texts, layer_idx, device):
    """Extract attention weights from Transformer models."""
    model.eval()
    model.to(device)  # Ensure model is on the correct device
    attention_weights = []
    
    def attention_hook(module, input, output):
        # For GPT-2 and most Transformer models, attention weights are in the output tuple
        if isinstance(output, tuple) and len(output) > 1:
            # output[1] typically contains attention weights
            attn_weights = output[1].detach().cpu()  # Shape: [batch, heads, seq, seq]
            attention_weights.append(attn_weights)
            print(f"    Captured attention weights: {attn_weights.shape}")
        else:
            print(f"    ⚠️  Unexpected output format: {type(output)}")
    
    # Get the attention layer (usually the first submodule in a Transformer block)
    layers = get_model_layers(model)
    layer = layers[layer_idx]
    
    # Find the attention module within the layer - try multiple approaches
    attention_module = None
    
    # Method 1: Look for attention in named modules
    for name, module in layer.named_modules():
        if 'attention' in name.lower() or 'attn' in name.lower():
            attention_module = module
            print(f"    Found attention module: {name}")
            break
    
    # Method 2: Try common attribute names (prioritize 'attn' for GPT-2)
    if attention_module is None:
        for attr_name in ['attn', 'attention', 'self_attention', 'self_attn']:
            if hasattr(layer, attr_name):
                attention_module = getattr(layer, attr_name)
                print(f"    Found attention module via attribute: {attr_name}")
                break
    
    # Method 3: Try first submodule (common pattern)
    if attention_module is None and len(list(layer.children())) > 0:
        first_child = list(layer.children())[0]
        if hasattr(first_child, 'attention') or hasattr(first_child, 'attn'):
            attention_module = first_child
            print(f"    Found attention module in first child")
    
    if attention_module is None:
        print(f"⚠️  Could not find attention module in layer {layer_idx}")
        print(f"    Available modules: {list(layer.named_modules())}")
        return None
    
    handle = attention_module.register_forward_hook(attention_hook)
    
    try:
        with torch.no_grad():
            for text in texts[:3]:  # Use subset for efficiency
                try:
                    inputs = safe_tokenize(text, tokenizer, MAX_SEQUENCE_LENGTH)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    # Enable attention output for GPT-2
                    outputs = model(**inputs, output_attentions=True)
                    if hasattr(outputs, 'attentions') and outputs.attentions:
                        # Check if layer_idx is within bounds
                        if layer_idx < len(outputs.attentions):
                            # Extract attention weights from model output
                            attn_weights = outputs.attentions[layer_idx].detach().cpu()
                            attention_weights.append(attn_weights)
                            print(f"    ✅ Extracted attention weights from model output: {attn_weights.shape}")
                            break
                        else:
                            print(f"    ⚠️  Layer index {layer_idx} out of bounds for attention weights (max: {len(outputs.attentions)-1})")
                            break
                    elif attention_weights:
                        break  # Use first successful forward pass
                except Exception as e:
                    print(f"⚠️  Forward pass failed: {e}")
                    continue
    finally:
        handle.remove()
    
    if attention_weights:
        print(f"    ✅ Extracted attention weights: {attention_weights[0].shape}")
        return attention_weights[0]
    else:
        print(f"    ⚠️  No attention weights captured")
        return None

def get_mamba_attention_weights(model, tokenizer, texts, layer_idx, device):
    """Extract hidden attention weights from Mamba models using MambaAttentionNeurons."""
    try:
        from attention_neurons import MambaAttentionNeurons
        
        # Try to use MambaAttentionNeurons for Mamba models
        mamba_analyzer = MambaAttentionNeurons(model, enable_attention_computation=True)
        
        # Use first text for attention extraction
        sample_text = texts[0] if texts else "Sample text for analysis"
        inputs = safe_tokenize(sample_text, tokenizer, MAX_SEQUENCE_LENGTH)
        
        # Extract attention vectors
        attention_data = mamba_analyzer.extract_attention_vectors(inputs["input_ids"], [layer_idx])
        
        if layer_idx in attention_data:
            layer_data = attention_data[layer_idx]
            
            # Try different methods to get attention matrix
            attention_matrix = None
            
            # Method 1: Use synthetic attention matrix if available
            if 'attention_matrix' in layer_data:
                attention_matrix = layer_data['attention_matrix']
                print(f"    ✅ Extracted synthetic Mamba attention matrix: {attention_matrix.shape}")
            # Method 2: Create from state transition matrix
            elif 'state_transition' in layer_data:
                A = layer_data['state_transition']
                # Create attention-like matrix from state transition
                if len(A.shape) == 2:
                    # For Mamba, create a proper attention matrix from state transition
                    seq_len = inputs["input_ids"].shape[1]
                    # Create attention matrix by expanding state transition
                    attention_matrix = A.unsqueeze(0).unsqueeze(0).expand(1, 1, seq_len, seq_len)
                else:
                    attention_matrix = A
                print(f"    ✅ Created attention matrix from state transition: {attention_matrix.shape}")
            # Method 3: Create from input projection
            elif 'input_projection' in layer_data:
                in_proj = layer_data['input_projection']
                # Create attention-like matrix from input projection
                seq_len = inputs["input_ids"].shape[1]
                # Use input projection to create attention weights
                attention_matrix = torch.randn(1, 1, seq_len, seq_len).to(device)  # Placeholder
                print(f"    ✅ Created placeholder attention matrix from input projection: {attention_matrix.shape}")
            
            if attention_matrix is not None:
                # Ensure proper dimensions for attention matrix
                if len(attention_matrix.shape) == 2:
                    # Add batch and head dimensions
                    attention_matrix = attention_matrix.unsqueeze(0).unsqueeze(0)
                elif len(attention_matrix.shape) == 3:
                    # Add head dimension
                    attention_matrix = attention_matrix.unsqueeze(1)
                
                return attention_matrix
            else:
                print(f"    ⚠️  No attention matrix could be created for layer {layer_idx}")
                return None
        else:
            print(f"    ⚠️  No attention data found for layer {layer_idx}")
            return None
            
    except Exception as e:
        print(f"    ⚠️  MambaAttentionNeurons failed: {e}")
        return None

def compute_attention_mass_on_last_n_tokens(attention_weights, n_tokens, position):
    """Compute attention mass on the last n_tokens that attend into 'position'."""
    if attention_weights is None:
        return 0.0

    # Normalize to [seq, seq] average over batch+heads when present
    if attention_weights.ndim == 4:  # [batch, heads, seq, seq]
        attn = attention_weights.mean(axis=(0, 1))  # -> [seq, seq]
    elif attention_weights.ndim == 3:  # [heads, seq, seq] or [batch, seq, seq]
        attn = attention_weights.mean(axis=0)
    elif attention_weights.ndim == 2:  # [seq, seq]
        attn = attention_weights
    else:
        return 0.0

    seq_len = attn.shape[0]
    if position >= seq_len:
        return 0.0

    start_pos = max(0, position - n_tokens + 1)
    # we want attention distribution from position -> previous tokens [start_pos .. position]
    # depending on matrix orientation, choose row or column consistently:
    # Here attn[position, j] is attention that token at 'position' pays to j (previous positions).
    return float(attn[position, start_pos:position+1].sum())


def collect_ngram_triggers_enhanced(model, tokenizer, texts, layer_idx, n_max=3, model_label=None):
    """
    Attention-aware n-gram trigger collection.
    
    This function now properly implements the mechanism described in the paper:
    - For Transformers: Uses real attention maps (attn_probs)
    - For Mamba: Uses HiddenMambaAttn to extract hidden attention weights
    - Only records n-gram triggers when attention mass on last n tokens exceeds threshold
    """
    # Determine activation threshold based on model
    if model_label == "GPT-2":
        activation_threshold = GPT2_ACTIVATION_THRESHOLD
    elif "Mamba" in model_label:
        activation_threshold = MAMBA_ACTIVATION_THRESHOLD
    else:
        activation_threshold = OTHER_MODELS_ACTIVATION_THRESHOLD
    
    # Attention mass threshold for considering n-gram detection
    if "Mamba" in model_label:
        ATTENTION_MASS_THRESHOLD = 0.005  # Lower threshold for Mamba models
    else:
        ATTENTION_MASS_THRESHOLD = 0.1
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    layers = get_model_layers(model)
    neuron_triggers = defaultdict(lambda: defaultdict(set))
    neuron_activations = defaultdict(list)
    
    # Get attention weights for this layer
    print(f"    Extracting attention weights for layer {layer_idx}...")
    if "GPT-2" in model_label or "Transformer" in model_label:
        attention_weights = get_transformer_attention_weights(model, tokenizer, texts[:3], layer_idx, device)
    elif "Mamba" in model_label:
        attention_weights = get_mamba_attention_weights(model, tokenizer, texts[:3], layer_idx, device)
    else:
        # Try to detect model type automatically
        if hasattr(model, 'layers') and len(model.layers) > 0:
            first_layer = model.layers[0]
            if hasattr(first_layer, 'mixer'):
                print(f"    Detected Mamba model, using Mamba attention extraction")
                attention_weights = get_mamba_attention_weights(model, tokenizer, texts[:3], layer_idx, device)
            else:
                print(f"    Detected Transformer model, using Transformer attention extraction")
                attention_weights = get_transformer_attention_weights(model, tokenizer, texts[:3], layer_idx, device)
        else:
            print(f"⚠️  Unknown model type: {model_label}, skipping attention-aware detection")
            attention_weights = None
    
    if attention_weights is not None:
        print(f"    ✅ Attention weights extracted: {attention_weights.shape}")
    else:
        print(f"    ⚠️  No attention weights available, using traditional detection")

    current_input_ids = None
    current_attention_mask = None

    def hook_fn(module, inp, out):
        nonlocal current_input_ids, current_attention_mask
        if isinstance(out, tuple):
            out = out[0]
        acts = out.detach()
        if current_input_ids is None:
            return
        
        # Check tensor dimensions and handle different shapes
        if acts.dim() < 2:
            print(f"⚠️  Unexpected tensor dimensions: {acts.shape}, skipping")
            return
            
        batch_size = acts.shape[0]
        if acts.dim() == 2:
            # Handle 2D tensors (batch_size, features) - this is common for some layers
            print(f"ℹ️  Detected 2D tensor {acts.shape}, treating as single sequence")
            seq_len = 1
            # Don't modify acts, handle indexing differently
        elif acts.dim() == 3:
            seq_len = acts.shape[1]
        else:
            print(f"⚠️  Unexpected tensor dimensions: {acts.shape}, skipping")
            return
        
        for b in range(batch_size):
            if current_attention_mask is not None and b < current_attention_mask.shape[0]:
                mask = current_attention_mask[b]
            else:
                mask = torch.ones(seq_len)
            
            for t in range(seq_len):
                if t < mask.shape[0] and mask[t] == 0:
                    continue
                    
                try:
                    tok_id = int(current_input_ids[b, t].cpu().item())
                    
                    # Handle different tensor dimensions correctly
                    if acts.dim() == 2:
                        # 2D tensor: acts[b] gives us the features for batch b
                        token_acts = acts[b].detach().cpu().numpy()
                    else:
                        # 3D tensor: acts[b, t] gives us the features for batch b, position t
                        token_acts = acts[b, t].detach().cpu().numpy()
                    
                    # Normalize activations per neuron before thresholding
                    token_acts = (token_acts - token_acts.mean()) / (token_acts.std() + 1e-8)
                    active_neurons = np.where(token_acts > activation_threshold)[0]
                except (IndexError, RuntimeError) as e:
                    print(f"⚠️  Tensor indexing error: {e}, skipping position ({b}, {t})")
                    continue
                
                # Store activation patterns for analysis
                for n in active_neurons:
                    neuron_activations[int(n)].append(token_acts[n])
                
                # Attention-aware n-gram detection
                if attention_weights is not None:
                    try:
                        # Check attention mass on last n tokens for each n-gram size
                        for nsize in range(1, n_max + 1):
                            if t >= nsize - 1:  # Need at least nsize-1 previous tokens
                                attention_mass = compute_attention_mass_on_last_n_tokens(
                                    attention_weights, nsize, t
                                )
                                
                                if attention_mass >= ATTENTION_MASS_THRESHOLD:
                                    if nsize == 1:
                                        # Unigram: current token
                                        for n in active_neurons:
                                            neuron_triggers[int(n)][1].add((tok_id,))
                                    else:
                                        # N-gram: last nsize tokens (robust bounds checking)
                                        start_idx = t - nsize + 1
                                        if start_idx < 0:
                                            continue
                                        # safety: ensure mask indices exist
                                        if current_attention_mask is not None and (start_idx < 0 or t >= current_attention_mask.shape[1]):
                                            continue
                                        # now safe to read tokens
                                        if start_idx >= 0 and t < seq_len and all(mask[start_idx + i] == 1 for i in range(nsize)):
                                            ngram_ids = current_input_ids[b, start_idx:t+1].cpu().numpy()
                                            ngram = tuple(int(x) for x in ngram_ids)
                                            for n in active_neurons:
                                                neuron_triggers[int(n)][nsize].add(ngram)
                    except Exception as e:
                        print(f"⚠️  Attention-aware detection failed: {e}, using fallback")
                        # Fallback to traditional detection
                        for n in active_neurons:
                            neuron_triggers[int(n)][1].add((tok_id,))
                        
                        for nsize in range(2, n_max + 1):
                            start_idx = t - nsize + 1
                            if start_idx < 0:
                                continue
                            # safety: ensure mask indices exist
                            if current_attention_mask is not None and (start_idx < 0 or t >= current_attention_mask.shape[1]):
                                continue
                            # now safe to read tokens
                            if start_idx >= 0 and t < seq_len and all(mask[start_idx + i] == 1 for i in range(nsize)):
                                ngram_ids = current_input_ids[b, start_idx:t+1].cpu().numpy()
                                ngram = tuple(int(x) for x in ngram_ids)
                                for n in active_neurons:
                                    neuron_triggers[int(n)][nsize].add(ngram)
                else:
                    # Fallback: traditional detection without attention awareness
                    for n in active_neurons:
                        neuron_triggers[int(n)][1].add((tok_id,))
                    
                    for nsize in range(2, n_max + 1):
                        start_idx = t - nsize + 1
                        if start_idx < 0:
                            continue
                        # safety: ensure mask indices exist
                        if current_attention_mask is not None and (start_idx < 0 or t >= current_attention_mask.shape[1]):
                            continue
                        # now safe to read tokens
                        if start_idx >= 0 and t < seq_len and all(mask[start_idx + i] == 1 for i in range(nsize)):
                            ngram_ids = current_input_ids[b, start_idx:t+1].cpu().numpy()
                            ngram = tuple(int(x) for x in ngram_ids)
                            for n in active_neurons:
                                neuron_triggers[int(n)][nsize].add(ngram)

    handle = layers[layer_idx].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            for i, text in enumerate(texts[:MAX_TEXTS_PER_LAYER]):
                # Clear memory periodically
                if i % MEMORY_CLEANUP_FREQUENCY == 0:
                    clear_memory()
                
                inputs = safe_tokenize(text, tokenizer, MAX_SEQUENCE_LENGTH)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                current_input_ids = inputs["input_ids"]
                current_attention_mask = inputs.get("attention_mask", None)
                _ = model(**inputs)
                
                # Clear intermediate tensors
                del inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        handle.remove()
    
    # Print statistics
    total_triggers = sum(len(triggers) for triggers in neuron_triggers.values())
    print(f"    📊 Collected {total_triggers} total triggers across all neurons")
    
    return dict(neuron_triggers), dict(neuron_activations)


def analyze_head_like_behavior(neuron_triggers, neuron_activations, layer_idx, model, 
                              tokenizer=None, texts=None, model_label=None, use_clustering=True):
    """
    Analyze head-like behavior by grouping neurons into "heads" and examining 
    their collective specialization patterns. This approximates the paper's 
    attention head analysis for Mamba models.
    
    Args:
        use_clustering: If True, use clustering-based grouping; if False, use original slicing
    """
    head_analysis = {
        'head_specializations': {},  # head_id -> dominant n-gram type
        'head_ngram_distributions': {},  # head_id -> {1: count, 2: count, 3: count}
        'head_specialization_scores': {},  # head_id -> specialization score
        'head_activation_patterns': {},  # head_id -> activation stats
        'head_compositions': {},  # head_id -> list of neuron indices
        'head_validations': {}  # head_id -> validation results
    }
    
    # Group neurons into "heads" using either clustering or slicing
    if use_clustering:
        heads = group_neurons_into_heads_clustering(neuron_triggers, model, layer_idx)
    else:
        heads = group_neurons_into_heads(neuron_triggers, model, layer_idx)
    
    # Analyze each head's collective behavior
    for head_id, neuron_indices in heads.items():
        head_analysis['head_compositions'][head_id] = neuron_indices
        
        # Collect all triggers for this head
        head_ngram_counts = defaultdict(int)
        head_activations = []
        
        for neuron_idx in neuron_indices:
            if neuron_idx in neuron_triggers:
                # Aggregate n-gram triggers for this head
                for ngram_size, ngrams in neuron_triggers[neuron_idx].items():
                    head_ngram_counts[ngram_size] += len(ngrams)
            
            if neuron_idx in neuron_activations:
                head_activations.extend(neuron_activations[neuron_idx])
        
        # Determine head specialization
        if head_ngram_counts:
            head_analysis['head_ngram_distributions'][head_id] = dict(head_ngram_counts)
            
            # Find dominant n-gram type for this head
            dominant_ngram_size = max(head_ngram_counts.keys(), key=lambda k: head_ngram_counts[k])
            head_analysis['head_specializations'][head_id] = dominant_ngram_size
            
            # Calculate specialization score
            total_triggers = sum(head_ngram_counts.values())
            if total_triggers > 0:
                specialization_score = head_ngram_counts[dominant_ngram_size] / total_triggers
                head_analysis['head_specialization_scores'][head_id] = specialization_score
        
        # Analyze collective activation patterns
        if len(head_activations) > 5:
            activations_array = np.array(head_activations)
            head_analysis['head_activation_patterns'][head_id] = {
                'mean': float(np.mean(activations_array)),
                'std': float(np.std(activations_array)),
                'max': float(np.max(activations_array)),
                'min': float(np.min(activations_array)),
                'neuron_count': len(neuron_indices)
            }
        
        # Validate head specialization with ablation (if data available)
        if (tokenizer is not None and texts is not None and 
            head_analysis['head_specializations'].get(head_id) is not None):
            try:
                # Use GPT-2 specific validation for GPT-2 models
                if model_label and "gpt" in model_label.lower():
                    validation_result = validate_head_specialization_with_ablation_gpt2(
                        model, tokenizer, texts[:5], layer_idx, neuron_indices[:3], 
                        device=next(model.parameters()).device, debug=False
                    )
                else:
                    validation_result = validate_head_specialization_with_ablation(
                        model, tokenizer, texts, layer_idx, neuron_indices,
                        head_analysis['head_specializations'][head_id], model_label
                    )
                head_analysis['head_validations'][head_id] = validation_result
            except Exception as e:
                print(f"Warning: Validation failed for {head_id}: {e}")
                head_analysis['head_validations'][head_id] = {
                    "validation_score": 0.0, 
                    "coverage_impact": 0.0,
                    "error": str(e)
                }
    
    return head_analysis


def group_neurons_into_heads(neuron_triggers, model, layer_idx):
    """
    Group neurons into head-like components by mapping actual neuron indices
    to a contiguous index space and slicing that space.
    """
    if not neuron_triggers:
        return {}

    # Get sorted neuron indices (real neuron ids)
    neuron_indices_sorted = sorted(int(n) for n in neuron_triggers.keys())
    N_active = len(neuron_indices_sorted)

    # Create mapping from real neuron id -> contiguous index (0..N_active-1)
    real_to_contig = {real: i for i, real in enumerate(neuron_indices_sorted)}
    contig_to_real = {i: real for real, i in real_to_contig.items()}

    # Determine number of heads from model config (as before)
    if hasattr(model.config, 'num_attention_heads'):
        num_heads = model.config.num_attention_heads
    elif hasattr(model.config, 'hidden_size'):
        hidden_size = model.config.hidden_size
        num_heads = min(8, max(4, hidden_size // 64))
    else:
        num_heads = 8

    num_heads = min(num_heads, max(1, N_active))  # don't create more heads than active neurons
    neurons_per_head = max(1, N_active // num_heads)

    heads = {}
    for head_id in range(num_heads):
        start = head_id * neurons_per_head
        end = (start + neurons_per_head) if (head_id < num_heads - 1) else N_active
        contig_indices = list(range(start, end))
        # map back to real neuron ids
        head_neurons_real = [contig_to_real[c] for c in contig_indices if c in contig_to_real]
        if head_neurons_real:
            heads[f"head_{head_id}"] = head_neurons_real

    return heads


def create_trigger_distribution_features(neuron_triggers, ngram_sizes=[1,2,3]):
    features = []
    neuron_indices = sorted(int(k) for k in neuron_triggers.keys())  # contiguous ordering by active id

    for neuron_idx in neuron_indices:
        triggers = neuron_triggers[neuron_idx]
        feature_vector = []
        total_triggers = 0
        for n in ngram_sizes:
            c = len(triggers.get(n, set()))
            feature_vector.append(c)
            total_triggers += c
        feature_vector.append(total_triggers)
        if total_triggers > 0:
            for n in ngram_sizes:
                feature_vector.append(len(triggers.get(n, set())) / float(total_triggers))
        else:
            feature_vector.extend([0.0]*len(ngram_sizes))
        features.append(feature_vector)

    return np.array(features, dtype=float), neuron_indices


def group_neurons_into_heads_clustering(neuron_triggers, model, layer_idx, n_clusters=None):
    """
    Group neurons into "head-like" components using clustering based on trigger distributions.
    This is more robust than arbitrary slicing.
    """
    if not neuron_triggers:
        return {}
    
    # Create feature vectors from trigger distributions
    features, neuron_indices = create_trigger_distribution_features(neuron_triggers)
    
    if len(features) < 2:
        # Not enough neurons to cluster
        return {"head_0": neuron_indices}
    
    # Determine number of clusters
    if n_clusters is None:
        # Use a reasonable number based on model architecture
        if hasattr(model.config, 'num_attention_heads'):
            n_clusters = model.config.num_attention_heads
        elif hasattr(model.config, 'hidden_size'):
            hidden_size = model.config.hidden_size
            # For Mamba models, use more clusters for better granularity
            if "mamba" in str(type(model)).lower():
                n_clusters = min(16, max(8, hidden_size // 32))  # More clusters for Mamba
            else:
                n_clusters = min(8, max(4, hidden_size // 64))
        else:
            n_clusters = min(8, max(2, len(neuron_indices) // 10))
    
    # Ensure we don't have more clusters than neurons
    n_clusters = min(n_clusters, len(neuron_indices))
    
    # Standardize features for clustering
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    # Group neurons by cluster
    heads = {}
    for cluster_id in range(n_clusters):
        cluster_neurons = [neuron_indices[i] for i, lbl in enumerate(cluster_labels) if lbl == cluster_id]
        if cluster_neurons:
            heads[f"head_{cluster_id}"] = cluster_neurons
    
    return heads


def extract_attention_from_outputs(outputs):
    """
    Safely obtain attention tensor from model outputs.
    Returns None or attn ndarray shaped [B, H, T, T].
    """
    attn = None
    # HF models sometimes provide .attentions or inside tuple
    if hasattr(outputs, "attentions") and outputs.attentions is not None:
        attn = outputs.attentions
    else:
        # some models return tuple(list) where attentions at position idx; try common locations
        try:
            # outputs may be a ModelOutput or tuple: (logits, hidden_states, attentions)
            if isinstance(outputs, (tuple, list)) and len(outputs) >= 3:
                candidate = outputs[2]
                if candidate is not None:
                    attn = candidate
        except Exception:
            attn = None

    if attn is None:
        return None

    # Convert to tensor and ensure proper shape
    attn_t = attn if isinstance(attn, torch.Tensor) else torch.stack(attn) if isinstance(attn, list) else None
    if attn_t is None:
        try:
            attn_t = torch.tensor(attn)
        except Exception:
            return None

    # Now attn_t should be [B, H, T, T] or [H, T, T] or [T, T]
    if attn_t.dim() == 4:
        return attn_t.detach().cpu().numpy()
    elif attn_t.dim() == 3:
        # treat as [B, T, T] or [H, T, T] -> expand to [B, H, T, T] conservatively
        if attn_t.shape[0] <= 8:  # probably [H, T, T]
            return attn_t.unsqueeze(0).detach().cpu().numpy()
        else:
            return attn_t.unsqueeze(1).detach().cpu().numpy()
    elif attn_t.dim() == 2:
        return attn_t.unsqueeze(0).unsqueeze(0).detach().cpu().numpy()
    else:
        return None


def percent_improvement(base, new):
    """Safely compute percentage improvement with division by zero protection."""
    if base is None:
        return None
    if base == 0 or abs(base) < 1e-12:
        # define as None or special label; avoid infinite/nan
        return None
    return 100.0 * (new - base) / base


def safe_mean(lst):
    """Safely compute mean of a list, returning None if empty."""
    return float(np.mean(lst)) if len(lst) else None


def safe_gradient_importance(model, inputs, layer_output):
    """
    Compute mean absolute gradient importance for `layer_output` wrt the final token logit.
    Handles variable dims safely.
    """
    model.zero_grad()
    # Ensure grads enabled
    with torch.enable_grad():
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits  # [B, T, V]
        # choose true token positions: use next-token (shifted) if available else last token
        B, T = logits.shape[0], logits.shape[1]
        # pick logit for last position and sum over batch
        target_logit = logits[:, -1, :].max(dim=-1)[0].sum()
        # find the hidden corresponding to the layer from outputs if needed
        # layer_output must come from outputs.hidden_states[layer_idx] in caller
        grads = torch.autograd.grad(target_logit, layer_output, retain_graph=False, allow_unused=True)[0]
        if grads is None:
            return 0.0
        # Reduce safely
        while grads.dim() > 1:
            grads = grads.mean(dim=-1)
        return float(grads.abs().mean().cpu().item())


def sanity_check_forward(model, tokenizer, texts, layer_idx, device):
    inputs = tokenizer(texts[0], return_tensors="pt", truncation=True, max_length=64).to(device)
    out = model(**inputs, output_hidden_states=True, return_dict=True)
    print("sanity: logits shape", out.logits.shape)
    if out.hidden_states is None:
        print("sanity: hidden_states missing")
    else:
        print("sanity: #hidden_states", len(out.hidden_states))
        if len(out.hidden_states) > layer_idx + 1:
            print("sanity: selected block hidden shape", out.hidden_states[layer_idx+1].shape)
        else:
            print("sanity: selected hidden shape", out.hidden_states[layer_idx].shape)


def compute_gradient_attribution_gpt2(model, tokenizer, inputs, layer_idx, debug=False):
    """
    Robust gradient attribution for HF GPT-2 style models.
    - uses hidden_states[layer_idx + 1] (post-block output)
    - applies ln_f if present
    - decodes via lm_head so scalar depends on the hidden
    Returns float importance (mean abs grad) or 0.0 on error.
    """
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    model.to(device)
    model.eval()

    with torch.enable_grad():
        # Forward to get hidden states
        outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            if debug: print("DEBUG: no hidden_states returned")
            return 0.0

        # Use block output: HF returns embeddings at index 0, block outputs at 1..N
        use_idx = layer_idx + 1 if len(hidden_states) > layer_idx + 1 else layer_idx
        block_hidden = hidden_states[use_idx]  # shape [B, T, H]
        if debug:
            print("DEBUG block_hidden.shape:", getattr(block_hidden, "shape", None),
                  "len(hidden_states):", len(hidden_states), "use_idx:", use_idx)

        # Force float32 and require grad on this hidden tensor
        hidden = block_hidden.to(dtype=torch.float32, device=device).detach().requires_grad_(True)

        # Apply final layer norm (ln_f) if present before lm_head
        lm_input = hidden
        if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
            try:
                lm_input = model.transformer.ln_f(hidden)
            except Exception:
                lm_input = hidden

        # Decode via lm_head (most GPT2LMHeadModel variants have lm_head)
        if hasattr(model, "lm_head"):
            lm_logits = model.lm_head(lm_input)  # [B, T, V]
        else:
            # fallback: compute logits via a fresh forward (less ideal)
            re_out = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
            lm_logits = re_out.logits

        if debug: print("DEBUG lm_logits.shape:", getattr(lm_logits, "shape", None))

        # Choose scalar: prefer true last-token logits if input_ids present, else max logit
        B, T, V = lm_logits.shape
        if "input_ids" in inputs and inputs["input_ids"].shape[1] >= 1:
            last_token_ids = inputs["input_ids"][:, -1].to(device)
            # guard gather shapes
            if last_token_ids.dim() == 1 and last_token_ids.shape[0] == B:
                target_logits = lm_logits[:, -1, :].gather(1, last_token_ids.unsqueeze(1)).squeeze(1)  # [B]
                scalar = target_logits.sum()
            else:
                scalar = lm_logits[:, -1, :].max(dim=-1)[0].sum()
        else:
            scalar = lm_logits[:, -1, :].max(dim=-1)[0].sum()

        # Backprop from scalar to the chosen hidden
        grads = torch.autograd.grad(scalar, hidden, retain_graph=False, allow_unused=True)[0]
        if grads is None:
            if debug: print("DEBUG grads is None")
            return 0.0

        importance = float(grads.abs().mean().cpu().item())
        if debug: print("DEBUG importance:", importance, "grads.shape:", grads.shape)
        return importance

    return 0.0



def validate_head_specialization_with_ablation_gpt2(model, tokenizer, texts, layer_idx, head_neurons, device=None, debug=False):
    """
    Ablate (zero) the neurons in `head_neurons` at the chosen block hidden layer
    and compute a validation score (KL divergence) between baseline and ablated logits.
    Returns dict with validation_score and diagnostics.
    """
    device = device or next(model.parameters()).device
    model.to(device)
    model.eval()

    baseline_logits = []
    ablated_logits = []
    valid_neurons_total = 0
    max_texts = min(len(texts), 10)

    for text in texts[:max_texts]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)

        # Baseline logits
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
            hs = out.hidden_states
            if hs is None:
                return {"validation_score": 0.0, "error": "no hidden_states"}
            use_idx = layer_idx + 1 if len(hs) > layer_idx + 1 else layer_idx
            hidden_base = hs[use_idx]
            lm_input = hidden_base
            if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
                lm_input = model.transformer.ln_f(hidden_base)
            logits_base = model.lm_head(lm_input) if hasattr(model, "lm_head") else out.logits
            baseline_logits.append(logits_base.detach().cpu())

        # Ablated logits: copy hidden, zero selected neurons, decode
        with torch.no_grad():
            out2 = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
            hs2 = out2.hidden_states
            use_idx2 = layer_idx + 1 if len(hs2) > layer_idx + 1 else layer_idx
            h2 = hs2[use_idx2].detach().clone()
            max_idx = h2.shape[-1] - 1
            valid_neurons = [n for n in head_neurons if 0 <= n <= max_idx]
            valid_neurons_total += len(valid_neurons)
            if debug:
                print(f"DEBUG: ablation valid_neurons count: {len(valid_neurons)} (requested {len(head_neurons)})")
            if valid_neurons:
                h2[..., valid_neurons] = 0.0
            lm_input_ab = h2
            if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
                lm_input_ab = model.transformer.ln_f(h2)
            logits_ab = model.lm_head(lm_input_ab) if hasattr(model, "lm_head") else out2.logits
            ablated_logits.append(logits_ab.detach().cpu())

    if not baseline_logits:
        return {"validation_score": 0.0, "error": "no baseline logits collected"}

    # Find the maximum sequence length to pad all tensors to the same size
    max_seq_len = max(logits.shape[1] for logits in baseline_logits)
    
    # Pad all tensors to the same sequence length
    padded_baseline = []
    padded_ablated = []
    
    for i, (base_logits, ablated_logits_item) in enumerate(zip(baseline_logits, ablated_logits)):
        seq_len = base_logits.shape[1]
        if seq_len < max_seq_len:
            # Pad with zeros
            pad_size = max_seq_len - seq_len
            base_padded = torch.cat([base_logits, torch.zeros(base_logits.shape[0], pad_size, base_logits.shape[2])], dim=1)
            ablated_padded = torch.cat([ablated_logits_item, torch.zeros(ablated_logits_item.shape[0], pad_size, ablated_logits_item.shape[2])], dim=1)
        else:
            base_padded = base_logits
            ablated_padded = ablated_logits_item
        
        padded_baseline.append(base_padded)
        padded_ablated.append(ablated_padded)
    
    baseline_all = torch.cat(padded_baseline, dim=0)  # [N, T, V]
    ablated_all = torch.cat(padded_ablated, dim=0)

    # compute average KL(baseline || ablated) safely
    try:
        p_log = F.log_softmax(baseline_all, dim=-1)
        q_log = F.log_softmax(ablated_all, dim=-1)
        p = p_log.exp()
        kl_per_pos = (p * (p_log - q_log)).sum(dim=-1)  # [N, T]
        kl = float(kl_per_pos.mean().item())
    except Exception:
        kl = float(torch.mean(torch.abs(baseline_all - ablated_all)).item())

    return {
        "validation_score": kl,
        "baseline_examples": baseline_all.shape[0],
        "ablation_examples": ablated_all.shape[0],
        "valid_neurons_used_total": valid_neurons_total
    }


def compute_validation_score(baseline_logits, ablated_logits):
    """Compute cosine similarity between token logits for more stable validation."""
    import torch.nn.functional as F
    base = F.normalize(baseline_logits.flatten(1), dim=-1)
    ablt = F.normalize(ablated_logits.flatten(1), dim=-1)
    return (base * ablt).sum(-1).mean().item()


def validate_head_specialization_with_ablation(
    model, tokenizer, texts, layer_idx, head_neurons, head_specialization, model_label=None
):
    """
    Compute KL divergence between logits with vs. without this head.
    """
    import torch.nn.functional as F
    model.eval()
    device = next(model.parameters()).device

    layers = get_model_layers(model)

    def ablation_hook(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        out = out.clone()
        if head_neurons and out.dim() >= 3:
            # Ensure we have the right dimensions [batch, seq, hidden]
            if out.shape[-1] > max(head_neurons):
                try:
                    out[:, :, head_neurons] = 0
                except Exception:
                    # If 3D indexing fails, try 2D indexing
                    if out.dim() == 2 and out.shape[-1] > max(head_neurons):
                        out[:, head_neurons] = 0
        return out

    baseline_logits, ablated_logits = [], []

    # Baseline
    with torch.no_grad():
        for text in texts[:10]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
            baseline_logits.append(model(**inputs).logits.detach())

    # Ablated
    handle = layers[layer_idx].register_forward_hook(lambda m, i, o: ablation_hook(m, i, o))
    with torch.no_grad():
        for text in texts[:10]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
            ablated_logits.append(model(**inputs).logits.detach())
    handle.remove()

    # KL divergence
    baseline = torch.cat(baseline_logits, dim=0)
    ablated = torch.cat(ablated_logits, dim=0)
    # Use cosine similarity instead of KL divergence for better numerical stability
    validation_score = compute_validation_score(baseline, ablated)

    return {"validation_score": validation_score, "coverage_impact": validation_score}


def validate_head_specialization_with_ablation_old(model, tokenizer, texts, layer_idx, 
                                             head_neurons, head_specialization, 
                                             model_label=None):
    """
    Validate head specialization by measuring token coverage impact when ablating the head.
    This is closer to the paper's ablation logic.
    """
    if not head_neurons:
        return {"validation_score": 0.0, "coverage_impact": 0.0}
    
    # Determine activation threshold
    if model_label == "GPT-2":
        activation_threshold = GPT2_ACTIVATION_THRESHOLD
    elif "Mamba" in model_label:
        activation_threshold = MAMBA_ACTIVATION_THRESHOLD
    else:
        activation_threshold = OTHER_MODELS_ACTIVATION_THRESHOLD
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    layers = get_model_layers(model)
    
    # Collect baseline token coverage (without ablation)
    baseline_coverage = set()
    ablation_coverage = set()
    
    def baseline_hook(module, inp, out):
        nonlocal baseline_coverage
        if isinstance(out, tuple):
            out = out[0]
        acts = out.detach()
        for b in range(acts.shape[0]):
            seq_len = acts.shape[1]
            for t in range(seq_len):
                token_acts = acts[b, t].cpu().numpy()
                # Normalize activations per neuron before thresholding
                token_acts = (token_acts - token_acts.mean()) / (token_acts.std() + 1e-8)
                active_neurons = np.where(token_acts > activation_threshold)[0]
                baseline_coverage.update(active_neurons)
    
    def ablation_hook(module, inp, out):
        nonlocal ablation_coverage
        if isinstance(out, tuple):
            out = out[0]
        acts = out.detach()
        for b in range(acts.shape[0]):
            seq_len = acts.shape[1]
            for t in range(seq_len):
                token_acts = acts[b, t].cpu().numpy()
                # Normalize activations per neuron before thresholding
                token_acts = (token_acts - token_acts.mean()) / (token_acts.std() + 1e-8)
                # Ablate the head neurons by setting their activations to 0
                token_acts_ablated = token_acts.copy()
                for neuron_idx in head_neurons:
                    if neuron_idx < len(token_acts_ablated):
                        token_acts_ablated[neuron_idx] = 0.0
                
                active_neurons = np.where(token_acts_ablated > activation_threshold)[0]
                ablation_coverage.update(active_neurons)
    
    # Measure baseline coverage
    handle = layers[layer_idx].register_forward_hook(baseline_hook)
    with torch.no_grad():
        for text in texts[:20]:  # Use subset for efficiency
            inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                             max_length=128, padding=True).to(device)
            _ = model(**inputs)
    handle.remove()
    
    # Measure ablation coverage
    handle = layers[layer_idx].register_forward_hook(ablation_hook)
    with torch.no_grad():
        for text in texts[:20]:  # Use subset for efficiency
            inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                             max_length=128, padding=True).to(device)
            _ = model(**inputs)
    handle.remove()
    
    # Calculate coverage impact
    baseline_size = len(baseline_coverage)
    ablation_size = len(ablation_coverage)
    
    if baseline_size > 0:
        coverage_impact = (baseline_size - ablation_size) / baseline_size
    else:
        coverage_impact = 0.0
    
    # Calculate validation score based on specialization alignment
    # Higher impact for heads specialized in the expected n-gram size
    validation_score = coverage_impact
    
    return {
        "validation_score": validation_score,
        "coverage_impact": coverage_impact,
        "baseline_coverage": baseline_size,
        "ablation_coverage": ablation_size,
        "head_size": len(head_neurons)
    }


def analyze_ngram_clustering_patterns(all_layer_results, model_label):
    """
    Analyze n-gram clustering patterns across layers using head-level analysis.
    This addresses the paper's finding about "later clustering" in larger models.
    Now includes validation results and comparison between clustering vs slicing methods.
    """
    clustering_analysis = {
        'layer_patterns': {},
        'head_distributions': {},
        'clustering_metrics': {},
        'validation_metrics': {},
        'method_comparison': {}
    }
    
    # Analyze patterns across layers
    for layer_idx, layer_results in all_layer_results.items():
        if 'head_analysis' in layer_results:
            head_analysis = layer_results['head_analysis']
            
            # Count heads by specialization type
            layer_patterns = defaultdict(int)
            head_distributions = {}
            validation_scores = []
            
            for head_id, specialization in head_analysis['head_specializations'].items():
                layer_patterns[specialization] += 1
                
                # Store detailed head information
                head_info = {
                    'specialization': specialization,
                    'specialization_score': head_analysis['head_specialization_scores'].get(head_id, 0),
                    'neuron_count': len(head_analysis['head_compositions'].get(head_id, [])),
                    'ngram_distribution': head_analysis['head_ngram_distributions'].get(head_id, {})
                }
                
                # Add validation results if available
                if head_id in head_analysis['head_validations']:
                    validation_result = head_analysis['head_validations'][head_id]
                    head_info['validation_score'] = validation_result.get('validation_score', 0.0)
                    head_info['coverage_impact'] = validation_result.get('coverage_impact', 0.0)
                    validation_scores.append(validation_result.get('validation_score', 0.0))
                
                head_distributions[head_id] = head_info
            
            clustering_analysis['layer_patterns'][layer_idx] = dict(layer_patterns)
            clustering_analysis['head_distributions'][layer_idx] = head_distributions
            
            # Store validation metrics for this layer
            if validation_scores:
                clustering_analysis['validation_metrics'][layer_idx] = {
                    'mean_validation_score': safe_mean(validation_scores),
                    'std_validation_score': np.std(validation_scores) if len(validation_scores) > 1 else 0.0,
                    'max_validation_score': np.max(validation_scores),
                    'min_validation_score': np.min(validation_scores),
                    'validated_heads': len(validation_scores)
                }
        
        # Compare clustering vs slicing methods if both are available
        if ('head_analysis' in layer_results and 
            'head_analysis_slicing' in layer_results):
            
            clustering_heads = layer_results['head_analysis']
            slicing_heads = layer_results['head_analysis_slicing']
            
            # Compare specialization distributions
            clustering_specializations = defaultdict(int)
            slicing_specializations = defaultdict(int)
            
            for head_id, spec in clustering_heads['head_specializations'].items():
                clustering_specializations[spec] += 1
            
            for head_id, spec in slicing_heads['head_specializations'].items():
                slicing_specializations[spec] += 1
            
            # Compare validation scores
            clustering_validation_scores = []
            slicing_validation_scores = []
            
            for head_id, validation in clustering_heads['head_validations'].items():
                clustering_validation_scores.append(validation.get('validation_score', 0.0))
            
            for head_id, validation in slicing_heads['head_validations'].items():
                slicing_validation_scores.append(validation.get('validation_score', 0.0))
            
            clustering_analysis['method_comparison'][layer_idx] = {
                'clustering_specializations': dict(clustering_specializations),
                'slicing_specializations': dict(slicing_specializations),
                'clustering_validation_mean': safe_mean(clustering_validation_scores),
                'slicing_validation_mean': safe_mean(slicing_validation_scores),
                'clustering_head_count': len(clustering_heads['head_specializations']),
                'slicing_head_count': len(slicing_heads['head_specializations'])
            }
    
    # Calculate clustering metrics
    layers = sorted(clustering_analysis['layer_patterns'].keys())
    ngram_sizes = [1, 2, 3]
    
    for ngram_size in ngram_sizes:
        head_counts = []
        for layer_idx in layers:
            count = clustering_analysis['layer_patterns'][layer_idx].get(ngram_size, 0)
            head_counts.append(count)
        
        # Calculate clustering metrics
        if len(head_counts) > 1:
            # Early vs late clustering
            early_layers = layers[:len(layers)//2]
            late_layers = layers[len(layers)//2:]
            
            early_count = sum(clustering_analysis['layer_patterns'][l].get(ngram_size, 0) for l in early_layers)
            late_count = sum(clustering_analysis['layer_patterns'][l].get(ngram_size, 0) for l in late_layers)
            
            clustering_analysis['clustering_metrics'][ngram_size] = {
                'early_count': early_count,
                'late_count': late_count,
                'early_late_ratio': early_count / late_count if late_count > 0 else float('inf'),
                'total_heads': sum(head_counts),
                'head_counts_per_layer': head_counts
            }
    
    return clustering_analysis


# -------------------------
# New: Save raw distributions
# -------------------------
def save_raw_head_distributions(all_results, save_dir="logs"):
    """
    Save raw per-layer head distributions to JSON for debugging.
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_dir, f"raw_head_distributions_{timestamp}.json")

    serializable = {}
    for model_label, model_results in all_results.items():
        serializable[model_label] = {}
        for layer_idx, layer_results in model_results.items():
            if isinstance(layer_idx, int) and "head_analysis" in layer_results:
                head_analysis = layer_results["head_analysis"]
                serializable[model_label][str(layer_idx)] = {
                    "head_specializations": head_analysis.get("head_specializations", {}),
                    "head_ngram_distributions": head_analysis.get("head_ngram_distributions", {}),
                    "head_specialization_scores": head_analysis.get("head_specialization_scores", {}),
                    "head_activation_patterns": head_analysis.get("head_activation_patterns", {}),
                }

    with open(log_file, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"✅ Raw head distributions saved to {log_file}")


def test_state_kernel_causality(model, tokenizer, texts, layer_indices=[0, 4, 8]):
    """
    Replace Mamba state kernels with random convolutions to test causal role of recurrence bias.
    """
    print("\n🧠 Running causal test: State-Kernel Replacement")

    # Step 1: Extract SSM layers
    ssm_layers = [m for m in model.modules() if 'ssm' in m.__class__.__name__.lower() or 'mamba' in m.__class__.__name__.lower()]
    if not ssm_layers:
        print("⚠️ No Mamba-style SSM layers found. Skipping.")
        return None

    # Step 2: Store original kernels
    original_kernels = [copy.deepcopy(getattr(layer, 'A', None)) for layer in ssm_layers]

    # Step 3: Replace kernels with random convolution-like weights
    for i, layer in enumerate(ssm_layers):
        A = getattr(layer, 'A', None)
        if A is not None:
            with torch.no_grad():
                rand_A = torch.randn_like(A) * 0.01  # small Gaussian random kernel
                layer.A.copy_(rand_A)
    print(f"🔄 Replaced {len(ssm_layers)} SSM kernels with random convolutions.")

    # Step 4: Run receptive field probe (same metric as before)
    def compute_rf_mass(model, layer_idx, tokenizer, texts):
        from torch.nn import functional as F
        model.eval()
        device = next(model.parameters()).device
        mass_1, mass_2, mass_3 = [], [], []
        with torch.no_grad():
            for text in texts[:10]:
                inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=64).to(device)
                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[layer_idx][0]  # [seq, dim]
                # Approximate receptive field by correlation of token offsets
                for n in [1, 2, 3]:
                    shifted = torch.roll(hidden, shifts=n, dims=0)
                    mass = torch.mean((hidden * shifted).abs()).item()
                    if n == 1: mass_1.append(mass)
                    elif n == 2: mass_2.append(mass)
                    else: mass_3.append(mass)
        return np.mean(mass_1), np.mean(mass_2), np.mean(mass_3)

    print("📊 Measuring receptive field mass before vs after kernel replacement...")

    results = {}
    for li in layer_indices:
        m1, m2, m3 = compute_rf_mass(model, li, tokenizer, texts)
        results[li] = {"1gram_mass": m1, "2gram_mass": m2, "3gram_mass": m3}
        print(f"  Layer {li}: 1g={m1:.4f}, 2g={m2:.4f}, 3g={m3:.4f}")

    # Step 5: Restore original kernels
    for layer, A_orig in zip(ssm_layers, original_kernels):
        if getattr(layer, 'A', None) is not None and A_orig is not None:
            layer.A.copy_(A_orig)
    print("✅ Restored original kernels.")

    return results


def interpolate_state_kernel_effect(model, tokenizer, texts,
                                    layer_idx=4, lambdas=np.linspace(0,1,6)):
    """
    Gradually interpolate between original and random Mamba state kernels
    and measure receptive-field mass at each interpolation point.
    """

    print("\n🧪 Running kernel interpolation experiment...")

    # 1️⃣ Locate Mamba/SSM layers
    ssm_layers = [m for m in model.modules()
                  if 'ssm' in m.__class__.__name__.lower()
                  or 'mamba' in m.__class__.__name__.lower()]
    if not ssm_layers:
        print("⚠️ No SSM layers found.")
        return None

    # 2️⃣ Store originals and make random versions safely
    original_As, rand_As = [], []
    for layer in ssm_layers:
        A = None
        for attr_name in ['A', 'A_log', 'state_A', 'kernel_A']:
            if hasattr(layer, attr_name):
                A = getattr(layer, attr_name)
                break
        if A is not None and isinstance(A, torch.Tensor):
            original_As.append(A.detach().clone())
            rand_As.append(torch.randn_like(A) * 0.01)

    if not original_As:
        print("⚠️ No valid state kernels (A-like matrices) found.")
        return None

    # 3️⃣ Helper to compute receptive-field mass
    def compute_rf_mass(model, layer_idx, tokenizer, texts):
        device = next(model.parameters()).device
        model.eval()
        mass1, mass2, mass3 = [], [], []
        with torch.no_grad():
            for text in texts[:10]:
                inputs = tokenizer(text, return_tensors='pt',
                                   truncation=True, max_length=64).to(device)
                out = model(**inputs, output_hidden_states=True)
                hidden = out.hidden_states[layer_idx][0]
                for n in [1,2,3]:
                    shifted = torch.roll(hidden, shifts=n, dims=0)
                    m = torch.mean((hidden * shifted).abs()).item()
                    (mass1 if n==1 else mass2 if n==2 else mass3).append(m)
        return np.mean(mass1), np.mean(mass2), np.mean(mass3)

    # 4️⃣ Sweep interpolation strengths
    results = []
    for lam in lambdas:
        # Apply interpolation to all found kernels
        for layer in ssm_layers:
            A = None
            attr_name = None
            for name in ['A', 'A_log', 'state_A', 'kernel_A']:
                if hasattr(layer, name):
                    A = getattr(layer, name)
                    attr_name = name
                    break
            
            if A is not None and isinstance(A, torch.Tensor):
                # Find corresponding original and random tensors
                for orig_A, rand_A in zip(original_As, rand_As):
                    if orig_A.shape == A.shape:
                        with torch.no_grad():
                            blended = (1-lam)*rand_A + lam*orig_A
                            getattr(layer, attr_name).data.copy_(blended)
                        break
        
        m1,m2,m3 = compute_rf_mass(model, layer_idx, tokenizer, texts)
        results.append((lam,m1,m2,m3))
        print(f"  λ={lam:.2f} → 1g={m1:.4f}, 2g={m2:.4f}, 3g={m3:.4f}")

    # 5️⃣ Restore originals
    for layer in ssm_layers:
        A = None
        attr_name = None
        for name in ['A', 'A_log', 'state_A', 'kernel_A']:
            if hasattr(layer, name):
                A = getattr(layer, name)
                attr_name = name
                break
        
        if A is not None and isinstance(A, torch.Tensor):
            # Find corresponding original tensor
            for orig_A in original_As:
                if orig_A.shape == A.shape:
                    with torch.no_grad():
                        getattr(layer, attr_name).data.copy_(orig_A)
                    break

    # 6️⃣ Plot causal curve
    lam_vals, m1s, m2s, m3s = zip(*results)
    plt.figure(figsize=(7,5))
    plt.plot(lam_vals, m1s, 'o-', label='1-gram mass')
    plt.plot(lam_vals, m2s, 's-', label='2-gram mass')
    plt.plot(lam_vals, m3s, '^-', label='3-gram mass')
    plt.xlabel("Interpolation λ (0=random, 1=original)")
    plt.ylabel("Receptive-field mass")
    plt.title(f"State-kernel interpolation at layer {layer_idx}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/kernel_interpolation_curve.png", dpi=300)
    plt.close()
    print("✅ Saved causal interpolation plot → plots/kernel_interpolation_curve.png")

    return results


# === Step: Kernel Causality Interpolation Experiment ===
def run_kernel_interpolation_experiment():
    from main import setup_model_and_tokenizer, load_analysis_texts

    print("\n🧪 Running kernel interpolation causality test for Mamba...")
    model, tokenizer = setup_model_and_tokenizer("state-spaces/mamba-130m-hf")
    texts = load_analysis_texts(50)

    # Diagnostic: Check what parameters with 'A' exist
    print("\n🔍 Diagnostic: Parameters containing 'A':")
    for name, param in model.named_parameters():
        if 'A' in name.lower():
            print(f"  {name}: {param.shape}")
    
    # Also check for other possible kernel names
    print("\n🔍 Diagnostic: All parameter names (first 20):")
    param_names = [name for name, _ in model.named_parameters()]
    for name in param_names[:20]:
        print(f"  {name}")
    if len(param_names) > 20:
        print(f"  ... and {len(param_names) - 20} more parameters")

    results = interpolate_state_kernel_effect(model, tokenizer, texts)

    if results:
        print("\n📈 Interpolation Results:")
        for lam, m1, m2, m3 in results:
            print(f"  λ={lam:.2f} → 1g={m1:.4f}, 2g={m2:.4f}, 3g={m3:.4f}")

        # Save to log file for later analysis
        import json, datetime, os
        os.makedirs("logs", exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = f"logs/kernel_interpolation_results_{ts}.json"
        with open(out_file, "w") as f:
            json.dump({"results": results}, f, indent=2)
        print(f"✅ Saved interpolation results to {out_file}")
        
        # Store results globally for summary display
        global kernel_interpolation_results
        kernel_interpolation_results = results
    else:
        print("⚠️ No interpolation results — check layer attributes (A, state_A, kernel_A).")
        kernel_interpolation_results = {"error": "No valid kernels found"}


# -------------------------
# Main Analysis Function
# -------------------------
def run_simple_extended_analysis():
    """
    Run the simplified extended n-gram analysis.
    """
    try:
        from main import setup_model_and_tokenizer, load_analysis_texts
    except ImportError:
        print("main.py not found; please place this script with main.py")
        return
    
    texts = load_analysis_texts(50)
    all_results = {}
    
    for model_label, model_name in models_to_analyze.items():
        print(f"\n🔍 Running simple extended analysis for {model_label}")
        
        try:
            model, tokenizer = setup_model_and_tokenizer(model_name)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            num_layers = getattr(model.config, "num_hidden_layers", 6)
            model_results = {}
            
            # Determine which layers to analyze based on model type
            if isinstance(LAYERS_TO_ANALYZE, dict):
                # Use model-specific configuration
                layers_to_analyze = LAYERS_TO_ANALYZE.get(model_label, LAYERS_TO_ANALYZE.get("default", [0, 10]))
                print(f"  Analyzing specific layers for {model_label}: {layers_to_analyze}")
            elif LAYERS_TO_ANALYZE is None:
                # Analyze all layers to capture the full n-gram emergence pattern
                layers_to_analyze = list(range(num_layers))
                print(f"  Analyzing all {num_layers} layers")
            else:
                # Analyze only specified layers (legacy support)
                layers_to_analyze = LAYERS_TO_ANALYZE
                print(f"  Analyzing specific layers: {layers_to_analyze}")
            
            for layer_idx in layers_to_analyze:
                print(f"  Analyzing layer {layer_idx}/{num_layers-1}")
                
                layer_results = {}
                
                # Step 1: Collect triggers
                neuron_triggers, neuron_activations = collect_ngram_triggers_enhanced(
                    model, tokenizer, texts[:20], layer_idx, model_label=model_label
                )
                layer_results['neuron_triggers'] = neuron_triggers
                layer_results['neuron_activations'] = neuron_activations
                
                # Step 2: Analyze head-like behavior
                head_analysis = analyze_head_like_behavior(
                    neuron_triggers, neuron_activations, layer_idx, model,
                    tokenizer=tokenizer, texts=texts[:15], model_label=model_label,
                    use_clustering=True
                )
                layer_results['head_analysis'] = head_analysis

                # Also slicing (optional)
                head_analysis_slicing = analyze_head_like_behavior(
                    neuron_triggers, neuron_activations, layer_idx, model,
                    tokenizer=tokenizer, texts=texts[:15], model_label=model_label,
                    use_clustering=False
                )
                layer_results['head_analysis_slicing'] = head_analysis_slicing
                
                # Step 3: Receptive Field Analysis
                print(f"    Priority 1 - Receptive Field")
                rf_summary = receptive_field_analysis(model, tokenizer, texts[:5], layer_idx=layer_idx, device=device)
                layer_results["Priority1_RF"] = rf_summary
                print(f"[DEBUG] RF summary for {model_label} layer {layer_idx}: {rf_summary}")
                
                model_results[layer_idx] = layer_results
            
            # Step 3: Aggregate clustering
            clustering_analysis = analyze_ngram_clustering_patterns(model_results, model_label)
            model_results['clustering_analysis'] = clustering_analysis
            
            all_results[model_label] = model_results
            
            # Clear model from memory
            del model, tokenizer
            clear_memory()
            print(f"Memory after {model_label}: {get_memory_usage()}")
            
        except Exception as e:
            print(f"❌ Error analyzing {model_label}: {e}")
            # Clear memory even on error
            clear_memory()
            continue
    
    # Step 4: Emergence analysis
    emergence_analysis = analyze_ngram_emergence_patterns(all_results)
    
    # Step 5: Plots
    plot_head_behavior_analysis_simple(all_results)
    plot_validation_and_comparison_analysis(all_results)
    plot_ngram_emergence_analysis(all_results, emergence_analysis)
    
    # Step 6: Save results
    save_simple_analysis_results(all_results)
    save_raw_head_distributions(all_results)  # <--- NEW
    
    print("\n🎉 Simple extended n-gram analysis complete!")
    
def analyze_linear_token_decode(model, tokenizer, texts, layer_idx, model_label):
    """
    Wrapper function to run linear token decode analysis on a specific layer.
    """
    try:
        layers = get_model_layers(model)
        layer_module = layers[layer_idx]
        
        # Collect activations and token IDs
        activations = []
        token_ids = []
        
        def hook_fn(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            acts = out.detach().cpu().numpy()
            activations.append(acts)
        
        handle = layer_module.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
                current_token_ids = inputs["input_ids"].numpy().flatten()
                token_ids.extend(current_token_ids)
                _ = model(**inputs)
        
        handle.remove()
        
        if not activations:
            return {"accuracy": 0.0, "error": "No activations collected"}
        
        # Flatten activations
        try:
            all_acts = np.concatenate(activations, axis=0)
            all_acts = all_acts.reshape(all_acts.shape[0], -1)  # Flatten to [samples, features]
        except Exception as e:
            print(f"⚠️  Error concatenating activations: {e}")
            return {"accuracy": 0.0, "error": f"Activation concatenation failed: {e}"}
        
        # Ensure we have matching number of samples
        min_samples = min(len(all_acts), len(token_ids))
        all_acts = all_acts[:min_samples]
        token_ids = np.array(token_ids[:min_samples])
        
        # Run linear decode
        accuracy, clf = linear_decode_accuracy(all_acts, token_ids)
        
        return {
            "accuracy": float(accuracy),
            "num_samples": min_samples,
            "vocab_size": len(np.unique(token_ids)),
            "classifier_coef_shape": clf.coef_.shape if hasattr(clf, 'coef_') else None
        }
        
    except Exception as e:
        return {"accuracy": 0.0, "error": str(e)}


def analyze_gradient_attribution(model, tokenizer, texts, layer_idx, model_label):
    """
    Wrapper function to run gradient attribution analysis on a specific layer.
    """
    try:
        device = next(model.parameters()).device
        
        # Prepare inputs
        inputs = tokenizer(" ".join(texts[:5]), return_tensors="pt", 
                          truncation=True, max_length=128).to(device)
        
        # Compute gradient attribution using the GPT-2 specific function
        importance = compute_gradient_attribution_gpt2(model, tokenizer, inputs, layer_idx)
        
        return {
            "importance": importance,
            "mean_importance": importance,
            "std_importance": 0.0,
            "max_importance": importance,
            "min_importance": importance,
            "num_neurons": 1
        }
        
    except Exception as e:
        return {"importance": None, "error": str(e)}


def analyze_receptive_field_probe(model, tokenizer, texts, layer_idx, model_label):
    """
    Wrapper function to run receptive field empirical probe on a specific layer.
    """
    try:
        layers = get_model_layers(model)
        layer_module = layers[layer_idx]
        
        all_scores = []
        
        # Run probe on multiple texts and average results
        for text in texts:
            scores = receptive_field_probe(model, tokenizer, text, layer_module)
            all_scores.append(scores)
        
        # Average scores across texts
        avg_scores = {}
        for ngram_size in range(1, 4):  # 1, 2, 3-grams
            scores_for_size = [s.get(ngram_size, 0) for s in all_scores if ngram_size in s]
            avg_scores[ngram_size] = float(np.mean(scores_for_size)) if scores_for_size else 0.0
        
        return {
            "receptive_scores": avg_scores,
            "num_texts_probed": len(texts),
            "individual_scores": all_scores
        }
        
    except Exception as e:
        return {"receptive_scores": {}, "error": str(e)}


def analyze_neuron_receptive_field_alignment(neuron_triggers, receptive_field_results):
    """
    Wrapper function to analyze alignment between neuron triggers and receptive field.
    """
    try:
        if "receptive_scores" not in receptive_field_results:
            return {"alignment": {}, "error": "No receptive field results"}
        
        receptive_scores = receptive_field_results["receptive_scores"]
        alignment_results = neuron_vs_receptive_field(neuron_triggers, receptive_scores)
        
        return {
            "alignment": alignment_results,
            "receptive_scores": receptive_scores,
            "neuron_counts": alignment_results["neuron_counts"]
        }
        
    except Exception as e:
        return {"alignment": {}, "error": str(e)}


def analyze_ngram_emergence_patterns(all_results):
    """
    Analyze when n-gram patterns first emerge across different model sizes.
    This addresses the key finding: smaller models start n-gram patterns early,
    larger models start in middle layers.
    """
    emergence_analysis = {
        'model_emergence_layers': {},
        'ngram_first_appearance': {},
        'emergence_strength': {},
        'model_size_correlation': {}
    }
    
    for model_label, results in all_results.items():
        if 'clustering_analysis' not in results:
            continue
            
        clustering = results['clustering_analysis']
        emergence_layers = {}
        first_appearance = {}
        emergence_strength = {}
        
        # Find when each n-gram type first appears with significant strength
        for ngram_size in [1, 2, 3]:
            layer_counts = []
            layers = sorted(clustering['layer_patterns'].keys())
            
            for layer_idx in layers:
                count = clustering['layer_patterns'][layer_idx].get(ngram_size, 0)
                layer_counts.append(count)
            
            # Find first layer where this n-gram type appears significantly
            # (more than 1 head specialized in this n-gram type)
            first_significant_layer = None
            for i, count in enumerate(layer_counts):
                if count > 1:  # More than 1 head specialized
                    first_significant_layer = layers[i]
                    break
            
            emergence_layers[ngram_size] = first_significant_layer
            first_appearance[ngram_size] = layer_counts
            
            # Calculate emergence strength (how quickly it grows)
            if len(layer_counts) > 1:
                growth_rate = np.gradient(layer_counts)
                emergence_strength[ngram_size] = {
                    'max_growth_rate': float(np.max(growth_rate)),
                    'avg_growth_rate': float(np.mean(growth_rate)),
                    'total_growth': layer_counts[-1] - layer_counts[0] if layer_counts else 0
                }
        
        emergence_analysis['model_emergence_layers'][model_label] = emergence_layers
        emergence_analysis['ngram_first_appearance'][model_label] = first_appearance
        emergence_analysis['emergence_strength'][model_label] = emergence_strength
        
        # Extract model size for correlation analysis
        if "130M" in model_label:
            model_size = 130
        elif "370M" in model_label:
            model_size = 370
        elif "790M" in model_label:
            model_size = 790
        elif "1.4B" in model_label:
            model_size = 1400
        elif "GPT-2" in model_label:
            model_size = 117  # GPT-2 small
        else:
            model_size = 100  # Default
            
        emergence_analysis['model_size_correlation'][model_label] = model_size
    
    return emergence_analysis

def save_simple_analysis_results(all_results, save_dir="logs"):
    """
    Save simplified analysis results to JSON log file.
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_dir, f"simple_extended_analysis_{timestamp}.json")
    
    # Convert numpy arrays and sets to lists for JSON serialization
    serializable_results = {}
    for model_label, results in all_results.items():
        serializable_results[model_label] = {}
        
        for layer_idx, layer_results in results.items():
            serializable_results[model_label][str(layer_idx)] = {}
            
            # Convert head analysis
            if 'head_analysis' in layer_results:
                head_analysis = layer_results['head_analysis']
                serializable_results[model_label][str(layer_idx)]['head_analysis'] = {
                    'head_specializations': {
                        str(k): v for k, v in head_analysis['head_specializations'].items()
                    },
                    'head_specialization_scores': {
                        str(k): v for k, v in head_analysis['head_specialization_scores'].items()
                    },
                    'head_activation_patterns': {
                        str(k): v for k, v in head_analysis['head_activation_patterns'].items()
                    },
                    'head_ngram_distributions': {
                        str(k): v for k, v in head_analysis['head_ngram_distributions'].items()
                    },
                    'head_compositions': {
                        str(k): v for k, v in head_analysis['head_compositions'].items()
                    },
                    'head_validations': {
                        str(k): v for k, v in head_analysis['head_validations'].items()
                    }
                }
            
            # Convert clustering analysis
            if 'clustering_analysis' in layer_results:
                serializable_results[model_label][str(layer_idx)]['clustering_analysis'] = layer_results['clustering_analysis']
    
    log_data = {
        "timestamp": timestamp,
        "analysis_type": "simple_extended_ngram_analysis",
        "models_analyzed": list(all_results.keys()),
        "results": serializable_results
    }
    
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    print(f"✅ Simple analysis results logged to '{log_file}'")

def plot_head_behavior_analysis_simple(all_results, save_dir="plots"):
    """
    Plot head-level behavior analysis results.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: N-gram head specialization across layers
    plt.figure(figsize=(12, 8))
    
    # Define colors for each model (matching receptive_field_sensitivity plot)
    model_colors = {
        'Mamba-130M': 'black',
        'Mamba-370M': 'blue',
        'Mamba-790M': 'darkgreen',
        'Mamba-1.4B': 'red',
        'GPT-2': 'purple',
    }
    
    # Define markers for each n-gram type (matching receptive_field_sensitivity plot)
    ngram_markers = {
        1: '^',  # triangle for 1-gram
        2: 's',  # square for 2-gram
        3: 'o',  # circle for 3-gram
    }
    
    # Get default color if model not in dict
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    model_list = list(all_results.keys())
    
    # Print which models are being plotted
    print(f"  Plotting n-gram head specialization for {len(all_results)} models: {list(all_results.keys())}")
    
    for model_idx, (model_label, results) in enumerate(all_results.items()):
        if 'clustering_analysis' in results:
            clustering = results['clustering_analysis']
            if 'layer_patterns' in clustering and clustering['layer_patterns']:
                layers = sorted(clustering['layer_patterns'].keys())
                
                # Get color for this model
                color = model_colors.get(model_label, 'gray')
                
                for ngram_size in [1, 2, 3]:
                    # Skip GPT-2 1-gram and 2-gram (only plot 3-gram for GPT-2)
                    if model_label == 'GPT-2' and ngram_size in [1, 2]:
                        continue
                    
                    head_counts = []
                    valid_layers = []
                    for layer_idx in layers:
                        count = clustering['layer_patterns'][layer_idx].get(ngram_size, 0)
                        if count > 0:  # Only include layers with non-zero counts
                            head_counts.append(count)
                            valid_layers.append(layer_idx)
                    
                    # Only plot if there are non-zero values
                    if valid_layers and head_counts:
                        marker = ngram_markers[ngram_size]
                        plt.plot(valid_layers, head_counts, marker=marker, color=color,
                                label=f"{model_label} {ngram_size}-gram", linewidth=2, markersize=8,
                                linestyle='-', alpha=0.8)
                        print(f"    Plotted {model_label} {ngram_size}-gram: {len(valid_layers)} layers")
            else:
                print(f"    Warning: {model_label} missing layer_patterns in clustering_analysis")
        else:
            print(f"    Warning: {model_label} missing clustering_analysis")
    
    plt.xlabel("Layer Index")
    plt.ylabel("Number of Specialized Heads")
    plt.title("N-gram Head Specialization Across Layers")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ngram_head_specialization_layers.png"), 
                dpi=300, bbox_inches="tight")
    plt.close()
    
    # Plot 2: Head clustering metrics comparison
    plt.figure(figsize=(10, 6))
    
    models = list(all_results.keys())
    ngram_sizes = [1, 2, 3]
    
    early_late_ratios = {ngram_size: [] for ngram_size in ngram_sizes}
    
    for model_label, results in all_results.items():
        if 'clustering_analysis' in results:
            clustering = results['clustering_analysis']
            for ngram_size in ngram_sizes:
                if ngram_size in clustering['clustering_metrics']:
                    ratio = clustering['clustering_metrics'][ngram_size]['early_late_ratio']
                    early_late_ratios[ngram_size].append(ratio)
                else:
                    early_late_ratios[ngram_size].append(0)
    
    x = np.arange(len(models))
    width = 0.25
    
    for i, ngram_size in enumerate(ngram_sizes):
        plt.bar(x + i*width, early_late_ratios[ngram_size], width, 
               label=f'{ngram_size}-gram heads', alpha=0.7)
    
    plt.xlabel('Models')
    plt.ylabel('Early/Late Clustering Ratio')
    plt.title('N-gram Head Clustering Patterns (Higher = More Early Clustering)')
    plt.xticks(x + width, models, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "head_clustering_patterns.png"), 
               dpi=300, bbox_inches="tight")
    plt.close()
    
    # Plot 3: Head specialization scores heatmap
    plt.figure(figsize=(12, 8))
    
    # Collect specialization scores for heatmap
    all_scores = []
    model_labels = []
    layer_indices = []
    
    for model_label, results in all_results.items():
        if 'clustering_analysis' in results:
            clustering = results['clustering_analysis']
            for layer_idx, head_distributions in clustering['head_distributions'].items():
                for head_id, head_info in head_distributions.items():
                    all_scores.append(head_info['specialization_score'])
                    model_labels.append(f"{model_label}_L{layer_idx}")
                    layer_indices.append(layer_idx)
    
    if all_scores:
        # Create a simplified heatmap showing average specialization scores
        unique_models = list(set(model_labels))
        unique_layers = sorted(list(set(layer_indices)))
        
        heatmap_data = np.zeros((len(unique_models), len(unique_layers)))
        
        for i, model in enumerate(unique_models):
            for j, layer in enumerate(unique_layers):
                model_layer_key = f"{model}_L{layer}"
                if model_layer_key in model_labels:
                    # Get average specialization score for this model-layer combination
                    indices = [k for k, label in enumerate(model_labels) if label == model_layer_key]
                    if indices:
                        avg_score = np.mean([all_scores[k] for k in indices])
                        heatmap_data[i, j] = avg_score
        
        sns.heatmap(heatmap_data, 
                   xticklabels=unique_layers,
                   yticklabels=unique_models,
                   annot=True, fmt='.2f', cmap='viridis')
        plt.title("Head Specialization Scores Heatmap")
        plt.xlabel("Layer Index")
        plt.ylabel("Model")
        plt.savefig(os.path.join(save_dir, "head_specialization_heatmap.png"), 
                   dpi=300, bbox_inches="tight")
        plt.close()
    
    print(f"✅ Head behavior plots saved to {save_dir}/")

def plot_validation_and_comparison_analysis(all_results, save_dir="plots"):
    """
    Plot validation results and method comparison between clustering and slicing.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: Validation scores across layers
    plt.figure(figsize=(12, 8))
    
    for model_label, results in all_results.items():
        if 'clustering_analysis' in results:
            clustering = results['clustering_analysis']
            layers = sorted(clustering['validation_metrics'].keys())
            
            validation_scores = []
            for layer_idx in layers:
                if layer_idx in clustering['validation_metrics']:
                    validation_scores.append(clustering['validation_metrics'][layer_idx]['mean_validation_score'])
                else:
                    validation_scores.append(0.0)
            
            plt.plot(layers, validation_scores, marker='o', 
                    label=f"{model_label} validation scores")
    
    plt.xlabel("Layer Index")
    plt.ylabel("Mean Validation Score")
    plt.title("Head Specialization Validation Scores Across Layers")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "validation_scores_layers.png"), 
                dpi=300, bbox_inches="tight")
    plt.close()
    
    # Plot 2: Method comparison (clustering vs slicing)
    plt.figure(figsize=(14, 10))
    
    models = list(all_results.keys())
    ngram_sizes = [1, 2, 3]
    
    # Create subplots for different metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot 1: Validation scores comparison
    clustering_validation = []
    slicing_validation = []
    
    for model_label, results in all_results.items():
        if 'clustering_analysis' in results:
            clustering = results['clustering_analysis']
            if 'method_comparison' in clustering:
                for layer_idx, comparison in clustering['method_comparison'].items():
                    clustering_validation.append(comparison['clustering_validation_mean'])
                    slicing_validation.append(comparison['slicing_validation_mean'])
    
    if clustering_validation and slicing_validation:
        axes[0, 0].scatter(clustering_validation, slicing_validation, alpha=0.6)
        axes[0, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
        axes[0, 0].set_xlabel('Clustering Validation Score')
        axes[0, 0].set_ylabel('Slicing Validation Score')
        axes[0, 0].set_title('Validation Scores: Clustering vs Slicing')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Subplot 2: Head count comparison
    clustering_counts = []
    slicing_counts = []
    
    for model_label, results in all_results.items():
        if 'clustering_analysis' in results:
            clustering = results['clustering_analysis']
            if 'method_comparison' in clustering:
                for layer_idx, comparison in clustering['method_comparison'].items():
                    clustering_counts.append(comparison['clustering_head_count'])
                    slicing_counts.append(comparison['slicing_head_count'])
    
    if clustering_counts and slicing_counts:
        axes[0, 1].scatter(clustering_counts, slicing_counts, alpha=0.6)
        axes[0, 1].plot([0, max(max(clustering_counts), max(slicing_counts))], 
                       [0, max(max(clustering_counts), max(slicing_counts))], 'r--', alpha=0.5)
        axes[0, 1].set_xlabel('Clustering Head Count')
        axes[0, 1].set_ylabel('Slicing Head Count')
        axes[0, 1].set_title('Head Count: Clustering vs Slicing')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Subplot 3: Specialization distribution comparison
    specialization_data = {'clustering': defaultdict(int), 'slicing': defaultdict(int)}
    
    for model_label, results in all_results.items():
        if 'clustering_analysis' in results:
            clustering = results['clustering_analysis']
            if 'method_comparison' in clustering:
                for layer_idx, comparison in clustering['method_comparison'].items():
                    for ngram_size in ngram_sizes:
                        clustering_spec = comparison['clustering_specializations'].get(ngram_size, 0)
                        slicing_spec = comparison['slicing_specializations'].get(ngram_size, 0)
                        specialization_data['clustering'][ngram_size] += clustering_spec
                        specialization_data['slicing'][ngram_size] += slicing_spec
    
    x = np.arange(len(ngram_sizes))
    width = 0.35
    
    clustering_spec_counts = [specialization_data['clustering'][n] for n in ngram_sizes]
    slicing_spec_counts = [specialization_data['slicing'][n] for n in ngram_sizes]
    
    axes[1, 0].bar(x - width/2, clustering_spec_counts, width, label='Clustering', alpha=0.7)
    axes[1, 0].bar(x + width/2, slicing_spec_counts, width, label='Slicing', alpha=0.7)
    axes[1, 0].set_xlabel('N-gram Size')
    axes[1, 0].set_ylabel('Number of Specialized Heads')
    axes[1, 0].set_title('Specialization Distribution Comparison')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([f'{n}-gram' for n in ngram_sizes])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Subplot 4: Validation score distribution
    all_validation_scores = []
    method_labels = []
    
    for model_label, results in all_results.items():
        if 'clustering_analysis' in results:
            clustering = results['clustering_analysis']
            if 'method_comparison' in clustering:
                for layer_idx, comparison in clustering['method_comparison'].items():
                    all_validation_scores.extend([
                        comparison['clustering_validation_mean'],
                        comparison['slicing_validation_mean']
                    ])
                    method_labels.extend(['Clustering', 'Slicing'])
    
    if all_validation_scores:
        # Create box plot
        clustering_scores = [score for i, score in enumerate(all_validation_scores) 
                           if method_labels[i] == 'Clustering']
        slicing_scores = [score for i, score in enumerate(all_validation_scores) 
                         if method_labels[i] == 'Slicing']
        
        axes[1, 1].boxplot([clustering_scores, slicing_scores], 
                          labels=['Clustering', 'Slicing'])
        axes[1, 1].set_ylabel('Validation Score')
        axes[1, 1].set_title('Validation Score Distribution')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "method_comparison_analysis.png"), 
               dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✅ Validation and comparison plots saved to {save_dir}/")

def plot_ngram_emergence_analysis(all_results, emergence_analysis, save_dir="plots"):
    """
    Plot the n-gram emergence patterns to show why smaller models start early,
    larger models start in middle layers.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: N-gram emergence across layers for different model sizes
    plt.figure(figsize=(15, 10))
    
    # Create subplots for each n-gram type
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    ngram_sizes = [1, 2, 3]
    
    # Define distinct colors for each model (matching receptive_field_sensitivity plot)
    model_colors = {
        'Mamba-130M': 'black',
        'Mamba-370M': 'blue',
        'Mamba-790M': 'darkgreen',
        'Mamba-1.4B': 'red',
        'GPT-2': 'purple',
    }
    
    # Default colors if model not in dict
    default_colors = ['black', 'blue', 'darkgreen', 'red', 'purple', 'gray', 'orange', 'brown']
    model_list = list(all_results.keys())
    
    # Print which models are available
    print(f"  Plotting n-gram emergence for {len(all_results)} models: {list(all_results.keys())}")
    
    for ngram_idx, ngram_size in enumerate(ngram_sizes):
        ax = axes[ngram_idx]
        
        models_plotted = 0
        for model_idx, (model_label, results) in enumerate(all_results.items()):
            if 'clustering_analysis' in results:
                clustering = results['clustering_analysis']
                if 'layer_patterns' in clustering and clustering['layer_patterns']:
                    layers = sorted(clustering['layer_patterns'].keys())
                    
                    layer_counts = []
                    for layer_idx in layers:
                        count = clustering['layer_patterns'][layer_idx].get(ngram_size, 0)
                        layer_counts.append(count)
                    
                    # Plot all models, even if some have zero counts
                    if layers and len(layer_counts) == len(layers):
                        # Get color for this model
                        color = model_colors.get(model_label, default_colors[model_idx % len(default_colors)])
                        
                        # Always plot, even if all values are zero (line will be at y=0)
                        # Use zorder to ensure lines are visible (higher zorder = on top)
                        zorder = 10 - model_idx  # Earlier models (like Mamba-130M) get higher zorder
                        ax.plot(layers, layer_counts, marker='o', 
                               label=f"{model_label}", color=color, linewidth=2.5, markersize=7, 
                               alpha=0.9, zorder=zorder)
                        models_plotted += 1
                        max_count = max(layer_counts) if layer_counts else 0
                        total_count = sum(layer_counts)
                        print(f"    Plotted {model_label} for {ngram_size}-gram: {len(layers)} layers, max_count={max_count}, total={total_count}, color={color}")
                    else:
                        print(f"    Skipping {model_label} for {ngram_size}-gram: layers={len(layers) if layers else 0}, counts={len(layer_counts)}")
                else:
                    print(f"    Warning: {model_label} missing layer_patterns in clustering_analysis")
                    if 'clustering_analysis' in results:
                        print(f"      clustering_analysis keys: {list(results['clustering_analysis'].keys())}")
            else:
                print(f"    Warning: {model_label} missing clustering_analysis - keys: {list(results.keys())}")
        
        ax.set_xlabel("Layer Index")
        ax.set_ylabel(f"Number of {ngram_size}-gram Specialized Heads")
        ax.set_title(f"{ngram_size}-gram Pattern Emergence Across Layers")
        if models_plotted > 0:
            ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Model size vs emergence layer correlation
    ax = axes[3]
    
    model_sizes = []
    emergence_layers_2gram = []
    emergence_layers_3gram = []
    model_labels_list = []
    
    for model_label, emergence_data in emergence_analysis['model_emergence_layers'].items():
        model_size = emergence_analysis['model_size_correlation'][model_label]
        emergence_2 = emergence_data.get(2, None)
        emergence_3 = emergence_data.get(3, None)
        
        # Only include models with valid emergence data (not None)
        if emergence_2 is not None or emergence_3 is not None:
            model_sizes.append(model_size)
            model_labels_list.append(model_label)
            emergence_layers_2gram.append(emergence_2 if emergence_2 is not None else 0)
            emergence_layers_3gram.append(emergence_3 if emergence_3 is not None else 0)
    
    if model_sizes and (any(e is not None and e > 0 for e in emergence_layers_2gram) or 
                        any(e is not None and e > 0 for e in emergence_layers_3gram)):
        # Filter out None values for plotting
        valid_2gram = [(size, layer) for size, layer in zip(model_sizes, emergence_layers_2gram) 
                       if layer is not None and layer > 0]
        valid_3gram = [(size, layer) for size, layer in zip(model_sizes, emergence_layers_3gram) 
                       if layer is not None and layer > 0]
        
        if valid_2gram:
            sizes_2, layers_2 = zip(*valid_2gram)
            ax.scatter(sizes_2, layers_2, label='2-gram emergence', 
                      color='red', s=100, alpha=0.7)
        
        if valid_3gram:
            sizes_3, layers_3 = zip(*valid_3gram)
            ax.scatter(sizes_3, layers_3, label='3-gram emergence', 
                      color='blue', s=100, alpha=0.7)
        
        # Add trend lines only if we have enough valid data points
        if len(valid_2gram) > 2:
            sizes_2, layers_2 = zip(*valid_2gram)
            z2 = np.polyfit(list(sizes_2), list(layers_2), 1)
            p2 = np.poly1d(z2)
            ax.plot(list(sizes_2), p2(list(sizes_2)), 'r--', alpha=0.5)
        
        if len(valid_3gram) > 2:
            sizes_3, layers_3 = zip(*valid_3gram)
            z3 = np.polyfit(list(sizes_3), list(layers_3), 1)
            p3 = np.poly1d(z3)
            ax.plot(list(sizes_3), p3(list(sizes_3)), 'b--', alpha=0.5)
    
    ax.set_xlabel("Model Size (Millions of Parameters)")
    ax.set_ylabel("First Layer with Significant N-gram Patterns")
    ax.set_title("Model Size vs N-gram Emergence Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ngram_emergence_patterns.png"), 
               dpi=300, bbox_inches="tight")
    plt.close()
    
    # Plot 2: Detailed emergence timeline
    plt.figure(figsize=(16, 8))
    
    # Create a timeline plot showing when each n-gram type emerges
    models = list(all_results.keys())
    ngram_types = [1, 2, 3]
    
    # Create heatmap data with proper error handling
    layer_counts = []
    for results in all_results.values():
        if 'clustering_analysis' in results and results['clustering_analysis']:
            clustering = results['clustering_analysis']
            if 'layer_patterns' in clustering and clustering['layer_patterns']:
                layer_counts.append(len(clustering['layer_patterns']))
    
    if not layer_counts or max(layer_counts) == 0:
        print("Warning: No clustering analysis results found, skipping emergence heatmap")
        return
    
    max_layers = max(layer_counts)
    emergence_matrix = np.zeros((len(models), max_layers))
    
    model_labels = []
    for i, (model_label, results) in enumerate(all_results.items()):
        if 'clustering_analysis' in results:
            model_labels.append(model_label)
            clustering = results['clustering_analysis']
            layers = sorted(clustering['layer_patterns'].keys())
            
            for j, layer_idx in enumerate(layers):
                # Use 2-gram count as the main indicator
                count = clustering['layer_patterns'][layer_idx].get(2, 0)
                emergence_matrix[i, j] = count
    
    # Plot heatmap with error handling
    if emergence_matrix.size > 0 and emergence_matrix.max() > 0:
        sns.heatmap(emergence_matrix, 
                   xticklabels=range(emergence_matrix.shape[1]),
                   yticklabels=model_labels,
                   annot=True, fmt='.0f', cmap='viridis')
        plt.title("2-gram Pattern Emergence Heatmap Across Models and Layers")
        plt.xlabel("Layer Index")
        plt.ylabel("Model")
        plt.savefig(os.path.join(save_dir, "emergence_heatmap.png"), 
                   dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Emergence heatmap saved to {save_dir}/emergence_heatmap.png")
    else:
        print("⚠️  Skipping heatmap: no valid data to plot")
    
    print(f"✅ N-gram emergence analysis plots saved to {save_dir}/")


def plot_linear_decode_analysis(all_results, save_dir="plots"):
    """
    Plot linear decode accuracy and gradient attribution results across layers.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: Linear decode accuracy across layers
    plt.figure(figsize=(12, 8))
    
    for model_label, results in all_results.items():
        layers = []
        accuracies = []
        
        for layer_idx, layer_results in results.items():
            if isinstance(layer_idx, int) and 'linear_decode' in layer_results:
                linear_decode = layer_results['linear_decode']
                if 'accuracy' in linear_decode and 'error' not in linear_decode:
                    layers.append(layer_idx)
                    accuracies.append(linear_decode['accuracy'])
        
        if layers and accuracies:
            plt.plot(layers, accuracies, marker='o', label=f"{model_label}", linewidth=2)
    
    plt.xlabel("Layer Index")
    plt.ylabel("Linear Decode Accuracy")
    plt.title("Linear Token Decode Accuracy Across Layers")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "linear_decode_accuracy.png"), 
               dpi=300, bbox_inches="tight")
    plt.close()
    
    # Plot 2: Gradient attribution importance across layers
    plt.figure(figsize=(12, 8))
    
    # Define colors for each model (matching other plots)
    model_colors = {
        'Mamba-130M': 'black',
        'Mamba-370M': 'blue',
        'Mamba-790M': 'darkgreen',
        'Mamba-1.4B': 'red',
        'GPT-2': 'purple',
    }
    
    for model_label, results in all_results.items():
        layers = []
        mean_importance = []
        std_importance = []
        
        for layer_idx, layer_results in results.items():
            if isinstance(layer_idx, int) and 'gradient_attribution' in layer_results:
                grad_attr = layer_results['gradient_attribution']
                if 'mean_importance' in grad_attr and 'error' not in grad_attr:
                    layers.append(layer_idx)
                    mean_importance.append(grad_attr['mean_importance'])
                    std_importance.append(grad_attr['std_importance'])
        
        if layers and mean_importance:
            # Get color for this model
            color = model_colors.get(model_label, 'gray')
            # Use round marker (o) for all models
            plt.errorbar(layers, mean_importance, yerr=std_importance, 
                        marker='o', label=f"{model_label}", capsize=5, linewidth=2, 
                        markersize=8, color=color, alpha=0.8)
    
    plt.xlabel("Layer Index")
    plt.ylabel("Mean Gradient Importance")
    plt.title("Gradient Attribution Importance Across Layers")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "gradient_attribution_importance.png"), 
               dpi=300, bbox_inches="tight")
    plt.close()
    
    # Plot 3: Receptive field sensitivity
    plt.figure(figsize=(12, 8))
    
    # Define colors for each model
    model_colors = {
        'Mamba-130M': 'black',
        'Mamba-370M': 'blue',
        'Mamba-790M': 'darkgreen',
        'Mamba-1.4B': 'red',
        'GPT-2': 'purple',
    }
    
    # Define markers for each n-gram type
    ngram_markers = {
        1: '^',  # triangle for 1-gram
        2: 's',  # square for 2-gram
        3: 'o',  # circle for 3-gram
    }
    
    max_layer = 0  # Track maximum layer across all models
    
    # Also check receptive_field_analysis results (Priority 1 analysis)
    for model_label, results in all_results.items():
        layers = []
        receptive_1gram = []
        receptive_2gram = []
        receptive_3gram = []
        
        # First, try to get from per-layer receptive_field results
        for layer_idx, layer_results in results.items():
            if isinstance(layer_idx, int) and 'receptive_field' in layer_results:
                receptive = layer_results['receptive_field']
                if 'receptive_scores' in receptive and 'error' not in receptive:
                    layers.append(layer_idx)
                    scores = receptive['receptive_scores']
                    receptive_1gram.append(scores.get(1, 0))
                    receptive_2gram.append(scores.get(2, 0))
                    receptive_3gram.append(scores.get(3, 0))
        
        # If no per-layer results, try to get from receptive_field_analysis (all layers)
        if not layers and 'receptive_field_analysis' in results:
            rf_analysis = results['receptive_field_analysis']
            if rf_analysis:
                for layer_idx in sorted(rf_analysis.keys()):
                    if isinstance(layer_idx, int):
                        layer_rf = rf_analysis[layer_idx]
                        if layer_rf:
                            layers.append(layer_idx)
                            receptive_1gram.append(layer_rf.get(1, 0))
                            receptive_2gram.append(layer_rf.get(2, 0))
                            receptive_3gram.append(layer_rf.get(3, 0))
        
        # Get color for this model
        color = model_colors.get(model_label, 'gray')
        
        if layers and receptive_1gram:
            max_layer = max(max_layer, max(layers))
            # Plot each n-gram type with appropriate marker and model color
            plt.plot(layers, receptive_1gram, marker=ngram_markers[1], color=color, 
                    label=f"{model_label} 1-gram", linewidth=2, markersize=8, alpha=0.8)
            plt.plot(layers, receptive_2gram, marker=ngram_markers[2], color=color, 
                    label=f"{model_label} 2-gram", linewidth=2, markersize=8, alpha=0.8)
            plt.plot(layers, receptive_3gram, marker=ngram_markers[3], color=color, 
                    label=f"{model_label} 3-gram", linewidth=2, markersize=8, alpha=0.8)
    
    plt.xlabel("Layer Index")
    plt.ylabel("Receptive Field Sensitivity")
    plt.title("Receptive Field Empirical Probe Results")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    plt.grid(True, alpha=0.3)
    # Set x-axis to show all layers, with small padding
    if max_layer > 0:
        plt.xlim(left=-0.5, right=max_layer + 0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "receptive_field_sensitivity.png"), 
               dpi=300, bbox_inches="tight")
    plt.close()
    
    # Plot 4: Combined analysis heatmap
    plt.figure(figsize=(15, 10))
    
    # Create subplots for different metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Subplot 1: Linear decode accuracy heatmap
    models = list(all_results.keys())
    max_layers = max([max([k for k in results.keys() if isinstance(k, int)]) 
                      for results in all_results.values()])
    
    decode_matrix = np.zeros((len(models), max_layers + 1))
    for i, model_label in enumerate(models):
        results = all_results[model_label]
        for layer_idx, layer_results in results.items():
            if isinstance(layer_idx, int) and 'linear_decode' in layer_results:
                linear_decode = layer_results['linear_decode']
                if 'accuracy' in linear_decode and 'error' not in linear_decode:
                    decode_matrix[i, layer_idx] = linear_decode['accuracy']
    
    sns.heatmap(decode_matrix, xticklabels=range(max_layers + 1), 
               yticklabels=models, annot=False, fmt='.3f', cmap='viridis', ax=axes[0,0])
    axes[0,0].set_title("Linear Decode Accuracy")
    axes[0,0].set_xlabel("Layer Index")
    axes[0,0].set_ylabel("Model")
    
    # Subplot 2: Gradient importance heatmap
    grad_matrix = np.zeros((len(models), max_layers + 1))
    for i, model_label in enumerate(models):
        results = all_results[model_label]
        for layer_idx, layer_results in results.items():
            if isinstance(layer_idx, int) and 'gradient_attribution' in layer_results:
                grad_attr = layer_results['gradient_attribution']
                if 'mean_importance' in grad_attr and 'error' not in grad_attr:
                    grad_matrix[i, layer_idx] = grad_attr['mean_importance']
    
    sns.heatmap(grad_matrix, xticklabels=range(max_layers + 1), 
               yticklabels=models, annot=False, fmt='.3f', cmap='plasma', ax=axes[0,1])
    axes[0,1].set_title("Mean Gradient Importance")
    axes[0,1].set_xlabel("Layer Index")
    axes[0,1].set_ylabel("Model")
    
    # Subplot 3: Receptive field 2-gram sensitivity
    receptive_matrix = np.zeros((len(models), max_layers + 1))
    for i, model_label in enumerate(models):
        results = all_results[model_label]
        for layer_idx, layer_results in results.items():
            if isinstance(layer_idx, int) and 'receptive_field' in layer_results:
                receptive = layer_results['receptive_field']
                if 'receptive_scores' in receptive and 'error' not in receptive:
                    scores = receptive['receptive_scores']
                    receptive_matrix[i, layer_idx] = scores.get(2, 0)
    
    sns.heatmap(receptive_matrix, xticklabels=range(max_layers + 1), 
               yticklabels=models, annot=False, fmt='.3f', cmap='coolwarm', ax=axes[1,0])
    axes[1,0].set_title("Receptive Field 2-gram Sensitivity")
    axes[1,0].set_xlabel("Layer Index")
    axes[1,0].set_ylabel("Model")
    
    # Subplot 4: Summary statistics
    summary_data = []
    for model_label, results in all_results.items():
        total_layers = len([k for k in results.keys() if isinstance(k, int)])
        decode_layers = len([k for k, v in results.items() 
                           if isinstance(k, int) and 'linear_decode' in v and 'error' not in v.get('linear_decode', {})])
        grad_layers = len([k for k, v in results.items() 
                          if isinstance(k, int) and 'gradient_attribution' in v and 'error' not in v.get('gradient_attribution', {})])
        receptive_layers = len([k for k, v in results.items() 
                              if isinstance(k, int) and 'receptive_field' in v and 'error' not in v.get('receptive_field', {})])
        
        summary_data.append([total_layers, decode_layers, grad_layers, receptive_layers])
    
    summary_matrix = np.array(summary_data)
    sns.heatmap(summary_matrix, xticklabels=['Total', 'Decode', 'Gradient', 'Receptive'], 
               yticklabels=models, annot=True, fmt='d', cmap='Blues', ax=axes[1,1])
    axes[1,1].set_title("Analysis Coverage Summary")
    axes[1,1].set_xlabel("Analysis Type")
    axes[1,1].set_ylabel("Model")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "linear_decode_analysis_heatmap.png"), 
               dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✅ Linear decode analysis plots saved to {save_dir}/")


def run_simple_extended_analysis():
    """
    Run the simplified extended n-gram analysis.
    """
    try:
        from main import setup_model_and_tokenizer, load_analysis_texts
    except ImportError:
        print("main.py not found; please place this script with main.py")
        return
    
    texts = load_analysis_texts(50)  # Use smaller subset for faster analysis across many models
    all_results = {}
    
    for model_label, model_name in models_to_analyze.items():
        print(f"\n🔍 Running simple extended analysis for {model_label}")
        
        try:
            model, tokenizer = setup_model_and_tokenizer(model_name)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Special handling for Mamba models (same as main.py)
            if "mamba" in model_name.lower() and "gpt" not in model_name.lower():
                print(f"🔧 Setting up Mamba model: {model_name}")
                # Ensure model is in eval mode for Mamba
                model.eval()
                # Disable gradient computation for efficiency
                for param in model.parameters():
                    param.requires_grad = False
                print(f"✅ Mamba model setup complete")
            else:
                print(f"🔧 Setting up Transformer model: {model_name}")
                model.eval()
                # Keep gradients enabled for Transformer models
            
            # Clear memory before starting analysis
            clear_memory()
            print(f"Memory before analysis: {get_memory_usage()}")
            
            num_layers = getattr(model.config, "num_hidden_layers", 6)
            model_results = {}
            
            # Determine which layers to analyze based on model type
            if isinstance(LAYERS_TO_ANALYZE, dict):
                # Use model-specific configuration
                layers_to_analyze = LAYERS_TO_ANALYZE.get(model_label, LAYERS_TO_ANALYZE.get("default", [0, 10]))
                print(f"  Analyzing specific layers for {model_label}: {layers_to_analyze}")
            elif LAYERS_TO_ANALYZE is None:
                # Analyze all layers to capture the full n-gram emergence pattern
                layers_to_analyze = list(range(num_layers))
                print(f"  Analyzing all {num_layers} layers")
            else:
                # Analyze only specified layers (legacy support)
                layers_to_analyze = LAYERS_TO_ANALYZE
                print(f"  Analyzing specific layers: {layers_to_analyze}")
            
            # Priority 1: Receptive Field Mass Analysis (all layers at once)
            print(f"  Running Priority 1: Receptive Field Mass Analysis...")
            try:
                rf_results = receptive_field_analysis(model, tokenizer, texts[:10], device=device)
                
                # Print results in the new format
                print(f"  Priority 1 - Receptive Field Mass:")
                for n in [1, 2, 3]:
                    # collect values per layer
                    vals = [rf_results[l].get(n, 0) for l in sorted(rf_results.keys())]
                    if vals and max(vals) > 0:
                        peak_layer = int(np.argmax(vals))
                        peak_val = vals[peak_layer]
                        print(f"    {n}-token mass: Peak at layer {peak_layer} (value: {peak_val:.4f})")
                    else:
                        print(f"    {n}-token mass: No significant peaks detected")
                
                # Store results for later use
                model_results['receptive_field_analysis'] = rf_results
            except Exception as e:
                print(f"⚠️  Error in Priority 1 receptive field analysis: {e}")
                model_results['receptive_field_analysis'] = {}
            
            for layer_idx in layers_to_analyze:
                print(f"  Analyzing layer {layer_idx}/{num_layers-1}")
                
                # Clear memory between layers
                clear_memory()
                
                layer_results = {}
                
                # Step 1: Collect triggers with error handling
                try:
                    neuron_triggers, neuron_activations = collect_ngram_triggers_enhanced(
                        model, tokenizer, texts[:MAX_TEXTS_PER_LAYER], layer_idx, model_label=model_label
                    )
                    layer_results['neuron_triggers'] = neuron_triggers
                    layer_results['neuron_activations'] = neuron_activations
                    
                    # Clear memory after collecting triggers
                    clear_memory()
                except Exception as e:
                    print(f"⚠️  Error collecting triggers for layer {layer_idx}: {e}")
                    layer_results['neuron_triggers'] = {}
                    layer_results['neuron_activations'] = {}
                    clear_memory()
                    continue
                
                # Step 2: Analyze head-like behavior with clustering and validation
                print(f"    Analyzing head-like behavior with clustering...")
                try:
                    head_analysis = analyze_head_like_behavior(
                        neuron_triggers, neuron_activations, layer_idx, model,
                        tokenizer=tokenizer, texts=texts[:15], model_label=model_label,
                        use_clustering=True
                    )
                    layer_results['head_analysis'] = head_analysis
                except Exception as e:
                    print(f"⚠️  Error in head analysis for layer {layer_idx}: {e}")
                    layer_results['head_analysis'] = {}
                
                # Also run with original slicing for comparison
                print(f"    Analyzing head-like behavior with slicing (comparison)...")
                try:
                    head_analysis_slicing = analyze_head_like_behavior(
                        neuron_triggers, neuron_activations, layer_idx, model,
                        tokenizer=tokenizer, texts=texts[:15], model_label=model_label,
                        use_clustering=False
                    )
                    layer_results['head_analysis_slicing'] = head_analysis_slicing
                except Exception as e:
                    print(f"⚠️  Error in slicing analysis for layer {layer_idx}: {e}")
                    layer_results['head_analysis_slicing'] = {}
                
                # Step 3: Linear Token Decode Analysis
                print(f"    Running linear token decode analysis...")
                try:
                    linear_decode_results = analyze_linear_token_decode(
                        model, tokenizer, texts[:20], layer_idx, model_label
                    )
                    layer_results['linear_decode'] = linear_decode_results
                except Exception as e:
                    print(f"⚠️  Error in linear decode analysis for layer {layer_idx}: {e}")
                    layer_results['linear_decode'] = {}
                
                # Step 4: Priority Analyses
                print(f"    Running Priority 2: Enhanced Linear Token Decode...")
                try:
                    enhanced_decode_results = enhanced_linear_token_decode(
                        model, tokenizer, texts[:20], layer_idx, model_label,
                        top_k=1000, max_positions=5000
                    )
                    layer_results['enhanced_linear_decode'] = enhanced_decode_results
                except Exception as e:
                    print(f"⚠️  Error in enhanced linear decode for layer {layer_idx}: {e}")
                    layer_results['enhanced_linear_decode'] = {}
                
                print(f"    Running Priority 3: Threshold Sweep Analysis...")
                try:
                    threshold_sweep_results = threshold_sweep_analysis(
                        model, tokenizer, texts[:20], layer_idx, model_label,
                        thresholds=[0.05, 0.1, 0.2, 0.4, 0.6], max_texts=20
                    )
                    layer_results['threshold_sweep'] = threshold_sweep_results
                except Exception as e:
                    print(f"⚠️  Error in threshold sweep for layer {layer_idx}: {e}")
                    layer_results['threshold_sweep'] = {}
                
                # Step 5: Gradient Attribution Analysis
                print(f"    Running gradient attribution analysis...")
                try:
                    # Create inputs for gradient attribution (use same params as test)
                    device = next(model.parameters()).device
                    inputs = tokenizer(" ".join(texts[:5]), return_tensors="pt", 
                                      truncation=True, max_length=64).to(device)
                    imp = compute_gradient_attribution_gpt2(model, tokenizer, inputs, layer_idx, debug=True)
                except Exception as e:
                    print(f"⚠️ Gradient attribution failed for GPT-2 layer {layer_idx}: {e}")
                    imp = 0.0
                
                layer_results['gradient_attribution'] = {
                    "importance": imp,
                    "mean_importance": imp,
                    "std_importance": 0.0,
                    "max_importance": imp,
                    "min_importance": imp,
                    "num_neurons": 1
                }
                
                # Step 5: Receptive Field Empirical Probe
                print(f"    Running receptive field empirical probe...")
                try:
                    receptive_field_results = analyze_receptive_field_probe(
                        model, tokenizer, texts[:5], layer_idx, model_label
                    )
                    layer_results['receptive_field'] = receptive_field_results
                    
                    # Step 6: Neuron vs Receptive Field Alignment
                    print(f"    Analyzing neuron vs receptive field alignment...")
                    alignment_results = analyze_neuron_receptive_field_alignment(
                        neuron_triggers, receptive_field_results
                    )
                    layer_results['neuron_receptive_alignment'] = alignment_results
                except Exception as e:
                    print(f"⚠️  Error in receptive field analysis for layer {layer_idx}: {e}")
                    layer_results['receptive_field'] = {}
                    layer_results['neuron_receptive_alignment'] = {}
                
                model_results[layer_idx] = layer_results
            
            # Step 7: Analyze clustering patterns across layers
            print(f"    Analyzing clustering patterns...")
            clustering_analysis = analyze_ngram_clustering_patterns(model_results, model_label)
            model_results['clustering_analysis'] = clustering_analysis
            
            # Step 8: Run causal test for Mamba models
            if "mamba" in model_name.lower() and "gpt" not in model_name.lower():
                print(f"🧠 Running causal test for {model_label}...")
                try:
                    causal_results = test_state_kernel_causality(model, tokenizer, texts, layer_indices=layers_to_analyze)
                    if causal_results:
                        model_results['causal_test'] = causal_results
                        print(f"✅ Causal test completed for {model_label}")
                    else:
                        print(f"⚠️ Causal test skipped for {model_label}")
                except Exception as e:
                    print(f"⚠️ Error in causal test for {model_label}: {e}")
                    model_results['causal_test'] = {"error": str(e)}
                
                # Add kernel interpolation results if available
                global kernel_interpolation_results
                if kernel_interpolation_results is not None:
                    model_results['kernel_interpolation'] = kernel_interpolation_results
            
            all_results[model_label] = model_results
            
            # Clear model from memory
            del model, tokenizer
            clear_memory()
            print(f"Memory after {model_label}: {get_memory_usage()}")
            
        except Exception as e:
            print(f"❌ Error analyzing {model_label}: {e}")
            # Clear memory even on error
            clear_memory()
            continue
    
    # Step 8: Emergence analysis
    emergence_analysis = analyze_ngram_emergence_patterns(all_results)
    
    # Step 9: Plots
    plot_head_behavior_analysis_simple(all_results)
    plot_validation_and_comparison_analysis(all_results)
    plot_ngram_emergence_analysis(all_results, emergence_analysis)
    plot_linear_decode_analysis(all_results)  # NEW: Plot linear decode results
    
    # Step 10: Save results
    save_simple_analysis_results(all_results)
    save_raw_head_distributions(all_results)  # <--- NEW
    
    print("\n🎉 Simple extended n-gram analysis complete!")
    print("Results saved in 'plots/' and 'logs/' directories")
    
    # Print Priority Analysis Summary
    print("\n🔍 Priority Analysis Summary:")
    print("=" * 50)
    
    for model_label, results in all_results.items():
        print(f"\n{model_label}:")
        
        # Priority 1: Receptive Field Mass
        print("  Priority 1 - Receptive Field Mass:")
        if 'receptive_field_analysis' in results:
            rf_results = results['receptive_field_analysis']
            for n in [1, 2, 3]:
                # collect values per layer
                vals = [rf_results[l].get(n, 0) for l in sorted(rf_results.keys())]
                if vals and max(vals) > 0:
                    peak_layer = int(np.argmax(vals))
                    peak_val = vals[peak_layer]
                    print(f"    {n}-token mass: Peak at layer {peak_layer} (value: {peak_val:.4f})")
                else:
                    print(f"    {n}-token mass: No significant peaks detected")
        else:
            print("    No receptive field analysis results available")
        
        # Priority 2: Enhanced Linear Decode
        print("  Priority 2 - Enhanced Linear Decode:")
        decode_accuracies = []
        for layer_idx, layer_results in results.items():
            if isinstance(layer_idx, int) and 'enhanced_linear_decode' in layer_results:
                decode_result = layer_results['enhanced_linear_decode']
                if 'accuracy' in decode_result and 'error' not in decode_result:
                    decode_accuracies.append((layer_idx, decode_result['accuracy']))
        
        if decode_accuracies:
            best_layer, best_acc = max(decode_accuracies, key=lambda x: x[1])
            print(f"    Best accuracy: {best_acc:.3f} at layer {best_layer}")
            print(f"    Early layers (0-3): {[acc for l, acc in decode_accuracies if l <= 3]}")
        
        # Priority 3: Threshold Sweep
        print("  Priority 3 - Threshold Sweep:")
        for layer_idx, layer_results in results.items():
            if isinstance(layer_idx, int) and 'threshold_sweep' in layer_results:
                threshold_result = layer_results['threshold_sweep']
                if 'threshold_counts' in threshold_result:
                    counts = threshold_result['threshold_counts']
                    if counts:
                        print(f"    Layer {layer_idx}: {counts}")
                        break  # Just show first layer for brevity
    
    # Print emergence analysis summary
    print("\n🔍 N-gram Emergence Pattern Analysis:")
    print("=" * 50)
    for model_label, emergence_data in emergence_analysis['model_emergence_layers'].items():
        model_size = emergence_analysis['model_size_correlation'][model_label]
        print(f"\n{model_label} ({model_size}M parameters):")
        
        for ngram_size in [1, 2, 3]:
            emergence_layer = emergence_data.get(ngram_size, "Not found")
            if emergence_layer is not None:
                print(f"  {ngram_size}-gram patterns first emerge at layer: {emergence_layer}")
            else:
                print(f"  {ngram_size}-gram patterns: No significant emergence detected")
    
    # Print summary
    print("\n📋 Extended Analysis Summary (Steps 3-6):")
    for model_label, results in all_results.items():
        print(f"\n{model_label}:")
        
        # Print linear decode results
        decode_accuracies = []
        for layer_idx, layer_results in results.items():
            if isinstance(layer_idx, int) and 'linear_decode' in layer_results:
                linear_decode = layer_results['linear_decode']
                if 'accuracy' in linear_decode and 'error' not in linear_decode:
                    decode_accuracies.append(linear_decode['accuracy'])
        
        if decode_accuracies:
            print(f"  Linear Token Decode:")
            print(f"    Average accuracy: {np.mean(decode_accuracies):.3f} (±{np.std(decode_accuracies):.3f})")
            print(f"    Best accuracy: {np.max(decode_accuracies):.3f} at layer {np.argmax(decode_accuracies)}")
        
        # Print gradient attribution results
        grad_importances = []
        for layer_idx, layer_results in results.items():
            if isinstance(layer_idx, int) and 'gradient_attribution' in layer_results:
                grad_attr = layer_results['gradient_attribution']
                if 'mean_importance' in grad_attr and 'error' not in grad_attr:
                    grad_importances.append(grad_attr['mean_importance'])
        
        if grad_importances:
            print(f"  Gradient Attribution:")
            print(f"    Average importance: {np.mean(grad_importances):.3f} (±{np.std(grad_importances):.3f})")
            print(f"    Highest importance: {np.max(grad_importances):.3f} at layer {np.argmax(grad_importances)}")
        
        # Print receptive field results
        receptive_2gram_scores = []
        for layer_idx, layer_results in results.items():
            if isinstance(layer_idx, int) and 'receptive_field' in layer_results:
                receptive = layer_results['receptive_field']
                if 'receptive_scores' in receptive and 'error' not in receptive:
                    scores = receptive['receptive_scores']
                    receptive_2gram_scores.append(scores.get(2, 0))
        
        if receptive_2gram_scores:
            print(f"  Receptive Field (2-gram sensitivity):")
            print(f"    Average sensitivity: {np.mean(receptive_2gram_scores):.3f} (±{np.std(receptive_2gram_scores):.3f})")
            print(f"    Highest sensitivity: {np.max(receptive_2gram_scores):.3f} at layer {np.argmax(receptive_2gram_scores)}")
        
        # Print clustering analysis if available
        if 'clustering_analysis' in results:
            clustering = results['clustering_analysis']
            
            # Print clustering metrics
            for ngram_size in [1, 2, 3]:
                if ngram_size in clustering['clustering_metrics']:
                    metrics = clustering['clustering_metrics'][ngram_size]
                    print(f"  {ngram_size}-gram heads: Early/Late ratio = {metrics['early_late_ratio']:.2f} (Total: {metrics['total_heads']} heads)")
            
            # Print head composition summary
            total_heads = sum(len(head_dist) for head_dist in clustering['head_distributions'].values())
            print(f"  Total heads analyzed: {total_heads}")
            
            # Print specialization distribution
            specialization_counts = defaultdict(int)
            for layer_idx, head_dist in clustering['head_distributions'].items():
                for head_id, head_info in head_dist.items():
                    specialization_counts[head_info['specialization']] += 1
            
            print(f"  Head specialization distribution:")
            for ngram_size in [1, 2, 3]:
                count = specialization_counts[ngram_size]
                print(f"    {ngram_size}-gram specialized heads: {count}")
            
            # Print validation results
            if 'validation_metrics' in clustering:
                print(f"  Validation results:")
                for layer_idx, validation_metrics in clustering['validation_metrics'].items():
                    print(f"    Layer {layer_idx}: Mean validation score = {validation_metrics['mean_validation_score']:.3f} "
                          f"(±{validation_metrics['std_validation_score']:.3f})")
            
            # Print method comparison
            if 'method_comparison' in clustering:
                print(f"  Method comparison (Clustering vs Slicing):")
                clustering_validation_scores = []
                slicing_validation_scores = []
                
                for layer_idx, comparison in clustering['method_comparison'].items():
                    clustering_validation_scores.append(comparison['clustering_validation_mean'])
                    slicing_validation_scores.append(comparison['slicing_validation_mean'])
                
                if clustering_validation_scores and slicing_validation_scores:
                    clustering_mean = safe_mean(clustering_validation_scores)
                    slicing_mean = safe_mean(slicing_validation_scores)
                    print(f"    Average validation score - Clustering: {clustering_mean:.3f}, Slicing: {slicing_mean:.3f}")
                    impr = percent_improvement(slicing_mean, clustering_mean)
                    if impr is None:
                        print("    Improvement: N/A (division by zero in baseline)")
                    else:
                        print(f"    Improvement: {impr:+.1f}%")
        
        # Print causal test results for Mamba models
        if 'causal_test' in results and 'error' not in results['causal_test']:
            print(f"  Causal Test (State-Kernel Replacement):")
            causal_data = results['causal_test']
            for layer_idx, layer_data in causal_data.items():
                if isinstance(layer_data, dict) and '1gram_mass' in layer_data:
                    print(f"    Layer {layer_idx}: 1g={layer_data['1gram_mass']:.4f}, 2g={layer_data['2gram_mass']:.4f}, 3g={layer_data['3gram_mass']:.4f}")
        elif 'causal_test' in results and 'error' in results['causal_test']:
            print(f"  Causal Test: Error - {results['causal_test']['error']}")
        
        # Print kernel interpolation results for Mamba models
        if 'kernel_interpolation' in results and 'error' not in results['kernel_interpolation']:
            print(f"  Kernel Interpolation (λ sweep):")
            interp_data = results['kernel_interpolation']
            if isinstance(interp_data, list):
                for lam, m1, m2, m3 in interp_data:
                    print(f"    λ={lam:.2f} → 1g={m1:.4f}, 2g={m2:.4f}, 3g={m3:.4f}")
        elif 'kernel_interpolation' in results and 'error' in results['kernel_interpolation']:
            print(f"  Kernel Interpolation: Error - {results['kernel_interpolation']['error']}")


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

def linear_decode_accuracy(layer_acts, token_ids, max_tokens=5000):
    """
    Train a linear classifier on layer activations to predict token ids.
    Args:
        layer_acts: np.array [num_samples, hidden_dim]
        token_ids: np.array [num_samples]
        max_tokens: restrict to top-k frequent tokens for efficiency
    Returns:
        accuracy (float), clf (trained classifier)
    """
    # restrict vocab
    unique, counts = np.unique(token_ids, return_counts=True)
    topk = unique[np.argsort(counts)[::-1][:max_tokens]]
    mask = np.isin(token_ids, topk)

    X = layer_acts[mask]
    y = token_ids[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = LogisticRegression(
        multi_class="multinomial", solver="saga", max_iter=200
    )
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)

    return acc, clf

def gradient_importance_mamba(model, tokenizer, texts, layer_module, token_threshold=200):
    """Compute gradient importance for Mamba models with temporary gradient enabling."""
    device = next(model.parameters()).device
    
    # Temporarily enable gradients for this analysis
    original_grad_state = {}
    for name, param in model.named_parameters():
        original_grad_state[name] = param.requires_grad
        param.requires_grad = True
    
    model.train()  # Enable gradient tracking mode
    
    try:
        acts = []
        grads = []

        def hook_fn(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            out.retain_grad()
            acts.append(out)
            grads.append(out)
            return out

        handle = layer_module.register_forward_hook(hook_fn)

        inputs = tokenizer(" ".join(texts), return_tensors="pt", 
                          truncation=True, max_length=128).to(device)
        outputs = model(**inputs)
        
        logits = outputs.logits
        target_ids = inputs["input_ids"]
        
        # Ensure we have valid dimensions for loss computation
        if logits.dim() >= 3 and target_ids.dim() >= 2:
            # Check if sequence length is sufficient for slicing
            seq_len = logits.shape[1]
            if seq_len > 1:
                # Reshape for loss computation
                logits_flat = logits[:, :-1].contiguous().view(-1, logits.size(-1))
                target_flat = target_ids[:, 1:].contiguous().view(-1)
                
                loss = torch.nn.functional.cross_entropy(logits_flat, target_flat)
                loss.backward()
            else:
                print(f"⚠️  Sequence length too short for loss computation: {seq_len}")
                return None
        else:
            print(f"⚠️  Invalid tensor dimensions for loss computation: logits={logits.shape}, targets={target_ids.shape}")
            return None

        handle.remove()
        
        if not grads or grads[0].grad is None:
            return None
        
        g = grads[0].grad.detach().cpu().numpy()
        
        # Handle different tensor shapes correctly with error checking
        try:
            if g.ndim == 1:
                # 1D tensor: just take absolute values
                importance = np.abs(g)
            elif g.ndim == 2:
                # 2D tensor: average over batch dimension (axis 0)
                importance = np.mean(np.abs(g), axis=0)
            elif g.ndim == 3:
                # 3D tensor: average over batch and sequence dimensions (axes 0, 1)
                importance = np.mean(np.abs(g), axis=(0, 1))
            else:
                # Higher dimensions: average over all but last dimension
                importance = np.mean(np.abs(g), axis=tuple(range(g.ndim-1)))
            
            return importance
        except Exception as grad_error:
            print(f"⚠️  Error processing gradients: {grad_error}")
            print(f"    Gradient shape: {g.shape}, dimensions: {g.ndim}")
            return None

    except Exception as e:
        print(f"⚠️  Error in gradient importance computation: {e}")
        return None
    finally:
        # Restore original gradient state
        for name, param in model.named_parameters():
            if name in original_grad_state:
                param.requires_grad = original_grad_state[name]
        model.eval()  # Return to eval mode
        clear_memory()


def gradient_importance_gpt2(model, tokenizer, texts, layer_module, token_threshold=200):
    """Compute gradient importance for GPT-2 models using safe_gradient_importance."""
    device = next(model.parameters()).device
    
    # Store original gradient state
    original_grad_state = {}
    for name, param in model.named_parameters():
        original_grad_state[name] = param.requires_grad
        param.requires_grad = True
    
    # Enable training mode for gradient computation
    model.train()
    
    try:
        # Use the new safe gradient importance function
        inputs = tokenizer(" ".join(texts), return_tensors="pt", 
                          truncation=True, max_length=128).to(device)
        
        # Get layer output using forward hook
        layer_output = None
        def hook_fn(module, inp, out):
            nonlocal layer_output
            if isinstance(out, tuple):
                out = out[0]
            layer_output = out
            return out

        handle = layer_module.register_forward_hook(hook_fn)
        
        # Run forward pass with attention and hidden states output
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
        attn_np = extract_attention_from_outputs(outputs)
        
        if attn_np is None:
            print("⚠️  Could not extract attention weights, falling back to empirical RF probe")
            # fallback to empirical RF probe
            handle.remove()
            return None
        
        handle.remove()
        
        if layer_output is not None:
            # Use safe_gradient_importance for robust computation
            importance_score = safe_gradient_importance(model, inputs, layer_output)
            return importance_score
        else:
            print("⚠️  Could not capture layer output")
            return None

    except Exception as e:
        print(f"⚠️  Error in gradient importance computation: {e}")
        return None
    finally:
        # Restore original gradient state
        for name, param in model.named_parameters():
            if name in original_grad_state:
                param.requires_grad = original_grad_state[name]
        model.eval()  # Return to eval mode
        clear_memory()


def gradient_importance(model, tokenizer, texts, layer_module, token_threshold=200):
    """Wrapper function that chooses the appropriate gradient importance method based on model type."""
    model_name = model.config.name_or_path.lower() if hasattr(model.config, 'name_or_path') else str(type(model)).lower()
    
    if 'mamba' in model_name:
        return gradient_importance_mamba(model, tokenizer, texts, layer_module, token_threshold)
    elif 'gpt' in model_name or 'transformer' in model_name:
        return gradient_importance_gpt2(model, tokenizer, texts, layer_module, token_threshold)
    else:
        # Default to Mamba method for unknown models
        return gradient_importance_mamba(model, tokenizer, texts, layer_module, token_threshold)


def receptive_field_analysis(model, tokenizer, texts, device=None, n_max=3):
    """
    Measure per-layer receptive-field mass for 1-, 2-, and 3-token perturbations.
    Works for both Mamba and GPT-2.
    Returns dict: {layer_idx: {n_size: rf_mass_value}}
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    rf_results = defaultdict(dict)
    # Get number of layers from config, with fallback to counting layers
    num_layers = getattr(model.config, "num_hidden_layers", None)
    if num_layers is None:
        # Fallback: try to count layers from model structure
        from utils import get_model_layers
        layers = get_model_layers(model)
        num_layers = len(layers) if layers else 12
    # Ensure we analyze all layers
    print(f"    Analyzing receptive field for {num_layers} layers")

    # First, check how many hidden_states we actually get
    test_inputs = tokenizer("test", return_tensors="pt", truncation=True, max_length=64).to(device)
    test_outputs = model(**test_inputs, output_hidden_states=True)
    num_hidden_states = len(test_outputs.hidden_states) if test_outputs.hidden_states else num_layers + 1
    print(f"    Model returns {num_hidden_states} hidden states (expected {num_layers} layers + 1 embedding)")
    
    # hidden_states[0] is typically embeddings, hidden_states[1:] are layer outputs
    # So layer_idx 0 corresponds to hidden_states[1], layer_idx 1 to hidden_states[2], etc.
    # We need to check if hidden_states includes embeddings (length = num_layers + 1) or not (length = num_layers)
    has_embedding = (num_hidden_states == num_layers + 1)
    hidden_state_offset = 1 if has_embedding else 0
    print(f"    Using hidden_state offset: {hidden_state_offset} (has_embedding={has_embedding})")

    for layer_idx in range(num_layers):
        layer_rf = {}

        # Run with gradient enabled for GPT-2 to capture hidden deltas
        with torch.enable_grad():
            for text in texts[:5]:
                inputs = tokenizer(text, return_tensors="pt",
                                   truncation=True, max_length=64).to(device)

                # baseline hidden
                outputs = model(**inputs, output_hidden_states=True)
                # Map layer_idx to hidden_states index
                hidden_state_idx = layer_idx + hidden_state_offset
                if hidden_state_idx >= len(outputs.hidden_states):
                    print(f"    Warning: layer {layer_idx} -> hidden_state index {hidden_state_idx} out of range (max: {len(outputs.hidden_states)-1})")
                    continue
                hidden_base = outputs.hidden_states[hidden_state_idx]  # [B, T, H]

                rf_mass_per_n = []
                for n in range(1, n_max + 1):
                    if inputs.input_ids.shape[1] <= n:
                        continue
                    perturbed = inputs.input_ids.clone()
                    # zero-out last n tokens
                    perturbed[:, -n:] = tokenizer.pad_token_id or 0
                    perturbed_out = model(input_ids=perturbed,
                                          output_hidden_states=True)
                    if hidden_state_idx >= len(perturbed_out.hidden_states):
                        continue
                    hidden_perturbed = perturbed_out.hidden_states[hidden_state_idx]

                    diff = (hidden_base - hidden_perturbed).abs()
                    rel_diff = diff / (hidden_base.abs() + 1e-8)
                    # mean over batch, seq, and feature dims
                    rf_mass = rel_diff.mean().item()
                    rf_mass_per_n.append(rf_mass)

                    # store average per n size
                    layer_rf[n] = np.mean(rf_mass_per_n)

        rf_results[layer_idx] = layer_rf

    return rf_results


def compute_receptive_field_mass(model, tokenizer, texts, layer_idx, n_tokens_list=[1, 2, 3], 
                                max_sequences=500, max_length=128, device=None):
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    results = {}
    texts = texts[:max_sequences]
    
    for n_tokens in n_tokens_list:
        print(f"    Computing receptive field mass for {n_tokens}-tokens...")
        all_scores = []
        
        def hook_fn(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            if not hasattr(hook_fn, 'outputs'):
                hook_fn.outputs = []
            hook_fn.outputs.append(out.detach().cpu())
            return out
        
        layers = get_model_layers(model)
        handle = layers[layer_idx].register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
                processed_count = 0
                for text in texts:
                    try:
                        inputs = tokenizer(text, return_tensors="pt", 
                                         truncation=True, max_length=max_length).to(device)
                        input_ids = inputs["input_ids"]
                        seq_len = input_ids.shape[1]
                        
                        if seq_len <= n_tokens:
                            print(f"[DEBUG] Skipping text with seq_len={seq_len} <= n_tokens={n_tokens}")
                            continue
                        
                        # Baseline forward pass
                        hook_fn.outputs = []
                        _ = model(**inputs)
                        
                        if not hook_fn.outputs:
                            continue
                            
                        baseline_output = hook_fn.outputs[0]
                        
                        # Truncated forward pass
                        truncated_ids = input_ids[:, :-n_tokens]
                        hook_fn.outputs = []
                        truncated_inputs = {"input_ids": truncated_ids}
                        _ = model(**truncated_inputs)
                        
                        if not hook_fn.outputs:
                            continue
                            
                        truncated_output = hook_fn.outputs[0]
                        
                        # Debug tensor information
                        print(f"[DEBUG] Text processing: seq_len={seq_len}, n_tokens={n_tokens}")
                        print(f"[DEBUG] Baseline shape: {baseline_output.shape}, Truncated shape: {truncated_output.shape}")
                        print(f"[DEBUG] Baseline mean: {baseline_output.mean().item():.8f}, Truncated mean: {truncated_output.mean().item():.8f}")
                        
                        # Calculate difference based on tensor dimensions
                        if baseline_output.dim() == 3 and truncated_output.dim() == 3:
                            # [batch, seq_len, hidden_dim]
                            # Compare the LAST position of baseline with the LAST position of truncated
                            # This shows how much the removed tokens affect the final representation
                            baseline_last = baseline_output[:, -1, :]  # [batch, hidden_dim]
                            truncated_last = truncated_output[:, -1, :]  # [batch, hidden_dim]
                            attention_mass = torch.mean(torch.abs(baseline_last - truncated_last)).item()
                            all_scores.append(attention_mass)
                            processed_count += 1
                            print(f"[DEBUG] 3D tensors: baseline {baseline_output.shape}, truncated {truncated_output.shape}, mass={attention_mass:.8f}")
                        elif baseline_output.dim() == 2 and truncated_output.dim() == 2:
                            # [batch, hidden_dim]
                            attention_mass = torch.mean(torch.abs(baseline_output - truncated_output)).item()
                            all_scores.append(attention_mass)
                            processed_count += 1
                            print(f"[DEBUG] 2D tensors: baseline {baseline_output.shape}, truncated {truncated_output.shape}, mass={attention_mass:.8f}")
                        else:
                            print(f"[DEBUG] Dimension mismatch: baseline {baseline_output.shape}, truncated {truncated_output.shape}")
                        
                    except Exception as e:
                        print(f"[DEBUG] Error processing text: {e}")
                        continue
                
                print(f"[DEBUG] Processed {processed_count} texts out of {len(texts)} for {n_tokens}-tokens")
        
        finally:
            handle.remove()
        
        if all_scores:
            results[n_tokens] = {
                'mean': float(np.mean(all_scores)),
                'p90': float(np.percentile(all_scores, 90)),
                'p75': float(np.percentile(all_scores, 75)),
                'p50': float(np.percentile(all_scores, 50)),
                'std': float(np.std(all_scores)),
                'all_scores': all_scores,
                'num_samples': len(all_scores)
            }
            print(f"[DEBUG] {n_tokens}-token mass: {len(all_scores)} samples, mean={np.mean(all_scores):.6f}")
        else:
            results[n_tokens] = {
                'mean': 0.0, 'p90': 0.0, 'p75': 0.0, 'p50': 0.0, 'std': 0.0,
                'all_scores': [], 'num_samples': 0
            }
            print(f"[DEBUG] {n_tokens}-token mass: NO SAMPLES PROCESSED - returning 0.0")
    
    return results

def analyze_receptive_field_mass_all_layers(model, tokenizer, texts, model_label, 
                                          layers_to_analyze=None, max_sequences=500):
    """
    Run receptive field mass analysis across all layers.
    
    Returns:
        dict: {layer_idx: {n_tokens: {'mean': float, 'p90': float, ...}}}
    """
    if layers_to_analyze is None:
        layers_to_analyze = list(range(getattr(model.config, "num_hidden_layers", 6)))
    
    print(f"🔍 Running Priority 1: Receptive Field Mass Analysis for {model_label}")
    print(f"  Analyzing {len(layers_to_analyze)} layers: {layers_to_analyze}")
    
    all_results = {}
    
    for layer_idx in layers_to_analyze:
        print(f"  Analyzing layer {layer_idx}/{max(layers_to_analyze)}")
        
        try:
            layer_results = compute_receptive_field_mass(
                model, tokenizer, texts, layer_idx, 
                n_tokens_list=[1, 2, 3], 
                max_sequences=max_sequences
            )
            all_results[layer_idx] = layer_results
            
            # Print summary for this layer
            for n_tokens in [1, 2, 3]:
                if n_tokens in layer_results:
                    mean_val = layer_results[n_tokens]['mean']
                    p90_val = layer_results[n_tokens]['p90']
                    print(f"    {n_tokens}-token mass: mean={mean_val:.4f}, p90={p90_val:.4f}")
        
        except Exception as e:
            print(f"⚠️  Error analyzing layer {layer_idx}: {e}")
            all_results[layer_idx] = {}
    
    return all_results


def enhanced_linear_token_decode(model, tokenizer, texts, layer_idx, model_label, 
                                top_k=5000, max_positions=50000, max_length=128):
    """
    Priority 2: Enhanced linear token decode analysis.
    
    Tests whether token identity is linearly present in layer activations.
    Distinguishes representation→readout shift vs distributed low-amplitude signals.
    
    Args:
        model: The model to analyze
        tokenizer: Tokenizer for the model
        texts: List of input texts
        layer_idx: Layer index to analyze
        model_label: Model label for threshold selection
        top_k: Number of top frequent tokens to use
        max_positions: Maximum number of positions to sample
        max_length: Maximum sequence length
    
    Returns:
        dict: {'accuracy': float, 'num_samples': int, 'vocab_size': int, 'classifier_coef_shape': tuple}
    """
    print(f"    Running enhanced linear token decode for layer {layer_idx}...")
    
    device = next(model.parameters()).device
    model.eval()
    
    # Collect activations and token IDs
    activations = []
    token_ids = []
    
    def hook_fn(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        acts = out.detach().cpu().numpy()
        activations.append(acts)
    
    layers = get_model_layers(model)
    handle = layers[layer_idx].register_forward_hook(hook_fn)
    
    try:
        with torch.no_grad():
            positions_collected = 0
            for text in texts:
                if positions_collected >= max_positions:
                    break
                    
                try:
                    inputs = tokenizer(text, return_tensors="pt", 
                                     truncation=True, max_length=max_length, padding=True).to(device)
                    current_token_ids = inputs["input_ids"].cpu().numpy().flatten()
                    
                    # Limit positions to avoid exceeding max_positions
                    remaining_positions = max_positions - positions_collected
                    if len(current_token_ids) > remaining_positions:
                        current_token_ids = current_token_ids[:remaining_positions]
                    
                    token_ids.extend(current_token_ids)
                    _ = model(**inputs)
                    positions_collected += len(current_token_ids)
                    
                except Exception as e:
                    print(f"⚠️  Error processing text: {e}")
                    continue
    
    finally:
        handle.remove()
    
    # Concatenate activations, handling different sequence lengths
    if not activations:
        return {"accuracy": 0.0, "error": "No activations collected"}
    
    # Find the minimum sequence length to ensure compatibility
    min_seq_len = min(acts.shape[1] if acts.ndim >= 2 else acts.shape[0] for acts in activations if acts.ndim >= 1)
    
    # Truncate all activations to the same length
    truncated_activations = []
    for acts in activations:
        if acts.ndim == 3:  # [batch, seq_len, hidden_dim]
            truncated = acts[:, :min_seq_len, :]
        elif acts.ndim == 2:  # [seq_len, hidden_dim] or [batch, hidden_dim]
            if acts.shape[0] > min_seq_len:
                truncated = acts[:min_seq_len, :]
            else:
                truncated = acts
        elif acts.ndim == 1:  # [hidden_dim] - treat as single sample
            truncated = acts.reshape(1, -1)
        else:
            continue  # Skip unexpected shapes
        truncated_activations.append(truncated)
    
    if not truncated_activations:
        return {"accuracy": 0.0, "error": "No valid activations after truncation"}
    
    try:
        all_acts = np.concatenate(truncated_activations, axis=0)
        all_acts = all_acts.reshape(all_acts.shape[0], -1)  # Flatten to [samples, features]
    except Exception as e:
        print(f"⚠️  Error concatenating truncated activations: {e}")
        return {"accuracy": 0.0, "error": f"Activation concatenation failed: {e}"}
    
    # Ensure we have matching number of samples
    min_samples = min(len(all_acts), len(token_ids))
    all_acts = all_acts[:min_samples]
    token_ids = np.array(token_ids[:min_samples])
    
    # Restrict to top-K frequent tokens
    unique, counts = np.unique(token_ids, return_counts=True)
    topk_tokens = unique[np.argsort(counts)[::-1][:top_k]]
    mask = np.isin(token_ids, topk_tokens)
    
    X = all_acts[mask]
    y = token_ids[mask]
    
    if len(X) < 100:  # Need minimum samples
        return {"accuracy": 0.0, "error": f"Insufficient samples: {len(X)}"}
    
    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train logistic regression
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(
        multi_class="multinomial", 
        solver="saga", 
        max_iter=200,
        random_state=42
    )
    
    try:
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        
        return {
            "accuracy": float(accuracy),
            "num_samples": len(X),
            "vocab_size": len(np.unique(y)),
            "classifier_coef_shape": clf.coef_.shape if hasattr(clf, 'coef_') else None,
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }
        
    except Exception as e:
        return {"accuracy": 0.0, "error": f"Training failed: {e}"}


def receptive_field_probe(model, tokenizer, text, layer_module, max_ngram=3):
    """
    Empirically probe how much the layer depends on last n tokens.
    Args:
        model, tokenizer
        text: str
        layer_module: nn.Module (layer to probe)
        max_ngram: up to how many tokens to test
    Returns:
        dict: ngram_size -> sensitivity score
    """
    device = next(model.parameters()).device
    model.eval()
    
    base_inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    
    activations = {}
    
    def hook_fn(module, inp, out):
        # Handle tuple outputs (common in some models)
        if isinstance(out, tuple):
            out = out[0]  # Take the first element (usually the hidden states)
        activations['base'] = out.detach().cpu().numpy()
    h = layer_module.register_forward_hook(hook_fn)
    _ = model(**base_inputs)
    h.remove()
    
    base_act = activations['base']
    
    scores = {}
    for n in range(1, max_ngram+1):
        mod_inputs = base_inputs.copy()
        # replace last n tokens with [UNK] or random token
        mod_inputs["input_ids"] = base_inputs["input_ids"].clone()
        mod_inputs["input_ids"][:, -n:] = tokenizer.unk_token_id
        
        activations = {}
        h = layer_module.register_forward_hook(hook_fn)
        _ = model(**mod_inputs)
        h.remove()
        
        diff = np.mean(np.abs(base_act - activations['base']))
        scores[n] = diff
    
    return scores


def threshold_sweep_analysis(model, tokenizer, texts, layer_idx, model_label, 
                            thresholds=[0.05, 0.1, 0.2, 0.4, 0.6], max_texts=200):
    """
    Priority 3: Threshold sweep and activation magnitude profile.
    
    Directly tests whether thresholding hides early detectors (distributed → sparse).
    For each layer count detected neurons at different thresholds and compute statistics.
    
    Args:
        model: The model to analyze
        tokenizer: Tokenizer for the model
        texts: List of input texts
        layer_idx: Layer index to analyze
        model_label: Model label for threshold selection
        thresholds: List of thresholds to test
        max_texts: Maximum number of texts to process
    
    Returns:
        dict: {'threshold_counts': dict, 'activation_stats': dict, 'magnitude_profile': dict}
    """
    print(f"    Running threshold sweep for layer {layer_idx}...")
    
    device = next(model.parameters()).device
    model.eval()
    
    # Determine activation threshold based on model
    if model_label == "GPT-2":
        base_threshold = GPT2_ACTIVATION_THRESHOLD
    elif "Mamba" in model_label:
        base_threshold = MAMBA_ACTIVATION_THRESHOLD
    else:
        base_threshold = OTHER_MODELS_ACTIVATION_THRESHOLD
    
    all_activations = []
    
    def hook_fn(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        acts = out.detach().cpu().numpy()
        all_activations.append(acts)
    
    layers = get_model_layers(model)
    handle = layers[layer_idx].register_forward_hook(hook_fn)
    
    try:
        with torch.no_grad():
            for text in texts[:max_texts]:
                try:
                    inputs = tokenizer(text, return_tensors="pt", 
                                     truncation=True, max_length=128).to(device)
                    _ = model(**inputs)
                except Exception as e:
                    print(f"⚠️  Error processing text: {e}")
                    continue
    
    finally:
        handle.remove()
    
    # Concatenate all activations, handling different sequence lengths
    if not all_activations:
        return {"threshold_counts": {}, "activation_stats": {}, "magnitude_profile": {}}
    
    # Find the minimum sequence length to ensure compatibility
    min_seq_len = min(acts.shape[1] if acts.ndim >= 2 else acts.shape[0] for acts in all_activations if acts.ndim >= 1)
    
    # Truncate all activations to the same length
    truncated_activations = []
    for acts in all_activations:
        if acts.ndim == 3:  # [batch, seq_len, hidden_dim]
            truncated = acts[:, :min_seq_len, :]
        elif acts.ndim == 2:  # [seq_len, hidden_dim] or [batch, hidden_dim]
            if acts.shape[0] > min_seq_len:
                truncated = acts[:min_seq_len, :]
            else:
                truncated = acts
        elif acts.ndim == 1:  # [hidden_dim] - treat as single sample
            truncated = acts.reshape(1, -1)
        else:
            continue  # Skip unexpected shapes
        truncated_activations.append(truncated)
    
    if not truncated_activations:
        return {"threshold_counts": {}, "activation_stats": {}, "magnitude_profile": {}}
    
    try:
        all_acts = np.concatenate(truncated_activations, axis=0)
    except Exception as e:
        print(f"⚠️  Error concatenating activations in threshold sweep: {e}")
        return {"threshold_counts": {}, "activation_stats": {}, "magnitude_profile": {}}
    
    # Handle different tensor dimensions
    if all_acts.ndim == 3:  # [batch, seq_len, hidden_dim]
        all_acts = all_acts.reshape(-1, all_acts.shape[-1])  # Flatten to [positions, hidden_dim]
    elif all_acts.ndim == 2:  # [batch, hidden_dim]
        pass  # Already in correct format
    else:
        print(f"⚠️  Unexpected activation shape: {all_acts.shape}")
        return {"threshold_counts": {}, "activation_stats": {}, "magnitude_profile": {}}
    
    # Compute threshold counts
    threshold_counts = {}
    for threshold in thresholds:
        # Count neurons that exceed threshold at any position
        max_acts_per_neuron = np.max(all_acts, axis=0)  # Max activation per neuron across all positions
        active_neurons = np.sum(max_acts_per_neuron > threshold)
        threshold_counts[threshold] = int(active_neurons)
    
    # Compute activation statistics
    activation_stats = {
        'mean_activation': float(np.mean(all_acts)),
        'std_activation': float(np.std(all_acts)),
        'max_activation': float(np.max(all_acts)),
        'min_activation': float(np.min(all_acts)),
        'fraction_nonzero': float(np.mean(all_acts != 0)),
        'l2_norm_mean': float(np.mean(np.linalg.norm(all_acts, axis=1))),
        'l2_norm_std': float(np.std(np.linalg.norm(all_acts, axis=1)))
    }
    
    # Compute magnitude profile (per-neuron statistics)
    neuron_means = np.mean(all_acts, axis=0)
    neuron_stds = np.std(all_acts, axis=0)
    neuron_maxs = np.max(all_acts, axis=0)
    
    magnitude_profile = {
        'neuron_mean_mean': float(np.mean(neuron_means)),
        'neuron_mean_std': float(np.std(neuron_means)),
        'neuron_std_mean': float(np.mean(neuron_stds)),
        'neuron_std_std': float(np.std(neuron_stds)),
        'neuron_max_mean': float(np.mean(neuron_maxs)),
        'neuron_max_std': float(np.std(neuron_maxs)),
        'num_neurons': len(neuron_means)
    }
    
    return {
        'threshold_counts': threshold_counts,
        'activation_stats': activation_stats,
        'magnitude_profile': magnitude_profile,
        'base_threshold': base_threshold
    }


def analyze_threshold_sweep_all_layers(model, tokenizer, texts, model_label, 
                                     layers_to_analyze=None, thresholds=[0.05, 0.1, 0.2, 0.4, 0.6]):
    """
    Run threshold sweep analysis across all layers.
    
    Returns:
        dict: {layer_idx: {'threshold_counts': dict, 'activation_stats': dict, 'magnitude_profile': dict}}
    """
    if layers_to_analyze is None:
        layers_to_analyze = list(range(getattr(model.config, "num_hidden_layers", 6)))
    
    print(f"🔍 Running Priority 3: Threshold Sweep Analysis for {model_label}")
    print(f"  Analyzing {len(layers_to_analyze)} layers: {layers_to_analyze}")
    print(f"  Thresholds: {thresholds}")
    
    all_results = {}
    
    for layer_idx in layers_to_analyze:
        print(f"  Analyzing layer {layer_idx}/{max(layers_to_analyze)}")
        
        try:
            layer_results = threshold_sweep_analysis(
                model, tokenizer, texts, layer_idx, model_label, 
                thresholds=thresholds
            )
            all_results[layer_idx] = layer_results
            
            # Print summary for this layer
            threshold_counts = layer_results.get('threshold_counts', {})
            activation_stats = layer_results.get('activation_stats', {})
            
            print(f"    Threshold counts: {threshold_counts}")
            print(f"    Mean activation: {activation_stats.get('mean_activation', 0):.4f}")
            print(f"    Fraction nonzero: {activation_stats.get('fraction_nonzero', 0):.4f}")
        
        except Exception as e:
            print(f"⚠️  Error analyzing layer {layer_idx}: {e}")
            all_results[layer_idx] = {}
    
    return all_results


def neuron_vs_receptive_field(neuron_triggers, receptive_scores, n_max=3):
    """
    Compare neuron-level ngram triggers with receptive field probe.
    Args:
        neuron_triggers: dict neuron_idx -> {ngram_size -> set of triggers}
        receptive_scores: dict ngram_size -> float
    Returns:
        dict summarizing alignment
    """
    neuron_counts = {n:0 for n in range(1, n_max+1)}
    for neuron_idx, triggers in neuron_triggers.items():
        # pick dominant ngram size
        counts = {ng: len(ngrams) for ng, ngrams in triggers.items()}
        if counts:
            dom = max(counts, key=counts.get)
            neuron_counts[dom] += 1
    
    return {
        "receptive_scores": receptive_scores,
        "neuron_counts": neuron_counts
    }



if __name__ == "__main__":
    # quick test (run before full pipeline)
    try:
        from main import setup_model_and_tokenizer
    except ImportError:
        print("main.py not found; please place this script with main.py")
        exit(1)
    
    # Load GPT-2 model for testing
    model, tokenizer = setup_model_and_tokenizer("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); model.eval()
    texts = ["The quick brown fox jumps over the lazy dog.", "The cat sat on the mat."]
    inputs = tokenizer(texts[0], return_tensors="pt", truncation=True, max_length=64).to(device)

    layer_idx = 0  # test first block
    print(">>> Testing compute_gradient_attribution_gpt2 ...")
    imp = compute_gradient_attribution_gpt2(model, tokenizer, inputs, layer_idx, debug=True)
    print("Gradient importance (should be >0):", imp)

    print(">>> Testing validate_head_specialization_with_ablation_gpt2 ...")
    val = validate_head_specialization_with_ablation_gpt2(model, tokenizer, texts, layer_idx, head_neurons=[0,1,2], device=device, debug=True)
    print("Validation result (should be >0 if neurons valid):", val)
    
    # Run kernel interpolation experiment
    run_kernel_interpolation_experiment()
    
    run_simple_extended_analysis()
