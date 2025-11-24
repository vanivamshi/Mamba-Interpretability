import sys
import os

# Add parent directory to path to allow importing from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speciality_neurons.specialty_neurons import find_specialty_neurons_fixed
from knowledge_neurons.knowledge_neurons import KnowledgeNeuronsFinder
from dead_neurons.neuron_characterization import find_dead_neurons
from causal_neurons.causal_analysis import find_causal_neurons_fixed

import torch
import numpy as np
from tqdm import tqdm
from utils import get_model_layers

def find_specialty_neurons_all_layers(model, tokenizer, texts, threshold=0.95, top_k=100):
    """
    Find specialty neurons for ALL layers.
    Returns {layer_idx: [specialty_scores]}
    """
    results = {}
    layers = get_model_layers(model)
    if layers is None:
        return {}

    for layer_idx, layer in enumerate(layers):
        print(f"Finding specialty neurons for layer {layer_idx}...")
        # find_specialty_neurons_fixed is designed to work for a specific layer
        specialty_scores = find_specialty_neurons_fixed(model, tokenizer, texts, layer_idx=layer_idx, top_k=top_k)
        results[layer_idx] = specialty_scores
    return results

def find_knowledge_neurons_all_layers(model, tokenizer, fact_data, batch_size=32):
    """
    Find knowledge neurons for ALL layers.
    Returns {layer_idx: [knowledge_scores]}
    """
    results = {}
    layers = get_model_layers(model)
    if layers is None:
        return {}

    finder = KnowledgeNeuronsFinder(model, tokenizer, fact_data, batch_size)
    for layer_idx, layer in enumerate(layers):
        print(f"Finding knowledge neurons for layer {layer_idx}...")
        knowledge_scores = finder.find_knowledge_neurons_for_layer(layer_idx)
        results[layer_idx] = knowledge_scores
    return results

def find_dead_neurons_all_layers(model, tokenizer, texts, activation_threshold=1e-6):
    """
    Find dead neurons for ALL layers.
    Returns {layer_idx: [is_dead_flags]}
    """
    dead_indices = {}
    activation_freqs = {}
    layers = get_model_layers(model)
    if layers is None:
        return {}, {}

    for layer_idx, layer in enumerate(layers):
        print(f"Finding dead neurons for layer {layer_idx}...")
        try:
            # find_dead_neurons returns (dead_neurons_list, activation_freq_array)
            dead, freqs = find_dead_neurons(model, tokenizer, texts, layer_idx, activation_threshold)
            dead_indices[layer_idx] = dead
            activation_freqs[layer_idx] = freqs.tolist() if isinstance(freqs, np.ndarray) else freqs
        except Exception as e:
            print(f"Error in layer {layer_idx}: {e}")
            
    return dead_indices, activation_freqs

def find_causal_neurons_all_layers(model, tokenizer, prompts, target_token_id, k=10):
    """
    Find causal neurons for ALL layers.
    Returns {layer_idx: [causal_scores]}
    """
    results = {}
    layers = get_model_layers(model)
    if layers is None:
        return {}

    for layer_idx, layer in enumerate(layers):
        print(f"Finding causal neurons for layer {layer_idx}...")
        # find_causal_neurons_fixed is designed to work for a specific layer
        causal_scores = find_causal_neurons_fixed(model, tokenizer, prompts, layer_idx=layer_idx, top_k=k)
        results[layer_idx] = causal_scores
    return results

def analyze_universality_all_layers(model, tokenizer, tasks):
    """
    Analyze universality across different tasks for ALL layers efficiently.
    Returns a dict {layer_idx: [scores_ordered_by_neuron_idx]}
    """
    task_activations = {} # {layer_idx: {task_name: [activations]}}
    
    # Initialize structure
    num_layers = 0
    
    for task_name, task_prompts in tasks.items():
        print(f"Processing task: {task_name}")
        
        for prompt in task_prompts:
            try:
                inputs = tokenizer(prompt, return_tensors="pt")
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                
                hidden_states = outputs.hidden_states
                if num_layers == 0:
                    num_layers = len(hidden_states)
                    for i in range(num_layers):
                        task_activations[i] = {}
                
                for layer_idx, layer_act in enumerate(hidden_states):
                    # Average over sequence length
                    avg_activation = layer_act.mean(dim=1).squeeze(0).cpu().numpy()
                    
                    if task_name not in task_activations[layer_idx]:
                        task_activations[layer_idx][task_name] = []
                    task_activations[layer_idx][task_name].append(avg_activation)
                    
            except Exception as e:
                print(f"Error processing prompt '{prompt[:30]}...': {e}")

    # Compute scores for each layer
    results = {}
    
    for layer_idx in range(num_layers):
        if layer_idx not in task_activations:
            continue
            
        layer_task_acts = task_activations[layer_idx]
        if not layer_task_acts:
            continue
            
        # Stack activations
        processed_task_acts = {}
        for t_name, acts in layer_task_acts.items():
            if acts:
                processed_task_acts[t_name] = np.stack(acts)
        
        if not processed_task_acts:
            continue
            
        # Calculate scores
        universal_scores = []
        # Assuming all tasks have same hidden size
        min_size = min(act.shape[-1] for act in processed_task_acts.values())
        
        for dim in range(min_size):
            task_means = []
            for t_name, acts in processed_task_acts.items():
                dim_mean = np.mean(acts[:, dim])
                task_means.append(dim_mean)
            
            mean_activation = np.mean(task_means)
            activation_variance = np.var(task_means)
            universality_score = mean_activation / (1 + activation_variance)
            universal_scores.append(universality_score)
            
        results[layer_idx] = universal_scores
        
    return results

def extract_deltas_all_layers(model, input_ids):
    """
    Extract delta parameters for ALL layers in one pass.
    """
    device = input_ids.device
    layers = get_model_layers(model)
    if layers is None:
        return {}
        
    delta_values = {} # {layer_idx: delta_tensor}
    handles = []
    
    def get_hook(layer_idx):
        def delta_hook(module, input, output):
            if isinstance(output, tuple):
                val = output[0].detach() if len(output) > 0 else torch.randn(1, 1, 512, device=device)
            else:
                val = output.detach()
            delta_values[layer_idx] = val
        return delta_hook

    # Register hooks
    possible_delta_modules = [
        lambda l: l.mixer.compute_delta if hasattr(l, 'mixer') and hasattr(l.mixer, 'compute_delta') else None,
        lambda l: l.ssm.compute_delta if hasattr(l, 'ssm') and hasattr(l.ssm, 'compute_delta') else None,
        lambda l: l.mixer.ssm.compute_delta if hasattr(l, 'mixer') and hasattr(l.mixer, 'ssm') and hasattr(l.mixer.ssm, 'compute_delta') else None,
        lambda l: l.mixer.dt_proj if hasattr(l, 'mixer') and hasattr(l.mixer, 'dt_proj') else None,
        lambda l: l.mixer if hasattr(l, 'mixer') else None,
    ]
    
    for i, layer in enumerate(layers):
        hook_registered = False
        for module_fn in possible_delta_modules:
            try:
                delta_module = module_fn(layer)
                if delta_module is not None:
                    h = delta_module.register_forward_hook(get_hook(i))
                    handles.append(h)
                    hook_registered = True
                    break
            except AttributeError:
                continue
        if not hook_registered:
            # Fallback
            h = layer.register_forward_hook(get_hook(i))
            handles.append(h)

    # Forward pass
    try:
        with torch.no_grad():
            _ = model(input_ids)
    finally:
        for h in handles:
            h.remove()
            
    return delta_values

def find_delta_sensitive_neurons_all_layers(model, tokenizer, texts):
    """
    Find delta sensitive neurons for ALL layers.
    Returns {layer_idx: [variances]}
    """
    layer_deltas = {} # {layer_idx: [deltas_for_each_text]}
    device = next(model.parameters()).device
    
    print(f"Extracting deltas for {len(texts)} texts...")
    for idx, text in enumerate(texts):
        if (idx+1) % 10 == 0:
            print(f"  Processed {idx+1}/{len(texts)} texts")
            
        try:
            input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
            deltas_dict = extract_deltas_all_layers(model, input_ids)
            
            for layer_idx, delta in deltas_dict.items():
                if layer_idx not in layer_deltas:
                    layer_deltas[layer_idx] = []
                
                # Process delta (mean over seq)
                if delta.dim() == 3:
                    delta_mean = delta.mean(dim=(0, 1))
                elif delta.dim() == 2:
                    delta_mean = delta.mean(dim=0)
                else:
                    delta_mean = delta
                
                layer_deltas[layer_idx].append(delta_mean.cpu().numpy())
                
        except Exception as e:
            print(f"Error processing text: {e}")

    results = {}
    print("Computing variances...")
    for layer_idx, deltas in layer_deltas.items():
        if not deltas:
            continue
        
        all_deltas = np.array(deltas)
        if all_deltas.ndim == 1:
            all_deltas = all_deltas.reshape(1, -1)
            
        variance = np.var(all_deltas, axis=0)
        results[layer_idx] = variance.tolist()
        
    return results

def find_projection_dominant_neurons_all_layers(model):
    """
    Find neurons with dominant projection weights for ALL layers.
    Returns {layer_idx: [magnitudes]}
    """
    results = {}
    layers = get_model_layers(model)
    
    if layers is None:
        return {}
        
    # Search for projections in known places
    possible_projections = [
        # Mamba
        lambda l: l.mixer.x_proj.weight if hasattr(l, 'mixer') and hasattr(l.mixer, 'x_proj') and hasattr(l.mixer.x_proj, 'weight') else None,
        lambda l: l.mixer.in_proj.weight if hasattr(l, 'mixer') and hasattr(l.mixer, 'in_proj') and hasattr(l.mixer.in_proj, 'weight') else None,
        lambda l: l.mixer.dt_proj.weight if hasattr(l, 'mixer') and hasattr(l.mixer, 'dt_proj') and hasattr(l.mixer.dt_proj, 'weight') else None,
        lambda l: l.in_proj.weight if hasattr(l, 'in_proj') and hasattr(l.in_proj, 'weight') else None,
        lambda l: l.linear.weight if hasattr(l, 'linear') and hasattr(l.linear, 'weight') else None,
        # Transformers
        lambda l: l.attn.c_proj.weight if hasattr(l, 'attn') and hasattr(l.attn, 'c_proj') and hasattr(l.attn.c_proj, 'weight') else None,
        lambda l: l.mlp.c_proj.weight if hasattr(l, 'mlp') and hasattr(l.mlp, 'c_proj') and hasattr(l.mlp.c_proj, 'weight') else None,
    ]
    
    for i, layer in enumerate(layers):
        projection_weights = None
        for proj_fn in possible_projections:
            try:
                weights = proj_fn(layer)
                if weights is not None:
                    projection_weights = weights.detach().cpu().numpy()
                    break
            except AttributeError:
                continue
                
        if projection_weights is not None:
            # Compute magnitude per neuron
            if projection_weights.ndim == 2:
                # Shape: (output_dim, input_dim) - we want input_dim importance usually?
                # Or output_dim? The original code did: np.linalg.norm(projection_weights, axis=0)
                # which reduces along axis 0 (output_dim), resulting in input_dim size.
                magnitudes = np.linalg.norm(projection_weights, axis=0)
            else:
                magnitudes = np.abs(projection_weights)
                
            results[i] = magnitudes.tolist()
            
    return results



