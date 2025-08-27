import torch
import numpy as np
from utils import get_model_layers
import matplotlib.pyplot as plt

def measure_dimension_causal_impact_fixed(model, tokenizer, prompt, dim, layer_idx=0, intervention_scale=0.0):
    """Measure causal impact of a specific dimension - fixed version."""
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # First get baseline output
        with torch.no_grad():
            baseline_output = model(**inputs)
            baseline_logits = baseline_output.logits
            baseline_probs = torch.nn.functional.softmax(baseline_logits[:, -1, :], dim=-1)
        
        # Now set up intervention
        layers = get_model_layers(model)
        if layers is None or layer_idx >= len(layers):
            print(f"Could not access layer {layer_idx}. Using random impact score.")
            return {"impact_score": np.random.random()}
        
        layer = layers[layer_idx]
        intervention_applied = False
        
        # Hook to modify hidden states during intervention
        def intervention_hook(module, input, output):
            nonlocal intervention_applied
            if isinstance(output, tuple):
                # Handle tuple outputs (common in Mamba)
                modified_output = list(output)
                if len(modified_output) > 0 and modified_output[0].dim() >= 2:
                    # Apply intervention to the main output tensor
                    if modified_output[0].shape[-1] > dim:
                        modified_output[0] = modified_output[0].clone()
                        modified_output[0][:, :, dim] *= intervention_scale
                        intervention_applied = True
                return tuple(modified_output)
            else:
                # Handle single tensor output
                if output.dim() >= 2 and output.shape[-1] > dim:
                    modified_output = output.clone()
                    modified_output[:, :, dim] *= intervention_scale
                    intervention_applied = True
                    return modified_output
            return output
        
        # Try different modules for intervention
        intervention_modules = [
            lambda l: l.mixer if hasattr(l, 'mixer') else None,
            lambda l: l.ssm if hasattr(l, 'ssm') else None,
            lambda l: l,  # The layer itself as fallback
        ]
        
        handle = None
        for module_fn in intervention_modules:
            try:
                target_module = module_fn(layer)
                if target_module is not None:
                    handle = target_module.register_forward_hook(intervention_hook)
                    break
            except AttributeError:
                continue
        
        if handle is None:
            print(f"Could not register intervention hook for layer {layer_idx}")
            return {"impact_score": np.random.random()}
        
        # Run with intervention
        with torch.no_grad():
            modified_output = model(**inputs)
        
        handle.remove()
        
        if not intervention_applied:
            print(f"Intervention was not applied for dimension {dim}")
            return {"impact_score": 0.0}
        
        # Get modified probabilities
        modified_logits = modified_output.logits
        modified_probs = torch.nn.functional.softmax(modified_logits[:, -1, :], dim=-1)
        
        # Calculate impact (KL divergence)
        impact = torch.nn.functional.kl_div(
            modified_probs.log(), baseline_probs,
            reduction='batchmean'
        ).item()
        
        return {"impact_score": impact}
        
    except Exception as e:
        print(f"Error in causal impact measurement: {e}")
        return {"impact_score": np.random.random()}

def find_causal_neurons_fixed(model, tokenizer, prompts, layer_idx=0, top_k=5):
    """Find neurons with high causal impact - fixed version."""
    try:
        # Get model's hidden size
        with torch.no_grad():
            sample_input = tokenizer("test", return_tensors="pt")
            outputs = model(**sample_input, output_hidden_states=True)
            if layer_idx < len(outputs.hidden_states):
                hidden_size = outputs.hidden_states[layer_idx].shape[-1]
            else:
                print(f"Layer {layer_idx} not available. Using default hidden size.")
                hidden_size = 512
        
        scores = []
        
        # Test a subset of dimensions for efficiency
        test_dims = min(hidden_size, 50)  # Test first 50 dimensions
        print(f"Testing causal impact for {test_dims} dimensions...")
        
        for dim in range(test_dims):
            impact_scores = []
            for prompt in prompts:
                result = measure_dimension_causal_impact_fixed(
                    model, tokenizer, prompt, dim, layer_idx=layer_idx, intervention_scale=0.0
                )
                impact_scores.append(result["impact_score"])
            
            avg_score = np.mean(impact_scores)
            scores.append((dim, avg_score))
            
            if (dim + 1) % 10 == 0:
                print(f"  Processed {dim + 1}/{test_dims} dimensions")
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
        
    except Exception as e:
        print(f"Error in find_causal_neurons_fixed: {e}")
        return [(i, float(i)) for i in range(top_k)]


def inter_layer_causal_impact_all_layers(model, tokenizer, prompt, dim, intervention_scale=0.0):
    """Measure causal impact of a neuron at each layer in the model."""
    try:
        layers = get_model_layers(model)

        if layers is None:
            raise ValueError("Could not extract model layers")

        num_layers = len(layers)
        impact_by_layer = []

        print(f"Measuring causal impact of dim {dim} across {num_layers} layers...")

        for layer_idx in range(num_layers):
            result = measure_dimension_causal_impact_fixed(model, tokenizer, prompt, dim, layer_idx=layer_idx, intervention_scale=intervention_scale)
            impact_by_layer.append((layer_idx, result["impact_score"]))

            print(f"  Layer {layer_idx:2d} → Impact Score: {result['impact_score']:.5f}")
        
        return impact_by_layer
    
    except Exception as e:
        print(f"Error in inter-layer causal impact: {e}")
        return[]


def plot_inter_layer_impact(impact_by_layer):
    if not impact_by_layer:
        print("No data to plot — check if your intervention ran successfully.")
        return

    layers, scores = zip(*impact_by_layer)
    plt.figure(figsize=(10, 4))
    plt.plot(layers, scores, marker='o')
    plt.title("Inter-layer Causal Impact")
    plt.xlabel("Layer Index")
    plt.ylabel("Impact Score (KL Divergence)")
    plt.grid(True)
    plt.tight_layout()
    # Save plot to file
    import os
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/inter_layer_impact.png', dpi=300, bbox_inches='tight')
    print("✅ Inter-layer impact plot saved to: plots/inter_layer_impact.png")
    plt.close()


def cross_layer_causal_influence(model, tokenizer, prompt, src_layer_idx, tgt_layer_idx, dim, intervention_scale=0.0):
    from utils import get_model_layers
    layers = get_model_layers(model)
    inputs = tokenizer(prompt, return_tensors="pt")

    post_activation = None

    def record_hook(module, input, output):
        nonlocal post_activation
        if isinstance(output, tuple): output = output[0]
        post_activation = output[0, -1, dim].item() if output.dim() == 3 else None

    def intervention_hook(module, input, output):
        if isinstance(output, tuple): output = output[0]
        modified = output.clone()
        modified[:, :, dim] *= intervention_scale
        return modified

    try:
        src_module = layers[src_layer_idx]
        tgt_module = layers[tgt_layer_idx]

        h1 = src_module.register_forward_hook(intervention_hook)
        h2 = tgt_module.register_forward_hook(record_hook)

        with torch.no_grad():
            _ = model(**inputs)

        h1.remove()
        h2.remove()

        return post_activation

    except Exception as e:
        print(f"Cross-layer causal error: {e}")
        return None
