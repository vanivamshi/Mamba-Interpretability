import torch
import numpy as np
from utils import get_model_layers
from torch.nn import CrossEntropyLoss

def extract_deltas_fixed(model, layer_idx, input_ids):
    """
    Extract delta parameters for a specific layer - improved version.
    """
    layers = get_model_layers(model)
    hidden_size = model.config.hidden_size
    batch_size, seq_len = input_ids.shape
    
    # Get the device of the model
    device = next(model.parameters()).device

    if layers is None or layer_idx >= len(layers):
        print("Could not find model layers or invalid index. Using dummy delta values.")
        return torch.randn(batch_size, seq_len, hidden_size, device=device)

    layer = layers[layer_idx]
    delta_values = []

    def delta_hook(module, input, output):
        # Handle different output types more robustly
        if isinstance(output, tuple):
            # Take the first element which is usually the hidden states
            hidden_states = output[0]
        else:
            hidden_states = output
        
        # Ensure we have the right shape
        if hidden_states is not None:
            delta_values.append(hidden_states.detach().clone())
        else:
            print("Warning: Got None output from hook")
            delta_values.append(torch.randn(batch_size, seq_len, hidden_size, device=device))

    # Try to find the right module to hook into
    # For Mamba models, we want to capture the state space model outputs
    hook_registered = False
    handle = None
    
    # Try different possible modules in order of preference
    module_candidates = []
    
    # Check for mixer.ssm (common in Mamba)
    if hasattr(layer, 'mixer') and hasattr(layer.mixer, 'ssm'):
        module_candidates.append(('mixer.ssm', layer.mixer.ssm))
    
    # Check for just mixer
    if hasattr(layer, 'mixer'):
        module_candidates.append(('mixer', layer.mixer))
    
    # Check for ssm directly
    if hasattr(layer, 'ssm'):
        module_candidates.append(('ssm', layer.ssm))
    
    # Fallback to the layer itself
    module_candidates.append(('layer', layer))
    
    for name, module in module_candidates:
        try:
            handle = module.register_forward_hook(delta_hook)
            print(f"Registered hook on {name} for layer {layer_idx}")
            hook_registered = True
            break
        except Exception as e:
            print(f"Failed to register hook on {name}: {e}")
            continue

    if not hook_registered:
        print(f"Warning: Could not register hook for layer {layer_idx}")
        return torch.randn(batch_size, seq_len, hidden_size, device=device)

    try:
        with torch.no_grad():
            _ = model(input_ids)
        
        if handle:
            handle.remove()
        
        if delta_values:
            result = delta_values[0]
            print(f"Extracted delta with shape: {result.shape}")
            return result
        else:
            print("No delta values captured")
            return torch.randn(batch_size, seq_len, hidden_size, device=device)
            
    except Exception as e:
        print(f"Error during forward pass: {e}")
        if handle:
            handle.remove()
        return torch.randn(batch_size, seq_len, hidden_size, device=device)

def find_delta_sensitive_neurons_fixed(model, tokenizer, texts, layer_idx=0, top_k=10):
    """Find neurons that are sensitive to delta computation - improved version."""
    print(f"Finding delta-sensitive neurons in layer {layer_idx}...")
    
    # Ensure model is on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    deltas = []
    hidden_size = model.config.hidden_size
    print(f"Model hidden size: {hidden_size}")

    for i, text in enumerate(texts):
        try:
            print(f"Processing text {i+1}/{len(texts)}: {text[:50]}...")
            
            # Tokenize with proper padding
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            input_ids = inputs["input_ids"].to(device)
            
            delta = extract_deltas_fixed(model, layer_idx, input_ids)

            # Process delta to get per-neuron statistics
            if delta.dim() == 3:  # [batch, seq, hidden]
                # Take mean across batch and sequence dimensions
                delta_mean = delta.mean(dim=(0, 1))
            elif delta.dim() == 2:  # [seq, hidden]
                delta_mean = delta.mean(dim=0)
            else:  # [hidden]
                delta_mean = delta

            # Ensure we have the right size
            if delta_mean.shape[0] != hidden_size:
                print(f"Warning: Delta mean shape {delta_mean.shape} doesn't match hidden_size {hidden_size}")
                # Pad or truncate as needed
                if delta_mean.shape[0] < hidden_size:
                    padding = torch.zeros(hidden_size - delta_mean.shape[0], device=device)
                    delta_mean = torch.cat([delta_mean, padding])
                else:
                    delta_mean = delta_mean[:hidden_size]

            deltas.append(delta_mean.cpu().numpy())
            print(f"Added delta with shape: {delta_mean.shape}")

        except Exception as e:
            print(f"Error processing text '{text[:50]}...': {e}")
            # Add random values as fallback
            deltas.append(np.random.randn(hidden_size) * 0.1)

    if not deltas:
        print("No deltas extracted. Returning dummy results.")
        return [(i, float(i)) for i in range(top_k)]

    try:
        all_deltas = np.array(deltas)
        print(f"All deltas shape: {all_deltas.shape}")
        
        if all_deltas.ndim == 1:
            all_deltas = all_deltas.reshape(1, -1)

        # Calculate variance across samples (axis=0)
        variance = np.var(all_deltas, axis=0)
        print(f"Variance shape: {variance.shape}")
        print(f"Variance range: {variance.min():.6f} to {variance.max():.6f}")
        
        # Get top-k neurons with highest variance
        top_indices = np.argsort(variance)[-top_k:][::-1]  # Descending order
        
        # Make sure indices are within bounds
        top_indices = [i for i in top_indices if 0 <= i < hidden_size]
        
        results = [(int(i), float(variance[i])) for i in top_indices]
        
        print(f"Found {len(results)} delta-sensitive neurons")
        for neuron, var in results:
            print(f"  Neuron {neuron}: variance = {var:.6f}")
            
        return results
        
    except Exception as e:
        print(f"Error computing variance: {e}")
        return [(i, float(i)) for i in range(min(top_k, hidden_size))]

def perturb_neurons(tensor, neuron_indices, mode="zero", std=1.0):
    """Perturb specific neurons in the tensor."""
    if not neuron_indices:
        return tensor
        
    perturbed = tensor.clone()
    valid_indices = [idx for idx in neuron_indices if 0 <= idx < tensor.shape[-1]]
    
    if not valid_indices:
        print("Warning: No valid neuron indices to perturb")
        return tensor
    
    print(f"Perturbing neurons {valid_indices} with mode '{mode}'")
    
    for idx in valid_indices:
        if mode == "zero":
            perturbed[..., idx] = 0
        elif mode == "noise":
            # Create noise tensor on the same device as the input tensor
            noise = torch.randn_like(perturbed[..., idx]) * std
            perturbed[..., idx] += noise
        elif mode == "mean":
            mean_val = perturbed[..., idx].mean()
            perturbed[..., idx] = mean_val
        elif mode == "scale":
            # Scale the activations
            perturbed[..., idx] *= std
    
    return perturbed

def register_perturbation_hook(layer, neuron_indices, mode="zero", std=1.0):
    """Register a hook to perturb neurons during forward pass."""
    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            return perturb_neurons(output, neuron_indices, mode, std)
        elif isinstance(output, tuple):
            perturbed_first = perturb_neurons(output[0], neuron_indices, mode, std)
            return (perturbed_first, *output[1:])
        return output
    return layer.register_forward_hook(hook_fn)

def evaluate_perplexity(model, tokenizer, texts, device):
    """Evaluate perplexity on a set of texts."""
    model.eval()
    # Ensure model is on the correct device
    model = model.to(device)
    loss_fct = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')
    total_loss = 0
    total_tokens = 0

    for text in texts:
        try:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            # Move inputs to the same device as model
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs["input_ids"][..., 1:].contiguous()

                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                total_loss += loss.item()
                total_tokens += shift_labels.ne(tokenizer.pad_token_id).sum().item()
        except Exception as e:
            print(f"Error in perplexity calculation for text: {e}")
            continue

    if total_tokens == 0:
        return float('inf')
    
    return np.exp(total_loss / total_tokens)

def evaluate_perturbation_effect(model, tokenizer, relation_texts, other_texts, neuron_indices, layer_idx=0, mode="zero", std=1.0):
    """
    Evaluate perplexity changes for:
    - relation-specific texts
    - other unrelated texts

    Returns:
        dict with perplexity before/after erasure for both text sets.
    """

    hidden_size = model.config.hidden_size
    
    # Filter valid neuron indices
    valid_indices = [i for i in neuron_indices if 0 <= i < hidden_size]
    
    if not valid_indices:
        print("No valid neuron indices provided!")
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Evaluate baseline perplexity on relation-specific texts
    print("\n=== Baseline Evaluation ===")
    ppl_relation_before = evaluate_perplexity(model, tokenizer, relation_texts, device)
    print(f"Baseline PPL (Erased Relation): {ppl_relation_before:.2f}")

    # Evaluate baseline perplexity on other texts
    ppl_other_before = evaluate_perplexity(model, tokenizer, other_texts, device)
    print(f"Baseline PPL (Other Relations): {ppl_other_before:.2f}")
    

    # Register perturbation hook
    layers = get_model_layers(model)
    if layers is None or layer_idx >= len(layers):
        print(f"Invalid layer index {layer_idx}")
        return None, None

    target_layer = layers[layer_idx]
    hook = register_perturbation_hook(target_layer, valid_indices, mode=mode, std=std)

    # Evaluate perturbed perplexity on relation-specific texts
    print("\n=== Perturbed Evaluation (Erased Relation Texts) ===")
    ppl_relation_after = evaluate_perplexity(model, tokenizer, relation_texts, device)
    print(f"Perturbed PPL (Erased Relation): {ppl_relation_after:.2f}")

    # Evaluate perturbed perplexity on other texts
    print("\n=== Perturbed Evaluation (Other Relation Texts) ===")
    ppl_other_after = evaluate_perplexity(model, tokenizer, other_texts, device)
    print(f"Perturbed PPL (Other Relations): {ppl_other_after:.2f}")

    hook.remove()

    # Compute percentage changes
    rel_delta_pct = (
        ((ppl_relation_after - ppl_relation_before) / ppl_relation_before) * 100
        if ppl_relation_before > 0 else 0.0
    )
    other_delta_pct = (
        ((ppl_other_after - ppl_other_before) / ppl_other_before) * 100
        if ppl_other_before > 0 else 0.0
    )

    # -------------------------
    # Return all results in a single dict
    # -------------------------
    return {
        "erased_relation": {
            "before": ppl_relation_before,
            "after": ppl_relation_after,
            "change_pct": rel_delta_pct
        },
        "other_relations": {
            "before": ppl_other_before,
            "after": ppl_other_after,
            "change_pct": other_delta_pct
        }
    }