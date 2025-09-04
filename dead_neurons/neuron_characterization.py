# neuron_characterization.py (Fixed version with proper neuron pruning and analysis)

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from delta_extraction import evaluate_perplexity # Keep this import for evaluate_perplexity
from attention_neurons import integrate_mamba_attention_neurons, MambaAttentionNeurons
from utils import get_model_layers, get_activation_hook_target

def plot_neuron_activation_distribution(model, tokenizer, texts, layer_idx=0, save_dir="plots", test_threshold=1e-1):
    """
    Plot neuron activation distributions individually:
    1. Distribution of max activations per neuron
    2. Distribution of activation frequencies per neuron
    3. Scatter plot of max activation vs frequency
    4. Cumulative distribution of max activations
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    layers = get_model_layers(model)
    if layers is None or layer_idx >= len(layers):
        print(f"Invalid layer index {layer_idx}")
        return

    hidden_size = model.config.hidden_size
    activation_counts = torch.zeros(hidden_size, device=device)
    max_activations = torch.zeros(hidden_size, device=device)
    total_tokens = 0

    def hook_fn(module, inp, out):
        nonlocal total_tokens, max_activations
        if isinstance(out, tuple):
            out = out[0]
        acts = out.detach()   # [batch, seq, hidden]

        # Track max activations per neuron
        max_acts = acts.abs().amax(dim=(0, 1))  # [hidden]
        max_activations = torch.maximum(max_activations, max_acts)

        # Track activation frequency (using the test threshold)
        active_mask = (acts.abs() > test_threshold).float()
        activation_counts.add_(active_mask.sum(dim=(0, 1)))
        total_tokens += acts.shape[0] * acts.shape[1]

    handle = layers[layer_idx].register_forward_hook(hook_fn)

    with torch.no_grad():
        for i, text in enumerate(texts):
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
                model(**inputs)
            except Exception as e:
                print(f"Error processing text {i}: {e}")
                continue

    handle.remove()

    activation_freq = activation_counts.cpu().numpy() / max(total_tokens, 1)
    max_acts_np = max_activations.cpu().numpy()

    os.makedirs(save_dir, exist_ok=True)

    # ---- 1. Distribution of max activations ----
    plt.figure(figsize=(8, 6))
    plt.hist(max_acts_np, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Maximum Activation Value')
    plt.ylabel('Number of Neurons')
    plt.title(f'Distribution of Max Activations (Layer {layer_idx})')
    plt.yscale('log')
    plt.legend()
    f1 = os.path.join(save_dir, f"max_activations_layer{layer_idx}.png")
    plt.savefig(f1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {f1}")

    # ---- 2. Distribution of activation frequencies ----
    plt.figure(figsize=(8, 6))
    plt.hist(activation_freq, bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Activation Frequency')
    plt.ylabel('Number of Neurons')
    plt.title(f'Distribution of Activation Frequencies (Layer {layer_idx})')
    plt.yscale('log')
    plt.legend()
    f2 = os.path.join(save_dir, f"activation_freq_layer{layer_idx}.png")
    plt.savefig(f2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {f2}")

    # ---- 3. Scatter plot: max activation vs frequency ----
    plt.figure(figsize=(8, 6))
    plt.scatter(max_acts_np, activation_freq, alpha=0.6, s=20)
    plt.xlabel('Maximum Activation Value')
    plt.ylabel('Activation Frequency')
    plt.title(f'Max Activation vs Frequency (Layer {layer_idx})')
    plt.xscale('log')
    plt.yscale('log')
    plt.axhline(y=0.001, color='red', linestyle='--', alpha=0.5, label='Cutoff 0.001')
    plt.legend()
    f3 = os.path.join(save_dir, f"scatter_activation_layer{layer_idx}.png")
    plt.savefig(f3, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {f3}")

    # ---- 4. Cumulative distribution ----
    plt.figure(figsize=(8, 6))
    sorted_max_acts = np.sort(max_acts_np)
    cumulative = np.arange(1, len(sorted_max_acts) + 1) / len(sorted_max_acts)
    plt.plot(sorted_max_acts, cumulative, 'b-', linewidth=2)
    plt.xlabel('Maximum Activation Value')
    plt.ylabel('Cumulative Fraction of Neurons')
    plt.title(f'Cumulative Distribution of Max Activations (Layer {layer_idx})')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    f4 = os.path.join(save_dir, f"cumulative_activations_layer{layer_idx}.png")
    plt.savefig(f4, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {f4}")

    return max_acts_np, activation_freq


# neuron_characterization.py
import torch, numpy as np, os, matplotlib.pyplot as plt
from utils import get_model_layers, get_activation_hook_target
# ... (rest of your imports)

def _find_dead_neurons_core(model, tokenizer, texts, layer_idx, threshold, freq_cutoff=0.001):
    """
    Active if |activation| > threshold; dead if active < freq_cutoff of tokens.
    For Transformers, measure post-activation MLP features (input to c_proj).
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Ensure pad token for GPT-2 and friends (prevents padding error downstream)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    layers = get_model_layers(model)
    if layers is None or layer_idx >= len(layers):
        print(f"Invalid layer index {layer_idx}")
        return [], np.array([])

    activation_counts = None  # we’ll size this lazily
    total_tokens = 0

    # pick the correct module / hook kind
    hook_module, hook_kind, _ = get_activation_hook_target(model, layer_idx)

    def _accumulate(acts: torch.Tensor):
        nonlocal activation_counts, total_tokens
        # acts: [batch, seq, features]
        if acts is None:
            return
        if isinstance(acts, tuple):
            acts = acts[0]
        # some modules return [seq, batch, features]; standardize
        if acts.dim() == 3 and acts.shape[0] < acts.shape[1] and acts.shape[0] < acts.shape[2]:
            pass  # assume [batch, seq, hidden]
        # lazily create counter with correct feature dim
        feat = acts.shape[-1]
        if activation_counts is None:
            activation_counts = torch.zeros(feat, device=acts.device)
        active_mask = (acts.abs() > threshold).float()
        activation_counts.add_(active_mask.sum(dim=(0, 1)))
        total_tokens += acts.shape[0] * acts.shape[1]

    def forward_hook(_module, _inp, out):
        _accumulate(out)

    def pre_hook(_module, inp):
        # pre-hook gets a tuple; the first element is the tensor input to this module
        if isinstance(inp, tuple) and len(inp) > 0:
            _accumulate(inp[0])

    # register the appropriate hook
    if hook_kind == "pre":
        handle = hook_module.register_forward_pre_hook(pre_hook)
    else:
        handle = hook_module.register_forward_hook(forward_hook)

    with torch.no_grad():
        for text in texts:
            if not text or not text.strip():
                continue
            try:
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)  # no padding needed here
                enc = {k: v.to(device) for k, v in enc.items()}
                _ = model(**enc)
            except Exception as e:
                print(f"Transformer error: {e}")
                continue

    handle.remove()

    if activation_counts is None or total_tokens == 0:
        return [], np.array([])

    activation_freq = activation_counts.detach().cpu().numpy() / float(total_tokens)
    dead_neurons = [i for i, f in enumerate(activation_freq) if f < freq_cutoff]
    print(f"Layer {layer_idx}: {len(dead_neurons)}/{activation_counts.numel()} dead neurons "
          f"({100.0*len(dead_neurons)/activation_counts.numel():.1f}%) [threshold={threshold}, freq_cutoff={freq_cutoff}]")
    return dead_neurons, activation_freq

# Original function signature, now calls the core with default threshold
def find_dead_neurons(model, tokenizer, texts, layer_idx=0, freq_cutoff=0.001):
    """
    Dead neurons per paper definition, using a reduced threshold of 0.1.
    """
    return _find_dead_neurons_core(model, tokenizer, texts, layer_idx, threshold=0.1, freq_cutoff=freq_cutoff)

# New function for custom thresholds
def find_dead_neurons_custom_threshold(model, tokenizer, texts, layer_idx=0, threshold=None, freq_cutoff=0.001):
    """
    Dead neurons with a custom activation threshold.
    """
    if threshold is None:
        raise ValueError("A custom threshold must be provided for find_dead_neurons_custom_threshold.")
    return _find_dead_neurons_core(model, tokenizer, texts, layer_idx, threshold=threshold, freq_cutoff=freq_cutoff)


def find_positional_neurons(model, tokenizer, texts, layer_idx=0, min_corr=0.3):
    """
    Positional neurons per paper:
    - Compute correlation between neuron activations and *relative* token positions within each sequence.
    - Neuron is positional if |avg_corr| > min_corr.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    layers = get_model_layers(model)
    if layers is None or layer_idx >= len(layers):
        print(f"Invalid layer index {layer_idx}")
        return [], []

    hidden_size = model.config.hidden_size
    
    # Store correlations for each neuron across all processed sequences
    all_neuron_correlations = [[] for _ in range(hidden_size)]
    
    def hook_fn(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        acts = out.detach().cpu().numpy()  # [batch, seq, hidden]
        
        for batch_idx in range(acts.shape[0]):
            seq_len = acts.shape[1]
            if seq_len < 2: # Need at least 2 tokens to compute correlation
                continue
            
            # Relative positions for the current sequence
            relative_positions = np.arange(seq_len)
            
            for neuron_idx in range(hidden_size):
                neuron_activations = acts[batch_idx, :, neuron_idx]
                
                # Compute correlation for this neuron in this sequence
                # Handle cases where std dev is zero (flat activations)
                if np.std(neuron_activations) > 1e-6: # Avoid division by zero
                    corr = np.corrcoef(relative_positions, neuron_activations)[0, 1]
                    if not np.isnan(corr):
                        all_neuron_correlations[neuron_idx].append(corr)

    handle = layers[layer_idx].register_forward_hook(hook_fn)

    with torch.no_grad():
        for i, text in enumerate(texts):
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
                # Ensure input_ids are not empty after tokenization
                if inputs["input_ids"].shape[1] > 1:
                    _ = model(**inputs)
            except Exception as e:
                # print(f"Error processing text {i} for positional neurons: {e}")
                continue

    handle.remove()

    # Calculate the average correlation for each neuron
    avg_correlations = []
    positional_neurons = []
    for i in range(hidden_size):
        if all_neuron_correlations[i]:
            avg_corr = np.mean(all_neuron_correlations[i])
            avg_correlations.append(avg_corr)
            if abs(avg_corr) > min_corr:
                positional_neurons.append(i)
        else:
            avg_correlations.append(0.0) # No data for this neuron, assume 0 correlation

    print(f"Found {len(positional_neurons)} positional neurons (>{min_corr} avg corr)")
    return positional_neurons, avg_correlations


def find_rarely_active_neurons(model, tokenizer, texts, layer_idx=0, threshold=0.05):
    """
    Identify rarely active neurons (not strictly dead, but activate on < threshold fraction of tokens).
    This function will now use the default find_dead_neurons logic, which uses 0.7 for activation threshold.
    If you need a custom activation threshold for 'rarely active', you'd need to pass it explicitly.
    """
    # Note: This function currently re-runs the activation frequency calculation.
    # It might be more efficient to pass activation_freq if already computed.
    _, activation_freq = find_dead_neurons(model, tokenizer, texts, layer_idx)

    rare = [i for i, f in enumerate(activation_freq) if f > 0 and f < threshold]
    print(f"Found {len(rare)} rarely active neurons (<{threshold:.0%} activation)")

    return rare, activation_freq


def analyze_neuron_overlap(dead_neurons, positional_neurons, delta_neurons):
    """
    Analyze overlap between different neuron categories.
    """
    dead_set = set(dead_neurons)
    pos_set = set(positional_neurons)
    delta_set = set(delta_neurons)

    overlap_results = {
        'dead_and_positional': len(dead_set & pos_set),
        'dead_and_delta': len(dead_set & delta_set),
        'positional_and_delta': len(pos_set & delta_set),
        'all_three': len(dead_set & pos_set & delta_set),
        'total_unique': len(dead_set | pos_set | delta_set)
    }

    print("\n=== Neuron Category Overlap Analysis ===")
    print(f"Dead neurons: {len(dead_neurons)}")
    print(f"Positional neurons: {len(positional_neurons)}")
    print(f"Delta-sensitive neurons: {len(delta_neurons)}")
    print(f"Dead ∩ Positional: {overlap_results['dead_and_positional']}")
    print(f"Dead ∩ Delta: {overlap_results['dead_and_delta']}")
    print(f"Positional ∩ Delta: {overlap_results['positional_and_delta']}")
    print(f"All three categories: {overlap_results['all_three']}")
    print(f"Total unique neurons: {overlap_results['total_unique']}")

    return overlap_results


def create_pruning_strategy(dead_neurons, positional_neurons, delta_neurons,
                          pruning_ratio=0.1, preserve_delta=True):
    """
    Create a neuron pruning strategy based on neuron characterization.
    """
    # Determine total_neurons from the max index + 1, assuming 0-indexed and contiguous
    # If lists are empty, default to 0 to avoid errors.
    all_indices = []
    if dead_neurons: all_indices.extend(dead_neurons)
    if positional_neurons: all_indices.extend(positional_neurons)
    if delta_neurons: all_indices.extend(delta_neurons)

    if not all_indices:
        total_neurons = 0
    else:
        total_neurons = max(all_indices) + 1 # Assuming 0-indexed and contiguous neurons

    target_prune_count = int(total_neurons * pruning_ratio)

    # Priority order for pruning
    pruning_candidates = []

    # Dead neurons are highest priority
    pruning_candidates.extend([(idx, 'dead', 1.0) for idx in dead_neurons])
    
    # Positional neurons (not already dead)
    pos_only = [idx for idx in positional_neurons if idx not in set(dead_neurons)]
    pruning_candidates.extend([(idx, 'positional', 0.7) for idx in pos_only])

    # Other neurons (not dead, not positional, and not delta-sensitive if preserving delta)
    all_special = set(dead_neurons) | set(positional_neurons)
    if preserve_delta:
        all_special |= set(delta_neurons)
    
    other_neurons = [idx for idx in range(total_neurons) if idx not in all_special]
    pruning_candidates.extend([(idx, 'other', 0.3) for idx in other_neurons])

    # Sort by score (descending)
    pruning_candidates.sort(key=lambda x: x[2], reverse=True)
    
    # Select neurons up to target_prune_count
    neurons_to_prune = [idx for idx, category, score in pruning_candidates[:target_prune_count]]

    print(f"\n=== Pruning Strategy ===")
    print(f"Total neurons considered: {total_neurons}")
    print(f"Target pruning ratio: {pruning_ratio:.1%}")
    print(f"Neurons to prune: {len(neurons_to_prune)}/{total_neurons}")

    prune_counts = {'dead': 0, 'positional': 0, 'other': 0}
    for idx in neurons_to_prune:
        if idx in set(dead_neurons):
            prune_counts['dead'] += 1
        elif idx in set(pos_only): # Check against positional-only set
            prune_counts['positional'] += 1
        elif idx in set(other_neurons): # Check against other neurons set
            prune_counts['other'] += 1

    print(f"Dead neurons pruned: {prune_counts['dead']}")
    print(f"Positional neurons pruned: {prune_counts['positional']}")
    print(f"Other neurons pruned: {prune_counts['other']}")

    if preserve_delta:
        delta_pruned = len(set(neurons_to_prune) & set(delta_neurons))
        print(f"Delta-sensitive neurons preserved: {len(delta_neurons) - delta_pruned}/{len(delta_neurons)}")

    return neurons_to_prune


def register_ablation_hook(model, layer_idx, neurons_to_ablate):
    """
    Registers a forward hook to ablate specific neurons in a given layer.
    Returns the hook handle.
    """
    layers = get_model_layers(model)
    if layers is None or layer_idx >= len(layers):
        print(f"Invalid layer index {layer_idx} for ablation.")
        return None

    def ablation_hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            # Ensure neurons_to_ablate are valid indices for hidden_states
            valid_neurons = [n for n in neurons_to_ablate if n < hidden_states.shape[-1]]
            if valid_neurons:
                hidden_states[:, :, valid_neurons] = 0
            return (hidden_states, *output[1:])
        else:
            valid_neurons = [n for n in neurons_to_ablate if n < output.shape[-1]]
            if valid_neurons:
                output[:, :, valid_neurons] = 0
            return output

    handle = layers[layer_idx].register_forward_hook(ablation_hook)
    return handle


def ablate_neurons_and_evaluate_perplexity(model, tokenizer, texts, layer_idx, neurons_to_ablate):
    """
    Ablates specified neurons in a given layer and evaluates the model's perplexity.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) # Ensure model is on the correct device

    if not neurons_to_ablate:
        print("No neurons to ablate. Returning baseline perplexity.")
        # If no neurons to ablate, just evaluate baseline perplexity
        return evaluate_perplexity(model, tokenizer, texts, device)

    # Register the ablation hook
    hook_handle = register_ablation_hook(model, layer_idx, neurons_to_ablate)
    if hook_handle is None:
        print(f"Warning: Could not register ablation hook for layer {layer_idx}. Returning inf perplexity.")
        return float('inf') # Indicate failure if hook couldn't be registered

    # Evaluate perplexity with ablated neurons
    ablated_ppl = evaluate_perplexity(model, tokenizer, texts, device)

    # Remove the hook to restore the model
    hook_handle.remove()

    return ablated_ppl


def collect_neuron_activations_for_visualization(model, tokenizer, text, layer_idx, neuron_indices_to_track, max_length=256):
    """
    Collects activations for specific neurons in a given layer for a single text.

    Args:
        model: The Hugging Face model.
        tokenizer: The Hugging Face tokenizer.
        text (str): The input text to analyze.
        layer_idx (int): The index of the layer to track.
        neuron_indices_to_track (list): A list of neuron indices whose activations should be collected.
        max_length (int): Maximum sequence length for tokenization.

    Returns:
        tuple: (activations_matrix, tokens)
               activations_matrix (np.ndarray): [sequence_length, num_tracked_neurons]
               tokens (list): List of tokens corresponding to the sequence.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    layers = get_model_layers(model)
    if layers is None or layer_idx >= len(layers):
        print(f"Invalid layer index {layer_idx} for activation collection.")
        return np.array([]), []

    collected_activations = []
    
    def hook_fn(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        # Detach, move to CPU, convert to numpy, and select only tracked neurons
        # Ensure neuron_indices_to_track are within bounds
        valid_indices = [idx for idx in neuron_indices_to_track if idx < out.shape[-1]]
        if valid_indices:
            acts = out.detach().cpu().numpy()[:, :, valid_indices]
            collected_activations.append(acts)

    handle = layers[layer_idx].register_forward_hook(hook_fn)

    tokens = []
    with torch.no_grad():
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            # Decode tokens to get readable labels for the heatmap
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu().tolist())
            _ = model(**inputs)
        except Exception as e:
            print(f"Error processing text for activation collection: {e}")

    handle.remove()

    if collected_activations:
        # Concatenate activations from all batches (if any)
        # Squeeze(0) if batch_size was 1, to get [seq_len, num_neurons]
        activations_matrix = np.concatenate(collected_activations, axis=0).squeeze(0)
        return activations_matrix, tokens
    else:
        return np.array([]), []


def run_pruning_experiment(model, tokenizer, texts, layer_idx, neurons_to_prune,
                          evaluation_texts=None):
    """
    Run pruning experiment and evaluate impact on model performance.
    """
    if evaluation_texts is None:
        evaluation_texts = texts

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Computing baseline perplexity...")
    baseline_ppl = evaluate_perplexity(model, tokenizer, evaluation_texts, device)

    print(f"Computing perplexity after pruning {len(neurons_to_prune)} neurons...")
    layers = get_model_layers(model)

    def pruning_hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            hidden_states[:, :, neurons_to_prune] = 0
            return (hidden_states, *output[1:])
        else:
            output[:, :, neurons_to_prune] = 0
            return output

    handle = layers[layer_idx].register_forward_hook(pruning_hook)
    pruned_ppl = evaluate_perplexity(model, tokenizer, evaluation_texts, device)
    handle.remove()

    ppl_change = ((pruned_ppl - baseline_ppl) / baseline_ppl) * 100

    results = {
        'baseline_perplexity': baseline_ppl,
        'pruned_perplexity': pruned_ppl,
        'perplexity_change_percent': ppl_change,
        'neurons_pruned': len(neurons_to_prune),
        'pruning_ratio': len(neurons_to_prune) / model.config.hidden_size
    }

    print(f"\n=== Pruning Results ===")
    print(f"Baseline perplexity: {baseline_ppl:.3f}")
    print(f"Pruned perplexity: {pruned_ppl:.3f}")
    print(f"Perplexity change: {ppl_change:+.2f}%")

    return results


def run_complete_neuron_analysis(model, tokenizer, texts, layer_idx=0, delta_neurons=None, pruning_ratio=0.1):
    """
    End-to-end neuron analysis pipeline.
    This function will continue to use the default find_dead_neurons (threshold=0.1).
    """
    print("\n=== Running Complete Neuron Analysis ===")

    # Step 1: Activation distributions
    max_acts, activation_freq = plot_neuron_activation_distribution(model, tokenizer, texts, layer_idx)

    # Step 2: Dead neurons (uses the default threshold of 0.1)
    dead_neurons, _ = find_dead_neurons(model, tokenizer, texts, layer_idx)

    # Step 3: Positional neurons
    positional_neurons, correlations = find_positional_neurons(model, tokenizer, texts, layer_idx)

    # Step 4: Rarely active neurons
    rare_neurons, _ = find_rarely_active_neurons(model, tokenizer, texts, layer_idx)

    # Step 5: Overlap analysis
    if delta_neurons is None:
        delta_neurons = []
    overlaps = analyze_neuron_overlap(dead_neurons, positional_neurons, delta_neurons)

    # Step 6: Create pruning strategy
    neurons_to_prune = create_pruning_strategy(dead_neurons, positional_neurons, delta_neurons,
                                               pruning_ratio=pruning_ratio, preserve_delta=True)

    # Step 7: Run pruning experiment
    pruning_results = run_pruning_experiment(model, tokenizer, texts, layer_idx, neurons_to_prune)
    
    # Step 8: Attention neurons analysis
    print("\nRunning attention neurons analysis...")
    try:
        # Create sample input for attention analysis
        sample_text = texts[0] if texts else "Sample text for analysis"
        sample_input = tokenizer(sample_text, return_tensors="pt")["input_ids"]
        
        attention_neurons = integrate_mamba_attention_neurons(
            model, sample_input, layer_indices=[layer_idx], methods=['attention_weighted']
        )
        
        print("✅ Attention neurons analysis completed successfully")
        
        # Add attention neurons summary to results
        if 'analysis_results' in attention_neurons and 'attention_weighted' in attention_neurons['analysis_results']:
            layer_data = attention_neurons['analysis_results']['attention_weighted'].get(layer_idx, {})
            if 'num_neurons' in layer_data:
                print(f"  Attention neurons analyzed: {layer_data['num_neurons']}")
                if 'mean_activation' in layer_data:
                    print(f"  Mean activation: {layer_data['mean_activation']:.4f}")
                if 'neuron_diversity' in layer_data:
                    print(f"  Neuron diversity: {layer_data['neuron_diversity']:.4f}")
        
    except Exception as e:
        print(f"❌ Attention neurons analysis failed: {e}")
        attention_neurons = None

    return {
        "dead_neurons": dead_neurons,
        "positional_neurons": positional_neurons,
        "rare_neurons": rare_neurons,
        "overlaps": overlaps,
        "neurons_to_prune": neurons_to_prune,
        "pruning_results": pruning_results,
        "attention_neurons": attention_neurons
    }
