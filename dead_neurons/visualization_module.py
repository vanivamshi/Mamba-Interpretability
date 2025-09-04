# visualization.py - Enhanced visualization for neuron analysis

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

import matplotlib.pyplot as plt

def plot_dead_neuron_distribution(all_models_data, save_path=None):
    """
    Plots the distribution of dead neurons across layers for multiple models.

    Args:
        all_models_data (dict): {model_name: {layer_idx: dead_ratio OR {"dead_ratio": float}, ...}}
    """
    plt.figure(figsize=(12, 6))

    for model_label, dead_neuron_data in all_models_data.items():
        layers = sorted(dead_neuron_data.keys())

        # Accept either float or dict structure
        dead_ratios = []
        for layer in layers:
            val = dead_neuron_data[layer]
            if isinstance(val, dict) and "dead_ratio" in val:
                dead_ratios.append(val["dead_ratio"])
            else:
                dead_ratios.append(float(val))

        plt.plot(layers, dead_ratios, marker="o", label=model_label)

    plt.xlabel("Layer")
    plt.ylabel("Fraction of Dead Neurons")
    plt.title("Dead Neuron Distribution Across Layers")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_dead_neuron_stats_by_layer(layer_results, model_name, save_dir="plots"):
    """
    Reproduce Figure 1 from the paper:
    (a) % of dead neurons by layer
    (b) average activation frequency among non-dead neurons by layer

    Args:
        layer_results: dict mapping layer_idx -> {"dead_neurons": [...], "activation_freq": [...]}
        model_name: str
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    num_layers = len(layer_results)
    layers = sorted(layer_results.keys())
    rel_layers = [i / max(layers) for i in layers]   # relative depth 0..1

    pct_dead = []
    avg_non_dead_freq = []

    for l in layers:
        dead = layer_results[l]["dead_neurons"]
        freqs = np.array(layer_results[l]["activation_freq"])
        hidden_size = len(freqs)

        # % dead
        pct_dead.append(100 * len(dead) / hidden_size if hidden_size > 0 else 0)

        # avg freq among non-dead
        non_dead = [freqs[i] for i in range(hidden_size) if i not in dead]
        avg_non_dead_freq.append(np.mean(non_dead) if len(non_dead) > 0 else 0)

    # Plot (a) % dead neurons
    plt.figure(figsize=(8, 6))
    plt.plot(rel_layers, pct_dead, marker="o", color="red", linewidth=2, markersize=8)
    plt.xlabel("Layer (relative depth)", fontsize=12)
    plt.ylabel("% Dead Neurons", fontsize=12)
    plt.title(f"(a) % Dead Neurons Across Layers\n{model_name}", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    filename_a = f"{model_name.replace('/', '_')}_fig1a_dead_percentage.png"
    filepath_a = os.path.join(save_dir, filename_a)
    plt.savefig(filepath_a, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # Plot (b) avg activation freq of non-dead
    plt.figure(figsize=(8, 6))
    plt.plot(rel_layers, avg_non_dead_freq, marker="o", color="blue", linewidth=2, markersize=8)
    plt.xlabel("Layer (relative depth)", fontsize=12)
    plt.ylabel("Avg Activation Frequency (non-dead)", fontsize=12)
    plt.title(f"(b) Avg Non-Dead Activation Frequency\n{model_name}", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename_b = f"{model_name.replace('/', '_')}_fig1b_activation_frequency.png"
    filepath_b = os.path.join(save_dir, filename_b)
    plt.savefig(filepath_b, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"✅ Saved Figure 1a (dead percentage) to {filepath_a}")
    print(f"✅ Saved Figure 1b (activation frequency) to {filepath_b}")
    return [filepath_a, filepath_b]


def plot_neuron_distribution(dead_freq, pos_corr, model_name, layer_idx, save_dir="plots"):
    """Plot distribution of neuron characteristics."""
    
    # Dead neuron frequency distribution
    plt.figure(figsize=(10, 6))
    plt.hist(dead_freq, bins=30, alpha=0.7, color='red', edgecolor='black')
    plt.title(f'Dead Neuron Activation Distribution\n{model_name} Layer {layer_idx}', fontsize=14)
    plt.xlabel('Maximum Activation Value', fontsize=12)
    plt.ylabel('Number of Neurons', fontsize=12)
    #plt.axvline(x=1e-1, color='darkred', linestyle='--', label=f'Threshold (1e-1)')
    plt.legend()
    #plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    filename1 = f"{model_name.replace('/', '_')}_layer{layer_idx}_dead_distribution.png"
    filepath1 = os.path.join(save_dir, filename1)
    plt.savefig(filepath1, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Positional correlation distribution
    plt.figure(figsize=(10, 6))
    plt.hist(pos_corr, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f'Position Correlation Distribution\n{model_name} Layer {layer_idx}', fontsize=14)
    plt.xlabel('Correlation with Token Position', fontsize=12)
    plt.ylabel('Number of Neurons', fontsize=12)
    #plt.axvline(x=0.3, color='darkblue', linestyle='--', label='Threshold (0.3)')
    #plt.axvline(x=-0.3, color='darkblue', linestyle='--')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename2 = f"{model_name.replace('/', '_')}_layer{layer_idx}_positional_distribution.png"
    filepath2 = os.path.join(save_dir, filename2)
    plt.savefig(filepath2, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    return [filepath1, filepath2]


def plot_overlap_analysis(overlap_results, model_name, layer_idx, save_dir="plots"):
    """Plot Venn diagram-style overlap analysis."""
    plt.figure(figsize=(10, 6))
    
    categories = ['Dead', 'Positional', 'Delta-Sensitive']
    overlaps = [
        overlap_results.get('dead_and_positional', 0),
        overlap_results.get('dead_and_delta', 0),
        overlap_results.get('positional_and_delta', 0),
        overlap_results.get('all_three', 0)
    ]
    
    overlap_labels = [
        'Dead ∩ Positional',
        'Dead ∩ Delta',
        'Positional ∩ Delta', 
        'All Three'
    ]
    
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    bars = plt.bar(overlap_labels, overlaps, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, value in zip(bars, overlaps):
        if value > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(value), ha='center', va='bottom', fontweight='bold')
    
    plt.title(f'Neuron Category Overlaps\n{model_name} Layer {layer_idx}', fontsize=14)
    plt.ylabel('Number of Neurons', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{model_name.replace('/', '_')}_layer{layer_idx}_overlap_analysis.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    return filepath


def plot_comprehensive_ablation(ablation_results, model_name, layer_idx, save_dir="plots"):
    """Enhanced ablation results visualization."""
    if not ablation_results or len(ablation_results) <= 1:
        print("No ablation results to plot")
        return None
    
    # Prepare data
    conditions = list(ablation_results.keys())
    perplexities = list(ablation_results.values())
    
    # Create more readable labels
    label_mapping = {
        "baseline": "Baseline",
        "dead_ablated": "Dead\nAblated",
        "positional_ablated": "Positional\nAblated", 
        "delta_ablated": "Delta\nAblated",
        "dead_and_positional_ablated": "Dead +\nPositional"
    }
    
    readable_labels = [label_mapping.get(cond, cond.replace('_', ' ').title()) for cond in conditions]
    
    # Calculate percentage changes from baseline
    baseline_ppl = ablation_results.get('baseline', perplexities[0])
    pct_changes = [((ppl - baseline_ppl) / baseline_ppl) * 100 if baseline_ppl > 0 else 0 
                   for ppl in perplexities]
    
    # Plot 1: Absolute perplexity values
    plt.figure(figsize=(12, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(conditions)]
    bars1 = plt.bar(range(len(conditions)), perplexities, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bar, ppl in zip(bars1, perplexities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(perplexities) * 0.01,
                f'{ppl:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(range(len(conditions)), readable_labels)
    plt.ylabel('Perplexity', fontsize=12)
    plt.title(f'Absolute Perplexity After Ablation\n{model_name} Layer {layer_idx}', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    filename1 = f"{model_name.replace('/', '_')}_layer{layer_idx}_ablation_absolute.png"
    filepath1 = os.path.join(save_dir, filename1)
    plt.savefig(filepath1, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Plot 2: Percentage changes
    plt.figure(figsize=(12, 6))
    change_colors = ['green' if change <= 0 else 'red' for change in pct_changes]
    bars2 = plt.bar(range(len(conditions)), pct_changes, color=change_colors, alpha=0.7, edgecolor='black')
    
    # Add percentage labels
    for bar, pct in zip(bars2, pct_changes):
        plt.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + (max(pct_changes) * 0.02 if pct >= 0 else min(pct_changes) * 0.02),
                f'{pct:+.1f}%', ha='center', va='bottom' if pct >= 0 else 'top', fontweight='bold')
    
    plt.xticks(range(len(conditions)), readable_labels)
    plt.ylabel('Perplexity Change (%)', fontsize=12)
    plt.title('Percentage Change from Baseline', fontsize=14)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    filename2 = f"{model_name.replace('/', '_')}_layer{layer_idx}_ablation_percentage.png"
    filepath2 = os.path.join(save_dir, filename2)
    plt.savefig(filepath2, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    return [filepath1, filepath2]


def plot_dead_neuron_stats(dead_neurons, dead_frequencies, model_name, layer_idx, save_dir="plots"):
    """Reproduce Fig. 1: (a) % of dead neurons, (b) avg activation frequency among non-dead neurons."""
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    total_neurons = len(dead_frequencies)
    num_dead = len(dead_neurons)
    pct_dead = (num_dead / total_neurons) * 100 if total_neurons > 0 else 0
    
    non_dead_freqs = [dead_frequencies[i] for i in range(total_neurons) if i not in dead_neurons]
    avg_non_dead_freq = np.mean(non_dead_freqs) if non_dead_freqs else 0
    
    # Plot (a) % of dead neurons
    plt.figure(figsize=(8, 6))
    plt.bar(["Dead neurons"], [pct_dead], color="red", alpha=0.7, edgecolor='black')
    plt.ylabel("% Dead Neurons", fontsize=12)
    plt.title(f"Percentage of Dead Neurons\n{model_name} Layer {layer_idx}", fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    filename_a = f"{model_name.replace('/', '_')}_layer{layer_idx}_fig1a_dead_percentage.png"
    filepath_a = os.path.join(save_dir, filename_a)
    plt.savefig(filepath_a, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    
    # Plot (b) avg activation among non-dead
    plt.figure(figsize=(8, 6))
    plt.bar(["Non-dead"], [avg_non_dead_freq], color="blue", alpha=0.7, edgecolor='black')
    plt.ylabel("Average Max Activation", fontsize=12)
    plt.title(f"Activation Frequency (Non-dead)\n{model_name} Layer {layer_idx}", fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    filename_b = f"{model_name.replace('/', '_')}_layer{layer_idx}_fig1b_activation_frequency.png"
    filepath_b = os.path.join(save_dir, filename_b)
    plt.savefig(filepath_b, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    
    print(f"Saved Figure 1a to {filepath_a}")
    print(f"Saved Figure 1b to {filepath_b}")
    return [filepath_a, filepath_b]


def plot_positional_neurons_scatter(positional_correlations, model_name, layer_idx, save_dir="plots"):
    """Reproduce Fig. 8: scatter of positional neurons with correlation coloring."""
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    neurons = np.arange(len(positional_correlations))
    correlations = np.array(positional_correlations)
    
    plt.figure(figsize=(12, 6))
    
    scatter = plt.scatter(neurons, correlations, 
                         c=correlations, cmap="coolwarm", 
                         s=40, alpha=0.8, edgecolor="black")
    
    plt.axhline(y=0.3, color="gray", linestyle="--", alpha=0.7)
    plt.axhline(y=-0.3, color="gray", linestyle="--", alpha=0.7)
    plt.xlabel("Neuron Index", fontsize=12)
    plt.ylabel("Correlation with Token Position", fontsize=12)
    plt.title(f"Positional Neurons\n{model_name} Layer {layer_idx}", fontsize=14)
    plt.colorbar(scatter, label="Correlation Strength")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{model_name.replace('/', '_')}_layer{layer_idx}_fig8_positional.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    
    print(f"Saved Figure 8 to {filepath}")
    return filepath


def plot_multiple_positional_neurons_scatter(all_models_correlations, layer_identifier, save_dir="plots"):
    """
    Plots positional neuron scatter plots for multiple models in a single figure
    with subplots.

    Args:
        all_models_correlations (dict): Dictionary where keys are model_label - layer_idx (str)
                                        and values are lists/arrays of positional correlations.
        layer_identifier (str or int): Identifier for the layer(s) being plotted (e.g., "All Layers" or a specific layer index).
        save_dir (str): Directory to save the plot.
    """
    num_models_layers = len(all_models_correlations)
    if num_models_layers == 0:
        print("No data to plot for multiple positional neurons.")
        return None

    # Determine grid size (e.g., 2x3 for 6 models, 3x2 for 5 models)
    n_cols = min(3, num_models_layers) # Max 3 columns
    n_rows = (num_models_layers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    axes = axes.flatten() # Flatten the 2D array of axes for easy iteration

    os.makedirs(save_dir, exist_ok=True)
    plot_paths = []

    for i, (plot_label, correlations) in enumerate(all_models_correlations.items()):
        ax = axes[i]
        neurons = np.arange(len(correlations))
        correlations_np = np.array(correlations)

        scatter = ax.scatter(neurons, correlations_np,
                             c=correlations_np, cmap="coolwarm",
                             s=20, alpha=0.8, edgecolor="black")

        ax.axhline(y=0.3, color="gray", linestyle="--", alpha=0.7)
        ax.axhline(y=-0.3, color="gray", linestyle="--", alpha=0.7)
        ax.set_xlabel("Neuron Index", fontsize=10)
        ax.set_ylabel("Correlation", fontsize=10)
        ax.set_title(f"{plot_label}", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add a colorbar for each subplot for clarity, but make it compact
        cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', shrink=0.7)
        cbar.set_label("Correlation Strength", fontsize=8)

    # Hide any unused subplots
    for i in range(num_models_layers, len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle(f"Positional Neurons Across Models)", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to make space for suptitle
    
    filename = f"all_models_positional_neurons_layer{str(layer_identifier).replace(' ', '_')}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    
    print(f"✅ Saved combined positional neuron scatter plot to {filepath}")
    return filepath


def plot_positional_ablation_results_multi_model(all_ablation_results, layer_identifier, save_dir="plots"):
    """
    Plots ablation study results for positional neurons across multiple models/layers
    in a single figure with subplots.

    Args:
        all_ablation_results (dict): Dictionary where keys are model_label - layer_idx (str)
                                     and values are dicts with 'baseline_ppl' and 'ablated_ppl'.
        layer_identifier (str or int): Identifier for the layer(s) being plotted.
        save_dir (str): Directory to save the plot.
    """
    num_results = len(all_ablation_results)
    if num_results == 0:
        print("No ablation results to plot.")
        return None

    n_cols = min(3, num_results)
    n_rows = (num_results + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    axes = axes.flatten()

    os.makedirs(save_dir, exist_ok=True)

    for i, (plot_label, results) in enumerate(all_ablation_results.items()):
        ax = axes[i]
        baseline_ppl = results.get('baseline_ppl')
        ablated_ppl = results.get('ablated_ppl')
        num_ablated = results.get('num_ablated_neurons', 0)

        if baseline_ppl is None or ablated_ppl is None:
            ax.text(0.5, 0.5, "Data N/A", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title(plot_label, fontsize=12)
            continue

        ppl_change_percent = ((ablated_ppl - baseline_ppl) / baseline_ppl) * 100 if baseline_ppl > 0 else 0

        conditions = ['Baseline', 'Ablated']
        perplexities = [baseline_ppl, ablated_ppl]
        colors = ['#1f77b4', '#d62728'] # Blue for baseline, red for ablated

        bars = ax.bar(conditions, perplexities, color=colors, alpha=0.7, edgecolor='black')

        for bar, ppl in zip(bars, perplexities):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(perplexities) * 0.01,
                    f'{ppl:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax.set_ylabel('Perplexity', fontsize=10)
        ax.set_title(f"{plot_label}\n({num_ablated} neurons ablated)", fontsize=12)
        ax.grid(axis='y', alpha=0.3)

        # Add percentage change text
        ax.text(0.5, 0.95, f'Change: {ppl_change_percent:+.1f}%', 
                transform=ax.transAxes, ha='center', va='top', 
                color='green' if ppl_change_percent <= 0 else 'red', 
                fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))

    # Hide any unused subplots
    for i in range(num_results, len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle(f"Positional Neuron Ablation Study (Layer {layer_identifier})", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    filename = f"all_models_positional_ablation_layer{str(layer_identifier).replace(' ', '_')}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    
    print(f"✅ Saved combined positional neuron ablation plot to {filepath}")
    return filepath


def plot_neuron_activation_heatmap(activations_matrix, tokens, neuron_indices, model_label, layer_idx, sample_text_idx, save_dir="plots"):
    """
    Plots a heatmap of activations for specific neurons across a sequence.

    Args:
        activations_matrix (np.ndarray): [sequence_length, num_tracked_neurons]
        tokens (list): List of tokens corresponding to the sequence.
        neuron_indices (list): The actual indices of the neurons being plotted.
        model_label (str): Label of the model.
        layer_idx (int): Index of the layer.
        sample_text_idx (int): Index of the sample text used.
        save_dir (str): Directory to save the plot.
    """
    if activations_matrix.size == 0:
        print(f"No activation data to plot for {model_label} Layer {layer_idx} Text {sample_text_idx}.")
        return None

    plt.figure(figsize=(max(10, len(tokens) * 0.5), max(6, len(neuron_indices) * 0.5)))
    
    # Use seaborn heatmap for better visualization
    sns.heatmap(activations_matrix.T, # Transpose to have neurons on Y-axis, tokens on X-axis
                cmap="viridis", # Or "magma", "plasma", "hot"
                cbar_kws={'label': 'Activation Value'},
                yticklabels=[f'N{n_idx}' for n_idx in neuron_indices], # Label Y-axis with neuron indices
                xticklabels=tokens,
                linewidths=.5, linecolor='gray',
                square=False # Set to False for non-square cells
               )
    
    plt.xlabel("Tokens", fontsize=12)
    plt.ylabel("Neuron Index", fontsize=12)
    plt.title(f"Neuron Activations Heatmap\n{model_label} Layer {layer_idx} (Sample Text {sample_text_idx})", fontsize=14)
    plt.xticks(rotation=90, ha='center', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    filename = f"{model_label.replace('/', '_')}_layer{layer_idx}_text{sample_text_idx}_heatmap.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    
    print(f"✅ Saved neuron activation heatmap to {filepath}")
    return filepath


def plot_pruning_results(pruning_results, model_name, layer_idx, save_dir="plots"):
    """Visualize pruning experiment results."""
    
    # Plot 1: Before/After comparison
    plt.figure(figsize=(10, 6))
    baseline_ppl = pruning_results['baseline_perplexity']
    pruned_ppl = pruning_results['pruned_perplexity']
    
    conditions = ['Baseline', 'After Pruning']
    perplexities = [baseline_ppl, pruned_ppl]
    colors = ['blue', 'red']
    
    bars = plt.bar(conditions, perplexities, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bar, ppl in zip(bars, perplexities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(perplexities) * 0.01,
                f'{ppl:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel('Perplexity', fontsize=12)
    plt.title(f'Pruning Impact on Perplexity\n{model_name} Layer {layer_idx}', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    filename1 = f"{model_name.replace('/', '_')}_layer{layer_idx}_pruning_impact.png"
    filepath1 = os.path.join(save_dir, filename1)
    plt.savefig(filepath1, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Plot 2: Pruning statistics
    plt.figure(figsize=(8, 8))
    neurons_pruned = pruning_results['neurons_pruned']
    total_neurons = int(neurons_pruned / pruning_results['pruning_ratio'])
    neurons_kept = total_neurons - neurons_pruned
    
    sizes = [neurons_kept, neurons_pruned]
    labels = [f'Kept\n({neurons_kept})', f'Pruned\n({neurons_pruned})']
    colors = ['lightgreen', 'lightcoral']
    
    wedges, texts, autotexts = plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                      startangle=90, textprops={'fontsize': 10})
    
    plt.title(f'Neuron Pruning Distribution\n({pruning_results["pruning_ratio"]:.1%} pruned)', fontsize=14)
    plt.tight_layout()
    
    filename2 = f"{model_name.replace('/', '_')}_layer{layer_idx}_pruning_distribution.png"
    filepath2 = os.path.join(save_dir, filename2)
    plt.savefig(filepath2, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    return [filepath1, filepath2]


def plot_attention_neurons_analysis(attention_data, model_name, layer_idx, save_dir="plots"):
    """
    Create comprehensive visualization for attention neurons analysis.
    
    Args:
        attention_data: Dictionary containing attention neurons data
        model_name: Name of the model being analyzed
        layer_idx: Layer index being analyzed
        save_dir: Directory to save the plot
    
    Returns:
        Path to the saved plot
    """
    import os
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a comprehensive attention neurons visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Attention Neurons Analysis - {model_name} (Layer {layer_idx})', fontsize=16)
    
    try:
        # Extract data from attention_data
        if 'analysis_results' in attention_data and 'attention_weighted' in attention_data['analysis_results']:
            layer_data = attention_data['analysis_results']['attention_weighted'].get(layer_idx, {})
            
            if 'neuron_activations' in layer_data:
                activations = layer_data['neuron_activations']
                if hasattr(activations, 'cpu'):
                    activations = activations.cpu().numpy()
                
                # Plot 1: Neuron activations distribution
                axes[0, 0].hist(activations, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                axes[0, 0].set_title('Neuron Activation Distribution')
                axes[0, 0].set_xlabel('Activation Value')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Plot 2: Top neurons by activation
                top_k = min(20, len(activations))
                top_indices = np.argsort(activations)[-top_k:]
                top_activations = activations[top_indices]
                axes[0, 1].bar(range(top_k), top_activations, color='lightcoral')
                axes[0, 1].set_title(f'Top {top_k} Neurons by Activation')
                axes[0, 1].set_xlabel('Neuron Rank')
                axes[0, 1].set_ylabel('Activation Value')
                axes[0, 1].set_xticks(range(top_k))
                axes[0, 1].set_xticklabels([f'#{idx}' for idx in top_indices], rotation=45)
                axes[0, 1].grid(True, alpha=0.3)
                
                # Plot 3: Activation vs importance scatter
                if 'neuron_importance' in layer_data:
                    importance = layer_data['neuron_importance']
                    if hasattr(importance, 'cpu'):
                        importance = importance.cpu().numpy()
                    
                    axes[0, 2].scatter(activations, importance, alpha=0.6, color='mediumseagreen')
                    axes[0, 2].set_title('Activation vs Importance')
                    axes[0, 2].set_xlabel('Activation Value')
                    axes[0, 2].set_ylabel('Importance Score')
                    axes[0, 2].grid(True, alpha=0.3)
                
                # Plot 4: Attention heatmap (if available)
                if 'attention_vectors' in attention_data.get('attention_data', {}).get(layer_idx, {}):
                    attention = attention_data['attention_data'][layer_idx]['attention_vectors']
                    if hasattr(attention, 'cpu'):
                        attention = attention.cpu().numpy()
                    
                    # Take first batch and average across heads
                    attention_avg = attention[0].mean(axis=0)  # Average across attention heads
                    im = axes[1, 0].imshow(attention_avg, cmap='viridis', aspect='auto')
                    axes[1, 0].set_title('Attention Heatmap (Layer Average)')
                    axes[1, 0].set_xlabel('Sequence Position')
                    axes[1, 0].set_ylabel('Neuron Index')
                    plt.colorbar(im, ax=axes[1, 0])
                
                # Plot 5: Neuron diversity analysis
                if 'neuron_diversity' in layer_data:
                    diversity = layer_data['neuron_diversity']
                    axes[1, 1].text(0.5, 0.5, f'Neuron Diversity:\n{diversity:.4f}', 
                                   ha='center', va='center', transform=axes[1, 1].transAxes,
                                   fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
                    axes[1, 1].set_title('Neuron Diversity Score')
                    axes[1, 1].axis('off')
                
                # Plot 6: Summary statistics
                stats_text = f"""Summary Statistics:
                Total Neurons: {len(activations)}
                Mean Activation: {np.mean(activations):.4f}
                Std Activation: {np.std(activations):.4f}
                Min Activation: {np.min(activations):.4f}
                Max Activation: {np.max(activations):.4f}"""
                
                axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes,
                               fontsize=12, verticalalignment='center',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
                axes[1, 2].set_title('Statistical Summary')
                axes[1, 2].axis('off')
        
        else:
            # If no analysis results, show error message
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No attention neurons data available', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=14, color='red')
                ax.set_title('Data Not Available')
                ax.axis('off')
    
    except Exception as e:
        # If any error occurs, show error message
        for ax in axes.flat:
            ax.text(0.5, 0.5, f'Error in visualization:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='red')
            ax.set_title('Visualization Error')
            ax.axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"{model_name.replace('/', '_')}_layer_{layer_idx}_attention_neurons_analysis.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"✅ Attention neurons visualization saved to: {filepath}")
    return filepath


def create_comprehensive_report(results, model_name, layer_idx, save_dir="plots"):
    """Create a comprehensive visualization report."""
    print(f"\n=== Creating Comprehensive Visualization Report ===")
    print(f"Model: {model_name}, Layer: {layer_idx}")
    
    saved_plots = []
    
    # 1. Neuron distributions
    if 'dead_frequencies' in results and 'positional_correlations' in results:
        plot_paths = plot_neuron_distribution(
            results['dead_frequencies'], 
            results['positional_correlations'],
            model_name, layer_idx, save_dir
        )
        saved_plots.extend(plot_paths)
    
    # 2. Overlap analysis
    if 'overlap_analysis' in results:
        plot_path = plot_overlap_analysis(
            results['overlap_analysis'],
            model_name, layer_idx, save_dir
        )
        saved_plots.append(plot_path)
    
    # 3. Comprehensive ablation
    if 'ablation_study' in results:
        plot_paths = plot_comprehensive_ablation(
            results['ablation_study'],
            model_name, layer_idx, save_dir
        )
        if plot_paths:
            saved_plots.extend(plot_paths)
    
    # 4. Pruning results
    if 'pruning_results' in results:
        plot_paths = plot_pruning_results(
            results['pruning_results'],
            model_name, layer_idx, save_dir
        )
        saved_plots.extend(plot_paths)

    # 5. Figure 1
    if 'dead_neurons' in results and 'dead_frequencies' in results:
        plot_paths = plot_dead_neuron_stats(
            results['dead_neurons'],
            results['dead_frequencies'],
            model_name, layer_idx, save_dir
        )
        saved_plots.extend(plot_paths)

    # 6. Figure 8
    if 'positional_correlations' in results:
        plot_path = plot_positional_neurons_scatter(
            results['positional_correlations'],
            model_name, layer_idx, save_dir
        )
        saved_plots.append(plot_path)
    
    # 7. Attention neurons visualization
    if 'attention_neurons' in results and results['attention_neurons'] is not None:
        try:
            plot_path = plot_attention_neurons_analysis(
                results['attention_neurons'],
                model_name, layer_idx, save_dir
            )
            saved_plots.append(plot_path)
        except Exception as e:
            print(f"Warning: Attention neurons visualization failed: {e}")
    
    print(f"Generated {len(saved_plots)} visualization plots:")
    for plot_path in saved_plots:
        print(f"  - {plot_path}")
    
    return saved_plots


# Example usage
if __name__ == "__main__":
    # Example for plot_multiple_positional_neurons_scatter
    example_multi_correlations = {
        "Model A - Layer 0": np.random.normal(0, 0.2, 500),
        "Model B - Layer 0": np.random.normal(0.1, 0.3, 500),
        "Model C - Layer 0": np.random.normal(-0.1, 0.25, 500),
        "Model D - Layer 1": np.random.normal(0.05, 0.15, 500),
        "Model E - Layer 1": np.random.normal(-0.05, 0.1, 500),
        "Model F - Layer 2": np.random.normal(0.2, 0.2, 500),
    }
    # plot_multiple_positional_neurons_scatter(example_multi_correlations, layer_identifier="Mixed Layers", save_dir="plots")

    # Example for plot_positional_ablation_results_multi_model
    example_ablation_results = {
        "Model A - Layer 0": {"baseline_ppl": 15.0, "ablated_ppl": 15.5, "num_ablated_neurons": 10},
        "Model B - Layer 0": {"baseline_ppl": 12.0, "ablated_ppl": 12.1, "num_ablated_neurons": 5},
        "Model C - Layer 1": {"baseline_ppl": 18.0, "ablated_ppl": 19.0, "num_ablated_neurons": 20},
    }
    # plot_positional_ablation_results_multi_model(example_ablation_results, layer_identifier="Mixed Layers", save_dir="plots")

    # Example for plot_neuron_activation_heatmap
    # dummy_activations = np.random.rand(50, 5) * 10 # 50 tokens, 5 neurons
    # dummy_tokens = [f"token_{i}" for i in range(50)]
    # dummy_neuron_indices = [10, 25, 30, 45, 50]
    # plot_neuron_activation_heatmap(dummy_activations, dummy_tokens, dummy_neuron_indices, "DummyModel", 0, 0, "plots")
