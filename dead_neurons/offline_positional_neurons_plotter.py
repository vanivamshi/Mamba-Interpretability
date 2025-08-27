# offline_positional_neurons_plotter.py
"""
Offline plotting script for positional neurons analysis results.
Reads logged JSON data and creates various visualizations for analysis.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime

# Set plot style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Default directories
DEFAULT_DATA_DIR = "analysis_outputs/data"
DEFAULT_PLOTS_DIR = "analysis_outputs/offline_plots"


def load_positional_neurons_data(data_file_path):
    """Load positional neurons data from JSON file."""
    try:
        with open(data_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Loaded data from: {data_file_path}")
        print(f"📊 Analysis timestamp: {data['metadata']['timestamp']}")
        print(f"🔍 Models analyzed: {', '.join(data['metadata']['models_analyzed'])}")
        return data
    except FileNotFoundError:
        print(f"❌ Data file not found: {data_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in data file: {data_file_path}")
        return None


def create_directory_structure():
    """Create necessary directories for offline plots."""
    os.makedirs(DEFAULT_PLOTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(DEFAULT_PLOTS_DIR, "scatter_plots"), exist_ok=True)
    os.makedirs(os.path.join(DEFAULT_PLOTS_DIR, "heatmaps"), exist_ok=True)
    os.makedirs(os.path.join(DEFAULT_PLOTS_DIR, "statistics"), exist_ok=True)
    os.makedirs(os.path.join(DEFAULT_PLOTS_DIR, "comparisons"), exist_ok=True)


def plot_positional_neurons_scatter(data, output_dir):
    """Create scatter plots for positional neurons (similar to original Figure 8 style)."""
    results = data['results']
    metadata = data['metadata']
    
    # Group by model
    models = {}
    for plot_label, correlations in results.items():
        model_name = plot_label.split(' - Layer ')[0]
        layer_num = int(plot_label.split(' - Layer ')[1])
        
        if model_name not in models:
            models[model_name] = {}
        models[model_name][layer_num] = correlations
    
    # Create subplots for each model
    num_models = len(models)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (model_name, layers_data) in enumerate(models.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Collect all neuron data for this model
        all_positions = []
        all_correlations = []
        all_layers = []
        
        for layer_num, correlations in layers_data.items():
            for neuron_idx, corr_data in correlations.items():
                if isinstance(corr_data, dict) and 'positions' in corr_data and 'correlations' in corr_data:
                    positions = np.array(corr_data['positions'])
                    correlations_array = np.array(corr_data['correlations'])
                    
                    all_positions.extend(positions)
                    all_correlations.extend(correlations_array)
                    all_layers.extend([layer_num] * len(positions))
        
        if all_positions:
            # Create scatter plot
            scatter = ax.scatter(all_positions, all_correlations, 
                               c=all_layers, cmap='viridis', alpha=0.6, s=20)
            
            ax.set_xlabel('Position')
            ax.set_ylabel('Correlation')
            ax.set_title(f'{model_name}\nPositional Neurons')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Layer')
            
            # Add statistics
            num_neurons = len(set(zip(all_positions, all_correlations)))
            ax.text(0.02, 0.98, f'Neurons: {num_neurons}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Hide empty subplots
    for idx in range(num_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, "scatter_plots", f"positional_neurons_scatter_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Scatter plot saved: {plot_path}")
    return plot_path


def plot_heatmap_by_model(data, output_dir):
    """Create heatmap showing positional neuron density by model and layer."""
    results = data['results']
    
    # Group by model
    models = {}
    for plot_label, correlations in results.items():
        model_name = plot_label.split(' - Layer ')[0]
        layer_num = int(plot_label.split(' - Layer ')[1])
        
        if model_name not in models:
            models[model_name] = {}
        models[model_name][layer_num] = len(correlations)
    
    # Create heatmap data
    model_names = list(models.keys())
    max_layers = max(len(layers) for layers in models.values())
    
    heatmap_data = np.zeros((len(model_names), max_layers))
    
    for i, model_name in enumerate(model_names):
        for layer_num, neuron_count in models[model_name].items():
            heatmap_data[i, layer_num] = neuron_count
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, 
                xticklabels=range(max_layers),
                yticklabels=model_names,
                annot=True, 
                fmt='d',
                cmap='YlOrRd',
                cbar_kws={'label': 'Number of Positional Neurons'})
    
    plt.xlabel('Layer')
    plt.ylabel('Model')
    plt.title('Positional Neurons Distribution by Model and Layer')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, "heatmaps", f"positional_neurons_heatmap_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🔥 Heatmap saved: {plot_path}")
    return plot_path


def plot_statistics_summary(data, output_dir):
    """Create statistical summary plots."""
    results = data['results']
    
    # Collect statistics
    model_stats = {}
    layer_stats = {}
    
    for plot_label, correlations in results.items():
        model_name = plot_label.split(' - Layer ')[0]
        layer_num = int(plot_label.split(' - Layer ')[1])
        
        # Model statistics
        if model_name not in model_stats:
            model_stats[model_name] = {'total_neurons': 0, 'layers_with_neurons': 0}
        model_stats[model_name]['total_neurons'] += len(correlations)
        model_stats[model_name]['layers_with_neurons'] += 1
        
        # Layer statistics
        if layer_num not in layer_stats:
            layer_stats[layer_num] = 0
        layer_stats[layer_num] += len(correlations)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Model comparison
    models = list(model_stats.keys())
    total_neurons = [model_stats[m]['total_neurons'] for m in models]
    layers_with_neurons = [model_stats[m]['layers_with_neurons'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax1.bar(x - width/2, total_neurons, width, label='Total Neurons', alpha=0.8)
    ax1.bar(x + width/2, layers_with_neurons, width, label='Layers with Neurons', alpha=0.8)
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Count')
    ax1.set_title('Positional Neurons by Model')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45)
    ax1.legend()
    
    # Layer distribution
    layers = sorted(layer_stats.keys())
    neuron_counts = [layer_stats[l] for l in layers]
    
    ax2.bar(layers, neuron_counts, alpha=0.8, color='skyblue')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Total Neurons')
    ax2.set_title('Positional Neurons by Layer (All Models)')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, "statistics", f"positional_neurons_stats_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Statistics plot saved: {plot_path}")
    return plot_path


def plot_correlation_distributions(data, output_dir):
    """Plot correlation value distributions for positional neurons."""
    results = data['results']
    
    # Collect correlation values
    all_correlations = []
    model_correlations = {}
    
    for plot_label, correlations in results.items():
        model_name = plot_label.split(' - Layer ')[0]
        
        if model_name not in model_correlations:
            model_correlations[model_name] = []
        
        for neuron_idx, corr_data in correlations.items():
            if isinstance(corr_data, dict) and 'correlations' in corr_data:
                correlations_array = np.array(corr_data['correlations'])
                all_correlations.extend(correlations_array)
                model_correlations[model_name].extend(correlations_array)
    
    if not all_correlations:
        print("⚠️ No correlation data found for distribution plot")
        return None
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Overall distribution
    ax1.hist(all_correlations, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Correlation Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Overall Correlation Distribution')
    ax1.axvline(np.mean(all_correlations), color='red', linestyle='--', 
                label=f'Mean: {np.mean(all_correlations):.3f}')
    ax1.legend()
    
    # Distribution by model
    for model_name, correlations in model_correlations.items():
        if correlations:
            ax2.hist(correlations, bins=30, alpha=0.6, label=model_name)
    
    ax2.set_xlabel('Correlation Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Correlation Distribution by Model')
    ax2.legend()
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, "statistics", f"correlation_distributions_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Correlation distributions saved: {plot_path}")
    return plot_path


def generate_analysis_report(data, output_dir):
    """Generate a comprehensive analysis report."""
    results = data['results']
    metadata = data['metadata']
    
    report_path = os.path.join(output_dir, f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("POSITIONAL NEURONS ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("METADATA:\n")
        f.write("-" * 10 + "\n")
        f.write(f"Analysis Date: {metadata['timestamp']}\n")
        f.write(f"Number of texts: {metadata['num_texts']}\n")
        f.write(f"Models analyzed: {', '.join(metadata['models_analyzed'])}\n\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 20 + "\n")
        
        # Group by model
        model_summary = {}
        for plot_label, correlations in results.items():
            model_name = plot_label.split(' - Layer ')[0]
            layer_num = int(plot_label.split(' - Layer ')[1])
            
            if model_name not in model_summary:
                model_summary[model_name] = {}
            model_summary[model_name][layer_num] = len(correlations)
        
        for model_name, layers in model_summary.items():
            f.write(f"\n{model_name}:\n")
            total_neurons = sum(layers.values())
            f.write(f"  Total positional neurons: {total_neurons}\n")
            f.write(f"  Layers with neurons: {len(layers)}\n")
            f.write("  Layer breakdown:\n")
            for layer_num, neuron_count in sorted(layers.items()):
                f.write(f"    Layer {layer_num}: {neuron_count} neurons\n")
        
        # Overall statistics
        total_neurons = sum(len(correlations) for correlations in results.values())
        total_layers = len(results)
        
        f.write(f"\nOVERALL STATISTICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total positional neurons found: {total_neurons}\n")
        f.write(f"Total model-layer combinations: {total_layers}\n")
        f.write(f"Average neurons per layer: {total_neurons/total_layers:.2f}\n")
    
    print(f"📋 Analysis report saved: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description='Offline plotting for positional neurons analysis')
    parser.add_argument('--data-file', type=str, 
                       help='Path to JSON data file (if not specified, will look for latest in data directory)')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_PLOTS_DIR,
                       help='Output directory for plots')
    parser.add_argument('--plots', type=str, nargs='+', 
                       choices=['scatter', 'heatmap', 'stats', 'correlations', 'report', 'all'],
                       default=['all'],
                       help='Types of plots to generate')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    create_directory_structure()
    
    # Find data file
    if args.data_file:
        data_file_path = args.data_file
    else:
        # Find latest data file
        data_dir = Path(DEFAULT_DATA_DIR)
        if not data_dir.exists():
            print(f"❌ Data directory not found: {DEFAULT_DATA_DIR}")
            return
        
        json_files = list(data_dir.glob("positional_neurons_data_*.json"))
        if not json_files:
            print(f"❌ No data files found in {DEFAULT_DATA_DIR}")
            return
        
        # Get latest file
        data_file_path = str(max(json_files, key=lambda x: x.stat().st_mtime))
        print(f"📁 Using latest data file: {data_file_path}")
    
    # Load data
    data = load_positional_neurons_data(data_file_path)
    if data is None:
        return
    
    print(f"\n🎨 Generating plots...")
    
    generated_plots = []
    
    # Generate requested plots
    if 'all' in args.plots or 'scatter' in args.plots:
        plot_path = plot_positional_neurons_scatter(data, args.output_dir)
        if plot_path:
            generated_plots.append(plot_path)
    
    if 'all' in args.plots or 'heatmap' in args.plots:
        plot_path = plot_heatmap_by_model(data, args.output_dir)
        if plot_path:
            generated_plots.append(plot_path)
    
    if 'all' in args.plots or 'stats' in args.plots:
        plot_path = plot_statistics_summary(data, args.output_dir)
        if plot_path:
            generated_plots.append(plot_path)
    
    if 'all' in args.plots or 'correlations' in args.plots:
        plot_path = plot_correlation_distributions(data, args.output_dir)
        if plot_path:
            generated_plots.append(plot_path)
    
    if 'all' in args.plots or 'report' in args.plots:
        report_path = generate_analysis_report(data, args.output_dir)
        if report_path:
            generated_plots.append(report_path)
    
    print(f"\n✅ Offline plotting complete!")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📊 Generated {len(generated_plots)} files:")
    for plot_path in generated_plots:
        print(f"  - {plot_path}")


if __name__ == "__main__":
    main() 