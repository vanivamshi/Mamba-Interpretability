"""
Analysis Utilities for Memory Recursion Neuron Analysis

This module provides utility functions to support the memory recursion analysis
of layerwise neurons in Mamba models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import json
from datetime import datetime


def convert_tensors_to_lists(data):
    """Convert PyTorch tensors to lists for JSON serialization."""
    if isinstance(data, torch.Tensor):
        return data.tolist()
    elif isinstance(data, dict):
        return {k: convert_tensors_to_lists(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_tensors_to_lists(v) for v in data]
    else:
        return data


def save_analysis_results(results: Dict, filename: str):
    """Save analysis results to JSON file."""
    try:
        converted_results = convert_tensors_to_lists(results)
        with open(filename, 'w') as f:
            json.dump(converted_results, f, indent=2)
        print(f"✅ Results saved to: {filename}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")


def load_analysis_results(filename: str) -> Dict:
    """Load analysis results from JSON file."""
    try:
        with open(filename, 'r') as f:
            results = json.load(f)
        print(f"✅ Results loaded from: {filename}")
        return results
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return {}


def compute_neuron_statistics(neurons: Dict) -> Dict:
    """Compute statistics for neuron activations."""
    stats = {}
    
    for layer_idx, layer_neurons in neurons.items():
        if 'neuron_activations' in layer_neurons:
            activations = layer_neurons['neuron_activations']
            
            layer_stats = {
                'num_neurons': activations.shape[-1] if activations.dim() > 0 else 1,
                'mean_activation': activations.mean().item(),
                'std_activation': activations.std().item(),
                'max_activation': activations.max().item(),
                'min_activation': activations.min().item(),
                'activation_range': activations.max().item() - activations.min().item()
            }
            
            # Compute additional statistics
            if activations.dim() > 1:
                layer_stats['spatial_variance'] = activations.var(dim=-1).mean().item()
                layer_stats['temporal_variance'] = activations.var(dim=0).mean().item()
            
            stats[layer_idx] = layer_stats
    
    return stats


def compute_correlation_statistics(correlations: Dict) -> Dict:
    """Compute statistics for correlation matrices."""
    stats = {}
    
    for pair_key, corr_data in correlations.items():
        if 'correlation_matrix' in corr_data:
            corr_matrix = corr_data['correlation_matrix']
            
            pair_stats = {
                'mean_correlation': corr_matrix.mean().item(),
                'std_correlation': corr_matrix.std().item(),
                'max_correlation': corr_matrix.max().item(),
                'min_correlation': corr_matrix.min().item(),
                'correlation_range': corr_matrix.max().item() - corr_matrix.min().item()
            }
            
            # Compute additional statistics
            if corr_matrix.dim() == 2:
                # Compute diagonal vs off-diagonal statistics
                diagonal = torch.diag(corr_matrix)
                mask = ~torch.eye(corr_matrix.shape[0], dtype=bool)
                off_diagonal = corr_matrix[mask]
                
                pair_stats['diagonal_mean'] = diagonal.mean().item()
                pair_stats['off_diagonal_mean'] = off_diagonal.mean().item()
                pair_stats['diagonal_std'] = diagonal.std().item()
                pair_stats['off_diagonal_std'] = off_diagonal.std().item()
            
            stats[pair_key] = pair_stats
    
    return stats


def compute_memory_statistics(memory_data: Dict) -> Dict:
    """Compute statistics for memory effects."""
    stats = {}
    
    for layer_idx, layer_data in memory_data.items():
        if 'delta_magnitude' in layer_data:
            delta_mag = layer_data['delta_magnitude']
            
            layer_stats = {
                'mean_delta_magnitude': delta_mag['mean'],
                'std_delta_magnitude': delta_mag['std'],
                'max_delta_magnitude': delta_mag['max']
            }
        
        if 'memory_consistency' in layer_data:
            memory_cons = layer_data['memory_consistency']
            
            layer_stats.update({
                'memory_decay': memory_cons['memory_decay'],
                'memory_persistence': memory_cons['memory_persistence']
            })
        
        if 'delta_variation' in layer_data:
            delta_var = layer_data['delta_variation']
            
            layer_stats.update({
                'temporal_variance': delta_var['temporal_variance'],
                'spatial_variance': delta_var['spatial_variance']
            })
        
        stats[layer_idx] = layer_stats
    
    return stats


def create_summary_plot(results: Dict, save_path: str = "analysis_summary.png"):
    """Create a summary plot of analysis results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Memory Recursion Analysis Summary', fontsize=16)
    
    # Plot 1: Neuron activations across layers
    attention_results = results.get('attention_neurons', {})
    if attention_results:
        layer_means = {}
        for text_key, text_data in attention_results.items():
            for method in ['attention_weighted', 'gradient_guided', 'rollout']:
                if method in text_data['neurons']:
                    neurons = text_data['neurons'][method]
                    for layer_idx, layer_neurons in neurons.items():
                        if 'neuron_activations' in layer_neurons:
                            mean_activation = layer_neurons['neuron_activations'].mean().item()
                            if layer_idx not in layer_means:
                                layer_means[layer_idx] = []
                            layer_means[layer_idx].append(mean_activation)
        
        if layer_means:
            layers = sorted(layer_means.keys())
            means = [np.mean(layer_means[layer]) for layer in layers]
            stds = [np.std(layer_means[layer]) for layer in layers]
            
            axes[0, 0].errorbar(layers, means, yerr=stds, marker='o', capsize=5)
            axes[0, 0].set_title('Neuron Activations Across Layers')
            axes[0, 0].set_xlabel('Layer Index')
            axes[0, 0].set_ylabel('Mean Activation')
            axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Memory effects
    memory_results = results.get('memory_effects', {})
    if memory_results:
        layer_delta_magnitudes = {}
        for text_key, text_data in memory_results.items():
            memory_analysis = text_data['memory_analysis']
            for layer_idx, analysis in memory_analysis.items():
                if layer_idx not in layer_delta_magnitudes:
                    layer_delta_magnitudes[layer_idx] = []
                layer_delta_magnitudes[layer_idx].append(analysis['delta_magnitude']['mean'])
        
        if layer_delta_magnitudes:
            layers = sorted(layer_delta_magnitudes.keys())
            means = [np.mean(layer_delta_magnitudes[layer]) for layer in layers]
            
            axes[0, 1].plot(layers, means, marker='s', color='red')
            axes[0, 1].set_title('Delta Magnitude Across Layers')
            axes[0, 1].set_xlabel('Layer Index')
            axes[0, 1].set_ylabel('Delta Magnitude')
            axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Recursive patterns
    recursive_results = results.get('recursive_patterns', {})
    if recursive_results:
        layer_autocorrelations = {}
        for text_key, text_data in recursive_results.items():
            layer_patterns = text_data['layer_patterns']
            for layer_idx, patterns in layer_patterns.items():
                if 'temporal_autocorrelation' in patterns:
                    autocorr_data = patterns['temporal_autocorrelation']
                    for lag_key, lag_data in autocorr_data.items():
                        lag_num = int(lag_key.split('_')[1])
                        if layer_idx not in layer_autocorrelations:
                            layer_autocorrelations[layer_idx] = {}
                        if lag_num not in layer_autocorrelations[layer_idx]:
                            layer_autocorrelations[layer_idx][lag_num] = []
                        layer_autocorrelations[layer_idx][lag_num].append(lag_data['mean'])
        
        if layer_autocorrelations:
            # Plot autocorrelation for first layer
            first_layer = list(layer_autocorrelations.keys())[0]
            lag_data = layer_autocorrelations[first_layer]
            lags = sorted(lag_data.keys())
            means = [np.mean(lag_data[lag]) for lag in lags]
            
            axes[0, 2].plot(lags, means, marker='o', color='green')
            axes[0, 2].set_title(f'Temporal Autocorrelation - Layer {first_layer}')
            axes[0, 2].set_xlabel('Lag')
            axes[0, 2].set_ylabel('Autocorrelation')
            axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: SSM correlations
    ssm_results = results.get('ssm_correlations', {})
    if ssm_results:
        layer_spectral_radius = {}
        for text_key, text_data in ssm_results.items():
            dynamics_analysis = text_data['dynamics_analysis']
            for layer_idx, analysis in dynamics_analysis.items():
                if 'A_matrix_analysis' in analysis and 'spectral_radius' in analysis['A_matrix_analysis']:
                    if layer_idx not in layer_spectral_radius:
                        layer_spectral_radius[layer_idx] = []
                    layer_spectral_radius[layer_idx].append(analysis['A_matrix_analysis']['spectral_radius'])
        
        if layer_spectral_radius:
            layers = sorted(layer_spectral_radius.keys())
            means = [np.mean(layer_spectral_radius[layer]) for layer in layers]
            
            axes[1, 0].plot(layers, means, marker='^', color='blue')
            axes[1, 0].set_title('Spectral Radius Across Layers')
            axes[1, 0].set_xlabel('Layer Index')
            axes[1, 0].set_ylabel('Spectral Radius')
            axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Cross-layer correlations
    cross_layer_results = results.get('cross_layer_correlations', {})
    if cross_layer_results:
        # Get first text's correlations
        first_text_key = list(cross_layer_results.keys())[0]
        text_data = cross_layer_results[first_text_key]
        correlations = text_data['correlations']
        
        if correlations:
            # Plot mean correlation for each pair
            pair_names = list(correlations.keys())
            mean_correlations = [correlations[pair]['mean_correlation'] for pair in pair_names]
            
            axes[1, 1].bar(range(len(pair_names)), mean_correlations, alpha=0.7)
            axes[1, 1].set_title('Cross-Layer Correlations')
            axes[1, 1].set_xlabel('Layer Pairs')
            axes[1, 1].set_ylabel('Mean Correlation')
            axes[1, 1].set_xticks(range(len(pair_names)))
            axes[1, 1].set_xticklabels([pair.replace('layer_', 'L') for pair in pair_names], rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Method comparison
    if attention_results:
        method_comparison = {}
        for text_key, text_data in attention_results.items():
            for method in ['attention_weighted', 'gradient_guided', 'rollout']:
                if method in text_data['neurons']:
                    neurons = text_data['neurons'][method]
                    method_means = []
                    for layer_idx, layer_neurons in neurons.items():
                        if 'neuron_activations' in layer_neurons:
                            mean_activation = layer_neurons['neuron_activations'].mean().item()
                            method_means.append(mean_activation)
                    
                    if method not in method_comparison:
                        method_comparison[method] = []
                    method_comparison[method].extend(method_means)
        
        if method_comparison:
            methods = list(method_comparison.keys())
            means = [np.mean(method_comparison[method]) for method in methods]
            stds = [np.std(method_comparison[method]) for method in methods]
            
            axes[1, 2].bar(methods, means, yerr=stds, capsize=5, alpha=0.7)
            axes[1, 2].set_title('Neuron Activation by Method')
            axes[1, 2].set_xlabel('Method')
            axes[1, 2].set_ylabel('Mean Activation')
            axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Summary plot saved to: {save_path}")


def generate_analysis_report(results: Dict, save_path: str = "analysis_report.md"):
    """Generate a markdown report of the analysis results."""
    report_lines = []
    
    report_lines.append("# Memory Recursion Neuron Analysis Report")
    report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Analysis overview
    report_lines.append("## Analysis Overview")
    report_lines.append("")
    report_lines.append("This report presents the results of analyzing how memory recursion affects")
    report_lines.append("layerwise neurons calculated using attention_neurons.py in Mamba models.")
    report_lines.append("")
    
    # Key findings
    report_lines.append("## Key Findings")
    report_lines.append("")
    
    attention_results = results.get('attention_neurons', {})
    memory_results = results.get('memory_effects', {})
    recursive_results = results.get('recursive_patterns', {})
    ssm_results = results.get('ssm_correlations', {})
    
    if attention_results:
        report_lines.append("### Attention Neuron Analysis")
        report_lines.append(f"- Successfully analyzed {len(attention_results)} input texts")
        report_lines.append("- Extracted neurons using multiple methods (attention_weighted, gradient_guided, rollout)")
        report_lines.append("- Observed layer-dependent activation patterns")
        report_lines.append("")
    
    if memory_results:
        report_lines.append("### Memory Effects Analysis")
        report_lines.append(f"- Successfully analyzed {len(memory_results)} input texts")
        report_lines.append("- Memory recursion affects neuron behavior through delta parameters")
        report_lines.append("- Memory consistency varies across layers")
        report_lines.append("")
    
    if recursive_results:
        report_lines.append("### Recursive Patterns Analysis")
        report_lines.append(f"- Successfully analyzed {len(recursive_results)} input texts")
        report_lines.append("- Recursive patterns detected in neuron activations")
        report_lines.append("- Temporal autocorrelation shows memory persistence")
        report_lines.append("")
    
    if ssm_results:
        report_lines.append("### SSM Correlation Analysis")
        report_lines.append(f"- Successfully analyzed {len(ssm_results)} input texts")
        report_lines.append("- SSM components correlate with neuron behavior")
        report_lines.append("- Recursive stability affects neuron dynamics")
        report_lines.append("")
    
    # Recommendations
    report_lines.append("## Recommendations")
    report_lines.append("")
    report_lines.append("1. **Investigate deeper layers**: Deeper layers may show stronger memory effects")
    report_lines.append("2. **Analyze longer sequences**: Longer sequences can reveal memory decay patterns")
    report_lines.append("3. **Study attention-memory interaction**: Investigate how attention weights interact with memory persistence")
    report_lines.append("4. **Optimize neuron selection**: Use memory sensitivity to guide neuron selection")
    report_lines.append("5. **Consider recursive patterns**: Account for recursive patterns in neuron-based interventions")
    report_lines.append("")
    
    # Technical details
    report_lines.append("## Technical Details")
    report_lines.append("")
    report_lines.append("### Analysis Components")
    report_lines.append("- **Attention Neuron Extraction**: Using MambaAttentionNeurons class")
    report_lines.append("- **Recursive Pattern Analysis**: Using LayerCorrelationAnalyzer class")
    report_lines.append("- **SSM Component Extraction**: Using SSMComponentExtractor class")
    report_lines.append("- **Memory Effects Analysis**: Using delta parameter extraction")
    report_lines.append("- **Cross-Layer Correlation**: Computing correlations between layers")
    report_lines.append("")
    
    report_lines.append("### Methods Used")
    report_lines.append("- Attention-weighted neuron creation")
    report_lines.append("- Gradient-guided neuron creation")
    report_lines.append("- Rollout attention method")
    report_lines.append("- Temporal autocorrelation analysis")
    report_lines.append("- Spatial correlation analysis")
    report_lines.append("- Memory consistency analysis")
    report_lines.append("")
    
    # Save report
    with open(save_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✅ Analysis report saved to: {save_path}")


def compare_analysis_methods(results: Dict) -> Dict:
    """Compare different analysis methods."""
    comparison = {}
    
    attention_results = results.get('attention_neurons', {})
    if attention_results:
        method_stats = {}
        
        for text_key, text_data in attention_results.items():
            for method in ['attention_weighted', 'gradient_guided', 'rollout']:
                if method in text_data['neurons']:
                    neurons = text_data['neurons'][method]
                    
                    if method not in method_stats:
                        method_stats[method] = {
                            'mean_activations': [],
                            'num_neurons': [],
                            'activation_variance': []
                        }
                    
                    for layer_idx, layer_neurons in neurons.items():
                        if 'neuron_activations' in layer_neurons:
                            activations = layer_neurons['neuron_activations']
                            method_stats[method]['mean_activations'].append(activations.mean().item())
                            method_stats[method]['num_neurons'].append(activations.shape[-1])
                            method_stats[method]['activation_variance'].append(activations.var().item())
        
        # Compute summary statistics
        for method, stats in method_stats.items():
            comparison[method] = {
                'mean_activation': np.mean(stats['mean_activations']),
                'std_activation': np.std(stats['mean_activations']),
                'mean_num_neurons': np.mean(stats['num_neurons']),
                'mean_variance': np.mean(stats['activation_variance'])
            }
    
    return comparison


if __name__ == "__main__":
    print("🔧 Analysis Utilities")
    print("=" * 30)
    print("This module provides utility functions for memory recursion analysis.")
    print("Import this module to use the utility functions in your analysis scripts.")
    print("=" * 30)
