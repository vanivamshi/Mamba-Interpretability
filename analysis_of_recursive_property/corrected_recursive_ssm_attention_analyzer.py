"""
Corrected Recursive SSM-Attention-Neuron Analysis for Mamba Models

This module properly analyzes the recursive effects in Mamba models by combining:
1. Real SSM components (A, B, C matrices, delta parameters)
2. Synthetic attention vectors derived from hidden states
3. Neuron definitions based on these synthetic attention patterns
4. Cross-layer analysis of how these components evolve

The key improvement is using actual SSM components rather than assuming 
traditional attention mechanisms exist.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
import json
import os

# Import the provided components
from ssm_component_extractor import SSMComponentExtractor
from layer_correlation_analyzer import LayerCorrelationAnalyzer
from attention_neurons import MambaAttentionNeurons

# Optional plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not available. Visualizations will be skipped.")


class CorrectedRecursiveSSMAttentionNeuronAnalyzer:
    """
    Analyzes recursive effects in Mamba models by properly combining:
    - SSM component extraction (real Mamba components)
    - Synthetic attention vector creation from hidden states
    - Neuron behavior analysis based on synthetic attention
    - Cross-layer correlation analysis
    """
    
    def __init__(self, model, device=None, max_sequence_length=512, batch_size=1):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        
        # Memory management settings
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self.memory_cleanup_enabled = True
        
        # Initialize component analyzers with the provided implementations
        self.ssm_extractor = SSMComponentExtractor(model, device)
        self.layer_analyzer = LayerCorrelationAnalyzer(model, device)
        self.attention_analyzer = MambaAttentionNeurons(model, enable_attention_computation=True)
        
        # Storage for analysis results
        self.analysis_results = {}
        self.ssm_data = {}
        self.attention_data = {}
        self.neuron_data = {}
        self.correlation_data = {}
        
        # Memory tracking
        self.memory_stats = {
            'peak_memory_usage': 0,
            'cleanup_count': 0,
            'tensor_count': 0
        }
        
    def analyze_recursive_ssm_attention_effects(self, input_texts: List[str], 
                                              layer_indices: List[int] = None) -> Dict:
        """
        Comprehensive analysis combining real SSM components with synthetic attention neurons.
        
        Args:
            input_texts: List of input texts to analyze
            layer_indices: Layers to analyze (default: [0, 3, 6, 9, 12])
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        if layer_indices is None:
            layer_indices = [0, 3, 6, 9, 12]
            
        print("🔬 Corrected Recursive SSM-Attention-Neuron Analysis")
        print("=" * 60)
        print(f"📝 Analyzing {len(input_texts)} input texts")
        print(f"🔍 Layer indices: {layer_indices}")
        
        # Use batch processing for memory efficiency
        comprehensive_results = self._process_text_batch(input_texts, layer_indices)
        
        # Step 8: Cross-text analysis
        print("\n8️⃣ Performing cross-text analysis...")
        cross_text_analysis = self._analyze_cross_text_patterns(comprehensive_results)
        
        final_results = {
            'individual_texts': comprehensive_results,
            'cross_text_analysis': cross_text_analysis,
            'analysis_metadata': {
                'layer_indices': layer_indices,
                'num_texts': len(input_texts),
                'methods_used': ['ssm_extraction', 'synthetic_attention', 'neuron_creation', 'correlation_analysis']
            }
        }
        
        self.analysis_results = final_results
        return final_results
    
    def _get_tokenizer(self):
        """Get tokenizer with proper fallback."""
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except ImportError:
            # Create a simple mock tokenizer for testing
            class MockTokenizer:
                def __call__(self, text, **kwargs):
                    # Simple word-based tokenization
                    words = text.split()
                    max_length = kwargs.get('max_length', 128)
                    token_ids = [hash(word) % 1000 for word in words[:max_length]]
                    
                    return {
                        'input_ids': torch.tensor([token_ids])
                    }
            return MockTokenizer()
    
    def _correlate_ssm_with_attention(self, ssm_components: Dict, attention_data: Dict, 
                                    layer_indices: List[int]) -> Dict:
        """Correlate real SSM components with synthetic attention patterns."""
        correlations = {}
        
        for layer_idx in layer_indices:
            try:
                if layer_idx not in ssm_components or layer_idx not in attention_data:
                    print(f"Warning: Missing data for layer {layer_idx}")
                    continue
                    
                layer_correlations = {}
                ssm_layer = ssm_components[layer_idx]
                attn_layer = attention_data[layer_idx]
                
                # Validate data structures
                if not isinstance(ssm_layer, dict) or not isinstance(attn_layer, dict):
                    print(f"Warning: Invalid data structure for layer {layer_idx}")
                    continue
                
                # Correlate A matrix properties with attention patterns
                if ssm_layer.get('A_matrix') is not None and 'attention_vectors' in attn_layer:
                    A_matrix = ssm_layer['A_matrix']
                    attention_vectors = attn_layer['attention_vectors']
                    
                    # Validate tensor properties
                    if not isinstance(A_matrix, torch.Tensor) or not isinstance(attention_vectors, torch.Tensor):
                        print(f"Warning: Non-tensor data for layer {layer_idx}")
                        continue
                    
                    # Analyze A matrix spectral properties
                    if A_matrix.dim() == 2 and A_matrix.shape[0] == A_matrix.shape[1]:
                        try:
                            # Check for NaN or infinite values
                            if torch.any(torch.isnan(A_matrix)) or torch.any(torch.isinf(A_matrix)):
                                print(f"Warning: A matrix contains NaN/Inf values for layer {layer_idx}")
                                continue
                                
                            eigenvals = torch.linalg.eigvals(A_matrix).real
                            spectral_radius = torch.max(torch.abs(eigenvals)).item()
                            
                            # Validate spectral radius
                            if np.isnan(spectral_radius) or np.isinf(spectral_radius):
                                print(f"Warning: Invalid spectral radius for layer {layer_idx}")
                                continue
                            
                            # Compute attention concentration (entropy)
                            attn_flat = attention_vectors.flatten()
                            if attn_flat.numel() == 0:
                                print(f"Warning: Empty attention vectors for layer {layer_idx}")
                                continue
                                
                            attn_probs = torch.softmax(attn_flat, dim=0)
                            attention_entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-8)).item()
                            
                            # Validate entropy
                            if np.isnan(attention_entropy) or np.isinf(attention_entropy):
                                print(f"Warning: Invalid attention entropy for layer {layer_idx}")
                                continue
                            
                            layer_correlations['A_matrix_vs_attention'] = {
                                'spectral_radius': spectral_radius,
                                'attention_entropy': attention_entropy,
                                'stability_attention_ratio': spectral_radius / (attention_entropy + 1e-8)
                            }
                        except Exception as e:
                            print(f"Warning: Could not compute A-matrix eigenvalues for layer {layer_idx}: {e}")
                            
                # Correlate hidden state magnitude with attention strength
                if ssm_layer.get('hidden_states') is not None and 'attention_vectors' in attn_layer:
                    hidden_states = ssm_layer['hidden_states']
                    attention_vectors = attn_layer['attention_vectors']
                    
                    if isinstance(hidden_states, torch.Tensor) and isinstance(attention_vectors, torch.Tensor):
                        try:
                            # Compute hidden state magnitude over time
                            hidden_magnitude = torch.norm(hidden_states, dim=-1).mean().item()
                            attention_strength = torch.norm(attention_vectors).item()
                            
                            # Validate magnitudes
                            if (np.isnan(hidden_magnitude) or np.isinf(hidden_magnitude) or
                                np.isnan(attention_strength) or np.isinf(attention_strength)):
                                print(f"Warning: Invalid magnitudes for layer {layer_idx}")
                                continue
                            
                            layer_correlations['hidden_attention_correlation'] = {
                                'hidden_magnitude': hidden_magnitude,
                                'attention_strength': attention_strength,
                                'magnitude_attention_ratio': hidden_magnitude / (attention_strength + 1e-8)
                            }
                        except Exception as e:
                            print(f"Warning: Could not compute hidden-attention correlation for layer {layer_idx}: {e}")
                
                # Correlate delta parameters with attention dynamics
                if ssm_layer.get('delta_parameters') is not None and 'attention_vectors' in attn_layer:
                    delta_params = ssm_layer['delta_parameters']
                    attention_vectors = attn_layer['attention_vectors']
                    
                    if isinstance(delta_params, torch.Tensor) and isinstance(attention_vectors, torch.Tensor):
                        try:
                            delta_variation = torch.std(delta_params).item()
                            attention_variation = torch.std(attention_vectors).item()
                            
                            # Validate variations
                            if (np.isnan(delta_variation) or np.isinf(delta_variation) or
                                np.isnan(attention_variation) or np.isinf(attention_variation)):
                                print(f"Warning: Invalid variations for layer {layer_idx}")
                                continue
                            
                            layer_correlations['delta_attention_variation'] = {
                                'delta_variation': delta_variation,
                                'attention_variation': attention_variation,
                                'variation_correlation': abs(delta_variation - attention_variation)
                            }
                        except Exception as e:
                            print(f"Warning: Could not compute delta-attention variation for layer {layer_idx}: {e}")
                
                correlations[layer_idx] = layer_correlations
                
            except Exception as e:
                print(f"Error processing layer {layer_idx}: {e}")
                continue
            
        return correlations
    
    def _analyze_neuron_evolution(self, neurons: Dict, layer_indices: List[int]) -> Dict:
        """Analyze how neuron behavior evolves across layers."""
        evolution_analysis = {}
        
        # Extract neuron activations for each layer
        layer_activations = {}
        for layer_idx in layer_indices:
            if layer_idx in neurons and neurons[layer_idx] is not None:
                neuron_data = neurons[layer_idx]
                if 'neuron_activations' in neuron_data:
                    layer_activations[layer_idx] = neuron_data['neuron_activations']
        
        if len(layer_activations) < 2:
            return {'error': 'Insufficient layers with neuron data for evolution analysis'}
        
        # Analyze evolution between consecutive layers
        layer_pairs = []
        sorted_layers = sorted(layer_activations.keys())
        for i in range(len(sorted_layers) - 1):
            layer_pairs.append((sorted_layers[i], sorted_layers[i + 1]))
        
        for layer_i, layer_j in layer_pairs:
            activations_i = layer_activations[layer_i]
            activations_j = layer_activations[layer_j]
            
            # Ensure tensors have compatible shapes for comparison
            min_size = min(activations_i.numel(), activations_j.numel())
            act_i_flat = activations_i.flatten()[:min_size]
            act_j_flat = activations_j.flatten()[:min_size]
            
            # Compute evolution metrics
            pair_key = f"layer_{layer_i}_to_{layer_j}"
            
            # Cosine similarity between activation patterns
            cos_sim = torch.nn.functional.cosine_similarity(
                act_i_flat.unsqueeze(0), act_j_flat.unsqueeze(0)
            ).item()
            
            # Magnitude change
            mag_change = (torch.norm(act_j_flat) - torch.norm(act_i_flat)).item()
            
            # Activation pattern correlation
            if act_i_flat.std() > 1e-8 and act_j_flat.std() > 1e-8:
                correlation = torch.corrcoef(torch.stack([act_i_flat, act_j_flat]))[0, 1].item()
            else:
                correlation = 0.0
            
            evolution_analysis[pair_key] = {
                'cosine_similarity': cos_sim,
                'magnitude_change': mag_change,
                'activation_correlation': correlation,
                'evolution_stability': 1.0 - abs(1.0 - cos_sim)  # Stability measure
            }
        
        return evolution_analysis
    
    def _analyze_cross_text_patterns(self, results: Dict) -> Dict:
        """Analyze patterns across different input texts."""
        cross_analysis = {
            'ssm_consistency': {},
            'attention_consistency': {},
            'neuron_consistency': {},
            'correlation_patterns': {}
        }
        
        # Extract data across all texts for consistency analysis
        all_ssm_data = []
        all_attention_data = []
        all_neuron_data = []
        
        for text_key, text_data in results.items():
            analysis = text_data['analysis']
            all_ssm_data.append(analysis.get('ssm_components', {}))
            all_attention_data.append(analysis.get('attention_data', {}))
            all_neuron_data.append(analysis.get('neurons', {}))
        
        # Analyze SSM consistency across texts
        if all_ssm_data:
            cross_analysis['ssm_consistency'] = self._compute_cross_text_ssm_consistency(all_ssm_data)
        
        # Analyze attention pattern consistency
        if all_attention_data:
            cross_analysis['attention_consistency'] = self._compute_cross_text_attention_consistency(all_attention_data)
        
        # Analyze neuron behavior consistency
        if all_neuron_data:
            cross_analysis['neuron_consistency'] = self._compute_cross_text_neuron_consistency(all_neuron_data)
        
        return cross_analysis
    
    def _compute_cross_text_ssm_consistency(self, ssm_data_list: List[Dict]) -> Dict:
        """Compute consistency of SSM components across different texts."""
        consistency_analysis = {}
        
        # Get common layers across all texts
        common_layers = None
        for ssm_data in ssm_data_list:
            current_layers = set(ssm_data.keys())
            if common_layers is None:
                common_layers = current_layers
            else:
                common_layers = common_layers.intersection(current_layers)
        
        if not common_layers:
            return {'error': 'No common layers found across texts'}
        
        # Analyze consistency for each common layer
        for layer_idx in common_layers:
            layer_consistency = {}
            
            # Extract A matrix properties across texts
            spectral_radii = []
            a_norms = []
            
            for ssm_data in ssm_data_list:
                if layer_idx in ssm_data and ssm_data[layer_idx].get('A_matrix') is not None:
                    A_matrix = ssm_data[layer_idx]['A_matrix']
                    a_norms.append(torch.norm(A_matrix).item())
                    
                    if A_matrix.dim() == 2 and A_matrix.shape[0] == A_matrix.shape[1]:
                        try:
                            eigenvals = torch.linalg.eigvals(A_matrix).real
                            spectral_radius = torch.max(torch.abs(eigenvals)).item()
                            spectral_radii.append(spectral_radius)
                        except:
                            pass
            
            if spectral_radii:
                layer_consistency['spectral_radius_consistency'] = {
                    'mean': np.mean(spectral_radii),
                    'std': np.std(spectral_radii),
                    'coefficient_of_variation': np.std(spectral_radii) / (np.mean(spectral_radii) + 1e-8)
                }
            
            if a_norms:
                layer_consistency['a_matrix_norm_consistency'] = {
                    'mean': np.mean(a_norms),
                    'std': np.std(a_norms),
                    'coefficient_of_variation': np.std(a_norms) / (np.mean(a_norms) + 1e-8)
                }
            
            consistency_analysis[f'layer_{layer_idx}'] = layer_consistency
        
        return consistency_analysis
    
    def _compute_cross_text_attention_consistency(self, attention_data_list: List[Dict]) -> Dict:
        """Compute consistency of synthetic attention patterns across texts."""
        consistency_analysis = {}
        
        # Get common layers
        common_layers = None
        for attn_data in attention_data_list:
            current_layers = set(attn_data.keys())
            if common_layers is None:
                common_layers = current_layers
            else:
                common_layers = common_layers.intersection(current_layers)
        
        if not common_layers:
            return {'error': 'No common layers found across texts'}
        
        # Analyze attention pattern consistency
        for layer_idx in common_layers:
            layer_consistency = {}
            
            attention_entropies = []
            attention_strengths = []
            
            for attn_data in attention_data_list:
                if layer_idx in attn_data and 'attention_vectors' in attn_data[layer_idx]:
                    attention_vectors = attn_data[layer_idx]['attention_vectors']
                    
                    # Compute attention entropy
                    attn_flat = attention_vectors.flatten()
                    attn_probs = torch.softmax(attn_flat, dim=0)
                    entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-8)).item()
                    attention_entropies.append(entropy)
                    
                    # Compute attention strength
                    strength = torch.norm(attention_vectors).item()
                    attention_strengths.append(strength)
            
            if attention_entropies:
                layer_consistency['attention_entropy_consistency'] = {
                    'mean': np.mean(attention_entropies),
                    'std': np.std(attention_entropies),
                    'coefficient_of_variation': np.std(attention_entropies) / (np.mean(attention_entropies) + 1e-8)
                }
            
            if attention_strengths:
                layer_consistency['attention_strength_consistency'] = {
                    'mean': np.mean(attention_strengths),
                    'std': np.std(attention_strengths),
                    'coefficient_of_variation': np.std(attention_strengths) / (np.mean(attention_strengths) + 1e-8)
                }
            
            consistency_analysis[f'layer_{layer_idx}'] = layer_consistency
        
        return consistency_analysis
    
    def _compute_cross_text_neuron_consistency(self, neuron_data_list: List[Dict]) -> Dict:
        """Compute consistency of neuron behavior across texts."""
        consistency_analysis = {}
        
        # Get common layers
        common_layers = None
        for neuron_data in neuron_data_list:
            current_layers = set(neuron_data.keys())
            if common_layers is None:
                common_layers = current_layers
            else:
                common_layers = common_layers.intersection(current_layers)
        
        if not common_layers:
            return {'error': 'No common layers found across texts'}
        
        # Analyze neuron consistency
        for layer_idx in common_layers:
            layer_consistency = {}
            
            mean_activations = []
            activation_stds = []
            
            for neuron_data in neuron_data_list:
                if (layer_idx in neuron_data and 
                    neuron_data[layer_idx] is not None and 
                    'neuron_activations' in neuron_data[layer_idx]):
                    
                    activations = neuron_data[layer_idx]['neuron_activations']
                    mean_activations.append(activations.mean().item())
                    activation_stds.append(activations.std().item())
            
            if mean_activations:
                layer_consistency['mean_activation_consistency'] = {
                    'mean': np.mean(mean_activations),
                    'std': np.std(mean_activations),
                    'coefficient_of_variation': np.std(mean_activations) / (np.mean(mean_activations) + 1e-8)
                }
            
            if activation_stds:
                layer_consistency['activation_variability_consistency'] = {
                    'mean': np.mean(activation_stds),
                    'std': np.std(activation_stds),
                    'coefficient_of_variation': np.std(activation_stds) / (np.mean(activation_stds) + 1e-8)
                }
            
            consistency_analysis[f'layer_{layer_idx}'] = layer_consistency
        
        return consistency_analysis
    
    def visualize_analysis_results(self, save_dir: str = "corrected_mamba_analysis"):
        """Create comprehensive visualizations of the analysis results."""
        os.makedirs(save_dir, exist_ok=True)
        
        if not self.analysis_results:
            print("No analysis results available. Run analyze_recursive_ssm_attention_effects first.")
            return
        
        if not HAS_PLOTTING:
            print("Matplotlib not available. Skipping visualizations.")
            return
        
        print(f"Creating visualizations in {save_dir}/")
        
        # 1. SSM Component Analysis
        self._plot_ssm_analysis(save_dir)
        
        # 2. Synthetic Attention Analysis
        self._plot_attention_analysis(save_dir)
        
        # 3. Neuron Evolution Analysis
        self._plot_neuron_evolution(save_dir)
        
        # 4. Cross-layer Correlation Analysis
        self._plot_correlation_analysis(save_dir)
        
        # 5. Cross-text Consistency Analysis
        self._plot_consistency_analysis(save_dir)
        
        # 6. Recursion-Attention Effects Analysis
        self._plot_recursion_attention_effects(save_dir)
        
        # 7. Recursive State vs Activation Change Analysis
        self.plot_recursive_state_vs_activation_change(save_dir)
        
        print(f"Visualizations saved to {save_dir}/")
    
    def _plot_ssm_analysis(self, save_dir: str):
        """Plot SSM component analysis."""
        if not HAS_PLOTTING:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SSM Component Analysis', fontsize=16)
        
        # Extract SSM data from results
        spectral_radii = []
        layer_indices = []
        
        for text_key, text_data in self.analysis_results['individual_texts'].items():
            ssm_components = text_data['analysis'].get('ssm_components', {})
            for layer_idx, layer_data in ssm_components.items():
                if layer_data.get('A_matrix') is not None:
                    A_matrix = layer_data['A_matrix']
                    if A_matrix.dim() == 2 and A_matrix.shape[0] == A_matrix.shape[1]:
                        try:
                            eigenvals = torch.linalg.eigvals(A_matrix).real
                            spectral_radius = torch.max(torch.abs(eigenvals)).item()
                            spectral_radii.append(spectral_radius)
                            layer_indices.append(layer_idx)
                        except:
                            pass
        
        if spectral_radii:
            axes[0, 0].scatter(layer_indices, spectral_radii, alpha=0.7)
            axes[0, 0].set_title('Spectral Radius by Layer')
            axes[0, 0].set_xlabel('Layer Index')
            axes[0, 0].set_ylabel('Spectral Radius')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add stability line
            axes[0, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Stability Threshold')
            axes[0, 0].legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/ssm_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_attention_analysis(self, save_dir: str):
        """Plot synthetic attention analysis."""
        if not HAS_PLOTTING:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Synthetic Attention Analysis', fontsize=16)
        
        # Extract attention data
        attention_entropies = []
        attention_strengths = []
        layer_indices = []
        
        for text_key, text_data in self.analysis_results['individual_texts'].items():
            attention_data = text_data['analysis'].get('attention_data', {})
            for layer_idx, layer_data in attention_data.items():
                if 'attention_vectors' in layer_data:
                    attention_vectors = layer_data['attention_vectors']
                    
                    # Compute entropy
                    attn_flat = attention_vectors.flatten()
                    attn_probs = torch.softmax(attn_flat, dim=0)
                    entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-8)).item()
                    
                    # Compute strength
                    strength = torch.norm(attention_vectors).item()
                    
                    attention_entropies.append(entropy)
                    attention_strengths.append(strength)
                    layer_indices.append(layer_idx)
        
        if attention_entropies and attention_strengths:
            axes[0, 0].scatter(layer_indices, attention_entropies, alpha=0.7, color='blue')
            axes[0, 0].set_title('Attention Entropy by Layer')
            axes[0, 0].set_xlabel('Layer Index')
            axes[0, 0].set_ylabel('Attention Entropy')
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].scatter(layer_indices, attention_strengths, alpha=0.7, color='red')
            axes[0, 1].set_title('Attention Strength by Layer')
            axes[0, 1].set_xlabel('Layer Index')
            axes[0, 1].set_ylabel('Attention Strength')
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[1, 0].scatter(attention_entropies, attention_strengths, alpha=0.7, color='green')
            axes[1, 0].set_title('Attention Entropy vs Strength')
            axes[1, 0].set_xlabel('Attention Entropy')
            axes[1, 0].set_ylabel('Attention Strength')
            axes[1, 0].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/attention_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_neuron_evolution(self, save_dir: str):
        """Plot neuron evolution analysis.""" 
        if not HAS_PLOTTING:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Neuron Evolution Analysis', fontsize=16)
        
        # Extract neuron evolution data
        cosine_similarities = []
        magnitude_changes = []
        evolution_stabilities = []
        
        for text_key, text_data in self.analysis_results['individual_texts'].items():
            neuron_evolution = text_data['analysis'].get('neuron_evolution', {})
            for pair_key, pair_data in neuron_evolution.items():
                if isinstance(pair_data, dict):
                    cosine_similarities.append(pair_data.get('cosine_similarity', 0))
                    magnitude_changes.append(pair_data.get('magnitude_change', 0))
                    evolution_stabilities.append(pair_data.get('evolution_stability', 0))
        
        if cosine_similarities:
            axes[0, 0].hist(cosine_similarities, bins=20, alpha=0.7, color='blue')
            axes[0, 0].set_title('Cosine Similarity Distribution')
            axes[0, 0].set_xlabel('Cosine Similarity')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)
            
        if magnitude_changes:
            axes[0, 1].hist(magnitude_changes, bins=20, alpha=0.7, color='red')
            axes[0, 1].set_title('Magnitude Change Distribution')
            axes[0, 1].set_xlabel('Magnitude Change')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
            
        if evolution_stabilities:
            axes[1, 0].hist(evolution_stabilities, bins=20, alpha=0.7, color='green')
            axes[1, 0].set_title('Evolution Stability Distribution')
            axes[1, 0].set_xlabel('Evolution Stability')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot evolution stability vs cosine similarity
        if cosine_similarities and evolution_stabilities:
            axes[1, 1].scatter(cosine_similarities, evolution_stabilities, alpha=0.7, color='purple')
            axes[1, 1].set_title('Evolution Stability vs Cosine Similarity')
            axes[1, 1].set_xlabel('Cosine Similarity')
            axes[1, 1].set_ylabel('Evolution Stability')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/neuron_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_analysis(self, save_dir: str):
        """Plot cross-layer correlation analysis."""
        if not HAS_PLOTTING:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cross-Layer Correlation Analysis', fontsize=16)
        
        # Extract correlation data
        mean_correlations = []
        max_correlations = []
        layer_pairs = []
        
        for text_key, text_data in self.analysis_results['individual_texts'].items():
            layer_correlations = text_data['analysis'].get('layer_correlations', {})
            for pair_key, corr_data in layer_correlations.items():
                if isinstance(corr_data, dict):
                    mean_correlations.append(corr_data.get('mean_correlation', 0))
                    max_correlations.append(corr_data.get('max_correlation', 0))
                    layer_pairs.append(pair_key)
        
        if mean_correlations and max_correlations:
            axes[0, 0].scatter(mean_correlations, max_correlations, alpha=0.7, color='blue')
            axes[0, 0].set_title('Mean vs Max Correlations')
            axes[0, 0].set_xlabel('Mean Correlation')
            axes[0, 0].set_ylabel('Max Correlation')
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].hist(mean_correlations, bins=20, alpha=0.7, color='red')
            axes[0, 1].set_title('Mean Correlation Distribution')
            axes[0, 1].set_xlabel('Mean Correlation')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[1, 0].hist(max_correlations, bins=20, alpha=0.7, color='green')
            axes[1, 0].set_title('Max Correlation Distribution')
            axes[1, 0].set_xlabel('Max Correlation')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/correlation_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_consistency_analysis(self, save_dir: str):
        """Plot cross-text consistency analysis."""
        if not HAS_PLOTTING:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cross-Text Consistency Analysis', fontsize=16)
        
        # Extract consistency data from cross-text analysis
        cross_text_data = self.analysis_results.get('cross_text_analysis', {})
        
        # Plot SSM consistency
        ssm_consistency = cross_text_data.get('ssm_consistency', {})
        if ssm_consistency:
            spectral_cvs = []
            layer_labels = []
            
            for layer_key, layer_data in ssm_consistency.items():
                if 'spectral_radius_consistency' in layer_data:
                    cv = layer_data['spectral_radius_consistency'].get('coefficient_of_variation', 0)
                    spectral_cvs.append(cv)
                    layer_labels.append(layer_key)
            
            if spectral_cvs:
                axes[0, 0].bar(range(len(spectral_cvs)), spectral_cvs, alpha=0.7, color='blue')
                axes[0, 0].set_title('SSM Spectral Radius Consistency')
                axes[0, 0].set_xlabel('Layer')
                axes[0, 0].set_ylabel('Coefficient of Variation')
                axes[0, 0].set_xticks(range(len(layer_labels)))
                axes[0, 0].set_xticklabels(layer_labels, rotation=45)
                axes[0, 0].grid(True, alpha=0.3)
        
        # Plot attention consistency
        attention_consistency = cross_text_data.get('attention_consistency', {})
        if attention_consistency:
            entropy_cvs = []
            strength_cvs = []
            layer_labels = []
            
            for layer_key, layer_data in attention_consistency.items():
                if 'attention_entropy_consistency' in layer_data:
                    entropy_cv = layer_data['attention_entropy_consistency'].get('coefficient_of_variation', 0)
                    entropy_cvs.append(entropy_cv)
                    
                if 'attention_strength_consistency' in layer_data:
                    strength_cv = layer_data['attention_strength_consistency'].get('coefficient_of_variation', 0)
                    strength_cvs.append(strength_cv)
                    
                layer_labels.append(layer_key)
            
            if entropy_cvs:
                axes[0, 1].bar(range(len(entropy_cvs)), entropy_cvs, alpha=0.7, color='red')
                axes[0, 1].set_title('Attention Entropy Consistency')
                axes[0, 1].set_xlabel('Layer')
                axes[0, 1].set_ylabel('Coefficient of Variation')
                axes[0, 1].set_xticks(range(len(layer_labels)))
                axes[0, 1].set_xticklabels(layer_labels, rotation=45)
                axes[0, 1].grid(True, alpha=0.3)
            
            if strength_cvs:
                axes[1, 0].bar(range(len(strength_cvs)), strength_cvs, alpha=0.7, color='green')
                axes[1, 0].set_title('Attention Strength Consistency')
                axes[1, 0].set_xlabel('Layer')
                axes[1, 0].set_ylabel('Coefficient of Variation')
                axes[1, 0].set_xticks(range(len(layer_labels)))
                axes[1, 0].set_xticklabels(layer_labels, rotation=45)
                axes[1, 0].grid(True, alpha=0.3)
        
        # Plot neuron consistency
        neuron_consistency = cross_text_data.get('neuron_consistency', {})
        if neuron_consistency:
            activation_cvs = []
            layer_labels = []
            
            for layer_key, layer_data in neuron_consistency.items():
                if 'mean_activation_consistency' in layer_data:
                    cv = layer_data['mean_activation_consistency'].get('coefficient_of_variation', 0)
                    activation_cvs.append(cv)
                    layer_labels.append(layer_key)
            
            if activation_cvs:
                axes[1, 1].bar(range(len(activation_cvs)), activation_cvs, alpha=0.7, color='purple')
                axes[1, 1].set_title('Neuron Activation Consistency')
                axes[1, 1].set_xlabel('Layer')
                axes[1, 1].set_ylabel('Coefficient of Variation')
                axes[1, 1].set_xticks(range(len(layer_labels)))
                axes[1, 1].set_xticklabels(layer_labels, rotation=45)
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/consistency_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_recursion_attention_effects(self, save_dir: str):
        """Plot how recursion affects attention vectors across layers."""
        if not HAS_PLOTTING:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('How Recursion Affects Attention Vectors Across Layers', fontsize=16)
        
        # Extract attention data from all texts
        layer_attention_data = {}
        layer_indices = []
        
        for text_key, text_data in self.analysis_results['individual_texts'].items():
            attention_data = text_data['analysis'].get('attention_data', {})
            for layer_idx, layer_attn in attention_data.items():
                if 'attention_vectors' in layer_attn:
                    if layer_idx not in layer_attention_data:
                        layer_attention_data[layer_idx] = []
                    layer_attention_data[layer_idx].append(layer_attn['attention_vectors'])
                    if layer_idx not in layer_indices:
                        layer_indices.append(layer_idx)
        
        layer_indices = sorted(layer_indices)
        
        # Plot 1: Attention entropy evolution (shows how focused attention becomes)
        entropies = []
        for layer_idx in layer_indices:
            layer_entropies = []
            for attn_vec in layer_attention_data[layer_idx]:
                attn_flat = attn_vec.flatten()
                attn_probs = torch.softmax(attn_flat, dim=0)
                entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-8)).item()
                layer_entropies.append(entropy)
            entropies.append(np.mean(layer_entropies))
        
        axes[0, 0].plot(layer_indices, entropies, 'o-', linewidth=2, markersize=8, color='blue')
        axes[0, 0].set_title('Attention Entropy Evolution\n(Higher = More Distributed)')
        axes[0, 0].set_xlabel('Layer Index')
        axes[0, 0].set_ylabel('Attention Entropy')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Attention strength evolution (shows magnitude changes)
        strengths = []
        for layer_idx in layer_indices:
            layer_strengths = []
            for attn_vec in layer_attention_data[layer_idx]:
                strength = torch.norm(attn_vec).item()
                layer_strengths.append(strength)
            strengths.append(np.mean(layer_strengths))
        
        axes[0, 1].plot(layer_indices, strengths, 'o-', linewidth=2, markersize=8, color='red')
        axes[0, 1].set_title('Attention Strength Evolution\n(Higher = Stronger Attention)')
        axes[0, 1].set_xlabel('Layer Index')
        axes[0, 1].set_ylabel('L2 Norm')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Attention similarity between consecutive layers (shows recursive propagation)
        similarities = []
        layer_pairs = []
        for i in range(len(layer_indices) - 1):
            layer_i = layer_indices[i]
            layer_j = layer_indices[i + 1]
            
            layer_similarities = []
            for attn_i, attn_j in zip(layer_attention_data[layer_i], layer_attention_data[layer_j]):
                # Compute cosine similarity
                flat_i = attn_i.flatten()
                flat_j = attn_j.flatten()
                similarity = torch.cosine_similarity(flat_i, flat_j, dim=0).item()
                layer_similarities.append(similarity)
            
            similarities.append(np.mean(layer_similarities))
            layer_pairs.append(f'{layer_i}→{layer_j}')
        
        axes[1, 0].bar(range(len(similarities)), similarities, alpha=0.7, color='green')
        axes[1, 0].set_title('Attention Similarity Between Consecutive Layers\n(Higher = More Recursive)')
        axes[1, 0].set_xlabel('Layer Transition')
        axes[1, 0].set_ylabel('Cosine Similarity')
        axes[1, 0].set_xticks(range(len(layer_pairs)))
        axes[1, 0].set_xticklabels(layer_pairs, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Attention stability (shows how much attention changes)
        stabilities = []
        for i in range(len(layer_indices) - 1):
            layer_i = layer_indices[i]
            layer_j = layer_indices[i + 1]
            
            layer_stabilities = []
            for attn_i, attn_j in zip(layer_attention_data[layer_i], layer_attention_data[layer_j]):
                # Compute relative change
                flat_i = attn_i.flatten()
                flat_j = attn_j.flatten()
                relative_change = torch.mean(torch.abs(flat_j - flat_i) / (torch.abs(flat_i) + 1e-8)).item()
                stability = 1.0 / (1.0 + relative_change)  # Higher stability = less change
                layer_stabilities.append(stability)
            
            stabilities.append(np.mean(layer_stabilities))
        
        axes[1, 1].bar(range(len(stabilities)), stabilities, alpha=0.7, color='purple')
        axes[1, 1].set_title('Attention Stability Across Layers\n(Higher = More Stable)')
        axes[1, 1].set_xlabel('Layer Transition')
        axes[1, 1].set_ylabel('Stability Score')
        axes[1, 1].set_xticks(range(len(layer_pairs)))
        axes[1, 1].set_xticklabels(layer_pairs, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/recursion_attention_effects.png", dpi=300, bbox_inches='tight')
        plt.close()
    


    def plot_recursive_state_vs_activation_change(self, save_dir: str = "recursive_analysis"):
        """
        Plot how recursive state changes affect activation values.
        Y-axis: Recursive state change measure
        X-axis: Activation value changes
        """
        if not HAS_PLOTTING:
            print("Matplotlib not available. Skipping visualization.")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Recursive State Changes vs Activation Value Changes', fontsize=16)
        
        # Collect data across all texts and layers
        recursive_changes = []
        activation_changes = []
        layer_labels = []
        text_labels = []
        
        print(f"Processing {len(self.analysis_results['individual_texts'])} texts for recursive state vs activation analysis")
        
        for text_key, text_data in self.analysis_results['individual_texts'].items():
            analysis = text_data['analysis']
            
            # Get SSM components for recursive state analysis
            ssm_components = analysis.get('ssm_components', {})
            # Get layer activations for activation change analysis
            layer_activations = analysis.get('layer_activations', {})
            
            layer_indices = sorted(set(ssm_components.keys()) & set(layer_activations.keys()))
            
            # Process all possible layer pairs to see full recursive effects
            for i in range(len(layer_indices)):
                for j in range(i + 1, len(layer_indices)):
                    current_layer = layer_indices[i]
                    next_layer = layer_indices[j]
                    
                    # Calculate recursive state change (using hidden states or matrix norms)
                    if (current_layer in ssm_components and next_layer in ssm_components):
                        try:
                            # Try to use hidden states first (most direct measure of recursive state)
                            if (ssm_components[current_layer].get('hidden_states') is not None and
                                ssm_components[next_layer].get('hidden_states') is not None):
                                
                                hidden_curr_data = ssm_components[current_layer]['hidden_states']
                                hidden_next_data = ssm_components[next_layer]['hidden_states']
                                
                                # Convert JSON data back to tensors if needed
                                if isinstance(hidden_curr_data, list):
                                    hidden_curr = torch.tensor(hidden_curr_data)
                                else:
                                    hidden_curr = hidden_curr_data
                                    
                                if isinstance(hidden_next_data, list):
                                    hidden_next = torch.tensor(hidden_next_data)
                                else:
                                    hidden_next = hidden_next_data
                                
                                # Measure recursive state change as norm difference of hidden states
                                recursive_change = torch.norm(hidden_next - hidden_curr).item()
                                
                            else:
                                # Fallback: use A matrix norm changes as proxy for recursive behavior
                                A_curr_data = ssm_components[current_layer]['A_matrix']
                                A_next_data = ssm_components[next_layer]['A_matrix']
                                
                                # Convert JSON data back to tensors if needed
                                if isinstance(A_curr_data, list):
                                    A_curr = torch.tensor(A_curr_data)
                                else:
                                    A_curr = A_curr_data
                                    
                                if isinstance(A_next_data, list):
                                    A_next = torch.tensor(A_next_data)
                                else:
                                    A_next = A_next_data
                                
                                # Use Frobenius norm difference as measure of recursive state change
                                norm_curr = torch.norm(A_curr).item()
                                norm_next = torch.norm(A_next).item()
                                
                                recursive_change = abs(norm_next - norm_curr)
                            
                            # Calculate activation value changes
                            if (current_layer in layer_activations and next_layer in layer_activations):
                                act_curr_data = layer_activations[current_layer]
                                act_next_data = layer_activations[next_layer]
                                
                                # Convert JSON data back to tensors if needed
                                if isinstance(act_curr_data, list):
                                    act_curr = torch.tensor(act_curr_data)
                                else:
                                    act_curr = act_curr_data
                                    
                                if isinstance(act_next_data, list):
                                    act_next = torch.tensor(act_next_data)
                                else:
                                    act_next = act_next_data
                                
                                # Ensure comparable shapes
                                min_size = min(act_curr.numel(), act_next.numel())
                                act_curr_flat = act_curr.flatten()[:min_size]
                                act_next_flat = act_next.flatten()[:min_size]
                                
                                # Activation change measure (L2 norm of difference)
                                activation_change = torch.norm(act_next_flat - act_curr_flat).item()
                                
                                recursive_changes.append(recursive_change)
                                activation_changes.append(activation_change)
                                layer_labels.append(f"L{current_layer}→L{next_layer}")
                                text_labels.append(text_key)
                                
                        except Exception as e:
                            print(f"Error processing layers {current_layer}->{next_layer}: {e}")
                            continue
        
        if not recursive_changes:
            print("No valid data found for recursive state vs activation analysis")
            print(f"Processed {len(layer_indices)} layers: {layer_indices}")
            print(f"Found SSM components for layers: {list(ssm_components.keys())}")
            print(f"Found layer activations for layers: {list(layer_activations.keys())}")
            return
        
        # Convert to numpy arrays for easier manipulation
        recursive_changes = np.array(recursive_changes)
        activation_changes = np.array(activation_changes)
        
        # Plot 1: Layer-wise recursive state changes (clear visualization)
        # Extract unique layer transitions and their data
        unique_transitions = list(set(layer_labels))
        unique_transitions.sort()  # Sort for consistent ordering
        
        # Calculate mean values for each transition across all texts
        transition_data = {}
        for transition in unique_transitions:
            transition_data[transition] = {
                'recursive_means': [],
                'activation_means': [],
                'recursive_stds': [],
                'activation_stds': []
            }
        
        # Group data by transition
        for i, label in enumerate(layer_labels):
            transition_data[label]['recursive_means'].append(recursive_changes[i])
            transition_data[label]['activation_means'].append(activation_changes[i])
        
        # Calculate statistics for each transition
        transition_stats = {}
        for transition, data in transition_data.items():
            if data['recursive_means']:
                transition_stats[transition] = {
                    'recursive_mean': np.mean(data['recursive_means']),
                    'recursive_std': np.std(data['recursive_means']),
                    'activation_mean': np.mean(data['activation_means']),
                    'activation_std': np.std(data['activation_means']),
                    'count': len(data['recursive_means'])
                }
        
        # Create a clear bar plot showing recursive state changes by layer transition
        transitions = list(transition_stats.keys())
        recursive_means = [transition_stats[t]['recursive_mean'] for t in transitions]
        recursive_stds = [transition_stats[t]['recursive_std'] for t in transitions]
        
        # Color bars by layer depth (darker = deeper layers)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(transitions)))
        
        bars = axes[0, 0].bar(range(len(transitions)), recursive_means, 
                             yerr=recursive_stds, capsize=5, alpha=0.7, 
                             color=colors, edgecolor='black', linewidth=1)
        
        # Customize the plot
        axes[0, 0].set_xlabel('Layer Transitions')
        axes[0, 0].set_ylabel('Recursive State Change (Hidden State Norm Δ)')
        axes[0, 0].set_title('Recursive State Changes Across Layer Transitions')
        axes[0, 0].set_xticks(range(len(transitions)))
        axes[0, 0].set_xticklabels(transitions, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, mean_val, std_val) in enumerate(zip(bars, recursive_means, recursive_stds)):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 0.5,
                           f'{mean_val:.1f}±{std_val:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Add summary statistics
        total_transitions = len(transitions)
        avg_recursive_change = np.mean(recursive_means)
        axes[0, 0].text(0.02, 0.98, f'Total Transitions: {total_transitions}\nAvg Change: {avg_recursive_change:.2f}', 
                        transform=axes[0, 0].transAxes, bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8),
                        verticalalignment='top', fontsize=9)
        
        # Plot 2: Activation changes across layer transitions
        activation_means = [transition_stats[t]['activation_mean'] for t in transitions]
        activation_stds = [transition_stats[t]['activation_std'] for t in transitions]
        
        bars2 = axes[0, 1].bar(range(len(transitions)), activation_means, 
                              yerr=activation_stds, capsize=5, alpha=0.7, 
                              color=colors, edgecolor='black', linewidth=1)
        
        axes[0, 1].set_xlabel('Layer Transitions')
        axes[0, 1].set_ylabel('Activation Change (L2 Norm)')
        axes[0, 1].set_title('Activation Changes Across Layer Transitions')
        axes[0, 1].set_xticks(range(len(transitions)))
        axes[0, 1].set_xticklabels(transitions, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, mean_val, std_val) in enumerate(zip(bars2, activation_means, activation_stds)):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_val + 1,
                           f'{mean_val:.1f}±{std_val:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Add summary statistics
        avg_activation_change = np.mean(activation_means)
        axes[0, 1].text(0.02, 0.98, f'Avg Activation Change: {avg_activation_change:.2f}', 
                        transform=axes[0, 1].transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen', alpha=0.8),
                        verticalalignment='top', fontsize=9)
        
        # Plot 3: Layer transition comparison - Recursive vs Activation changes side by side
        # Create a grouped bar chart for better comparison
        x_pos = np.arange(len(transitions))
        width = 0.35
        
        # Normalize values for better comparison (scale to 0-1)
        recursive_norm = np.array(recursive_means) / max(recursive_means) if max(recursive_means) > 0 else np.array(recursive_means)
        activation_norm = np.array(activation_means) / max(activation_means) if max(activation_means) > 0 else np.array(activation_means)
        
        bars1 = axes[1, 0].bar(x_pos - width/2, recursive_norm, width, 
                               label='Recursive Changes (Normalized)', alpha=0.8, 
                               color='skyblue', edgecolor='navy', linewidth=1)
        bars2 = axes[1, 0].bar(x_pos + width/2, activation_norm, width, 
                               label='Activation Changes (Normalized)', alpha=0.8, 
                               color='lightcoral', edgecolor='darkred', linewidth=1)
        
        # Customize the plot
        axes[1, 0].set_xlabel('Layer Transitions')
        axes[1, 0].set_ylabel('Normalized Values (0-1)')
        axes[1, 0].set_title('Comparison: Recursive vs Activation Changes by Layer Transition')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(transitions, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            # Add recursive change value
            axes[1, 0].text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.01,
                           f'{recursive_means[i]:.1f}', ha='center', va='bottom', 
                           fontsize=7, rotation=90)
            # Add activation change value
            axes[1, 0].text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.01,
                           f'{activation_means[i]:.1f}', ha='center', va='bottom', 
                           fontsize=7, rotation=90)
        
        # Add correlation info as text
        if len(recursive_changes) > 2:
            correlation = np.corrcoef(activation_changes, recursive_changes)[0, 1]
            correlation_strength = "Strong" if abs(correlation) > 0.7 else "Moderate" if abs(correlation) > 0.3 else "Weak"
            correlation_direction = "Positive" if correlation > 0 else "Negative"
            
            axes[1, 0].text(0.02, 0.98, 
                           f'Correlation: {correlation:.3f}\nStrength: {correlation_strength}\nDirection: {correlation_direction}', 
                           transform=axes[1, 0].transAxes, 
                           bbox=dict(boxstyle="round", facecolor='lightyellow', alpha=0.8),
                           verticalalignment='top', fontsize=9)
        
        # Plot 4: Summary statistics table
        axes[1, 1].axis('off')  # Turn off axis for table
        
        # Create summary data for the table
        summary_data = []
        for transition in transitions:
            stats = transition_stats[transition]
            summary_data.append([
                transition,
                f"{stats['recursive_mean']:.2f}",
                f"{stats['recursive_std']:.2f}",
                f"{stats['activation_mean']:.2f}",
                f"{stats['activation_std']:.2f}",
                str(stats['count'])
            ])
        
        # Create table
        table_data = [['Transition', 'Recursive\nMean', 'Recursive\nStd', 'Activation\nMean', 'Activation\nStd', 'Count']]
        table_data.extend(summary_data)
        
        table = axes[1, 1].table(cellText=table_data[1:], colLabels=table_data[0],
                                cellLoc='center', loc='center',
                                colWidths=[0.15, 0.15, 0.12, 0.15, 0.12, 0.08])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)
        
        # Color header row
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color alternating rows
        for i in range(1, len(table_data)):
            for j in range(len(table_data[0])):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        axes[1, 1].set_title('Summary Statistics by Layer Transition', pad=20, fontsize=12, weight='bold')
        
        plt.tight_layout()
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/recursive_state_vs_activation_changes.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Recursive state vs activation change plot saved to {save_dir}/recursive_state_vs_activation_changes.png")
        
        # Return summary statistics
        return {
            'correlation': np.corrcoef(activation_changes, recursive_changes)[0, 1] if len(recursive_changes) > 1 else 0,
            'num_data_points': len(recursive_changes),
            'mean_recursive_change': np.mean(recursive_changes),
            'mean_activation_change': np.mean(activation_changes),
            'recursive_change_std': np.std(recursive_changes),
            'activation_change_std': np.std(activation_changes)
        }




    def save_analysis_results(self, filepath: str = None):
        """Save analysis results to JSON file."""
        if not self.analysis_results:
            print("No analysis results to save. Run analyze_recursive_ssm_attention_effects first.")
            return
        
        if filepath is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"corrected_ssm_attention_analysis_{timestamp}.json"
        
        def convert_tensors(obj):
            """Convert PyTorch tensors to lists for JSON serialization."""
            if hasattr(obj, 'tolist'):  # PyTorch tensor
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(v) for v in obj]
            else:
                return obj
        
        with open(filepath, 'w') as f:
            json.dump(convert_tensors(self.analysis_results), f, indent=2)
        
        print(f"Analysis results saved to: {filepath}")
        return filepath
    
    def _cleanup_memory(self):
        """Clean up GPU memory and track usage."""
        if self.memory_cleanup_enabled:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                current_memory = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
                self.memory_stats['peak_memory_usage'] = max(
                    self.memory_stats['peak_memory_usage'], current_memory
                )
                self.memory_stats['cleanup_count'] += 1
    
    def _validate_tensor_shapes(self, tensor_dict: Dict, expected_shapes: Dict) -> bool:
        """Validate tensor shapes against expected dimensions."""
        for key, expected_shape in expected_shapes.items():
            if key in tensor_dict:
                tensor = tensor_dict[key]
                if isinstance(tensor, torch.Tensor):
                    if len(tensor.shape) != len(expected_shape):
                        print(f"Warning: Shape mismatch for {key}: expected {expected_shape}, got {tensor.shape}")
                        return False
                    for i, (actual_dim, expected_dim) in enumerate(zip(tensor.shape, expected_shape)):
                        if expected_dim is not None and actual_dim != expected_dim:
                            print(f"Warning: Dimension mismatch for {key}[{i}]: expected {expected_dim}, got {actual_dim}")
                            return False
        return True
    
    def _process_text_batch(self, texts: List[str], layer_indices: List[int]) -> Dict:
        """Process texts in batches to manage memory usage."""
        batch_results = {}
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            print(f"Processing batch {i//self.batch_size + 1}/{(len(texts) + self.batch_size - 1)//self.batch_size}")
            
            batch_text_results = {}
            for j, text in enumerate(batch_texts):
                text_idx = i + j
                print(f"📝 Processing text {text_idx + 1}/{len(texts)}: '{text[:50]}...'")
                
                try:
                    # Truncate text if too long
                    if len(text) > self.max_sequence_length:
                        text = text[:self.max_sequence_length]
                        print(f"Warning: Text truncated to {self.max_sequence_length} characters")
                    
                    text_results = self._analyze_single_text(text, layer_indices)
                    batch_text_results[f"text_{text_idx}"] = {
                        'text': text,
                        'analysis': text_results
                    }
                    
                    # Cleanup after each text
                    self._cleanup_memory()
                    
                except Exception as e:
                    print(f"Error processing text {text_idx}: {e}")
                    batch_text_results[f"text_{text_idx}"] = {
                        'text': text,
                        'analysis': {'error': str(e)}
                    }
            
            batch_results.update(batch_text_results)
        
        return batch_results
    
    def _analyze_single_text(self, text: str, layer_indices: List[int]) -> Dict:
        """Analyze a single text with memory management."""
        text_results = {}
        
        try:
            # Step 1: Extract real SSM components
            print("1️⃣ Extracting SSM components...")
            ssm_components = self.ssm_extractor.extract_ssm_components(layer_indices, text)
            text_results['ssm_components'] = ssm_components
            self._cleanup_memory()
            
            # Step 2: Extract synthetic attention vectors
            print("2️⃣ Extracting synthetic attention vectors...")
            tokenizer = self._get_tokenizer()
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_sequence_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            attention_data = self.attention_analyzer.extract_attention_vectors(
                inputs["input_ids"], layer_indices
            )
            text_results['attention_data'] = attention_data
            self._cleanup_memory()
            
            # Step 3: Create neurons from synthetic attention
            print("3️⃣ Creating neurons from synthetic attention...")
            neurons = self.attention_analyzer.create_mamba_neurons(
                attention_data, method='attention_weighted'
            )
            text_results['neurons'] = neurons
            self._cleanup_memory()
            
            # Step 4: Extract layer activations for correlation analysis
            print("4️⃣ Analyzing layer correlations...")
            layer_activations = self.layer_analyzer.extract_layer_activations(layer_indices, text)
            correlations = self.layer_analyzer.compute_cross_layer_correlations()
            text_results['layer_activations'] = layer_activations
            text_results['layer_correlations'] = correlations
            self._cleanup_memory()
            
            # Step 5: Analyze recursive patterns within layers
            print("5️⃣ Analyzing recursive patterns...")
            recursive_patterns = {}
            for layer_idx in layer_indices:
                if layer_idx in layer_activations:
                    patterns = self.layer_analyzer.analyze_recursive_patterns(layer_idx)
                    recursive_patterns[layer_idx] = patterns
            text_results['recursive_patterns'] = recursive_patterns
            self._cleanup_memory()
            
            # Step 6: Correlate SSM components with synthetic attention
            print("6️⃣ Correlating SSM components with synthetic attention...")
            ssm_attention_correlations = self._correlate_ssm_with_attention(
                ssm_components, attention_data, layer_indices
            )
            text_results['ssm_attention_correlations'] = ssm_attention_correlations
            self._cleanup_memory()
            
            # Step 7: Analyze neuron behavior evolution
            print("7️⃣ Analyzing neuron behavior evolution...")
            neuron_evolution = self._analyze_neuron_evolution(neurons, layer_indices)
            text_results['neuron_evolution'] = neuron_evolution
            self._cleanup_memory()
            
        except Exception as e:
            print(f"Error in single text analysis: {e}")
            text_results['error'] = str(e)
        
        return text_results
    
    def get_memory_stats(self) -> Dict:
        """Get current memory usage statistics."""
        stats = self.memory_stats.copy()
        if torch.cuda.is_available():
            stats['current_memory_gb'] = torch.cuda.memory_allocated(self.device) / 1024**3
            stats['max_memory_gb'] = torch.cuda.max_memory_allocated(self.device) / 1024**3
        return stats
    
    def reset_memory_stats(self):
        """Reset memory statistics."""
        self.memory_stats = {
            'peak_memory_usage': 0,
            'cleanup_count': 0,
            'tensor_count': 0
        }
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)


def demonstrate_corrected_analysis():
    """Demonstrate the corrected recursive SSM-attention-neuron analysis."""
    print("🚀 Corrected Recursive SSM-Attention-Neuron Analysis Demo")
    print("=" * 70)
    
    # Load model
    try:
        from transformers import AutoModelForCausalLM
        model_name = "state-spaces/mamba-130m-hf"
        print(f"📥 Loading model: {model_name}")
        
        model = AutoModelForCausalLM.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Initialize analyzer
        analyzer = CorrectedRecursiveSSMAttentionNeuronAnalyzer(model, device)
        
        # Load WikiText dataset
        print("📚 Loading WikiText dataset...")
        try:
            from datasets import load_dataset
            dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")
            
            # Extract text samples (first 3 samples for analysis)
            test_texts = []
            for i in range(min(3, len(dataset))):
                text = dataset[i]['text'].strip()
                if text and len(text) > 20:  # Filter out empty or very short texts
                    test_texts.append(text)
                    if len(test_texts) >= 3:
                        break
            
            print(f"✅ Loaded {len(test_texts)} texts from WikiText dataset")
            
        except ImportError:
            print("⚠️ datasets library not available, using fallback texts")
            test_texts = [
                "Mamba models demonstrate efficient recursive processing through state space models.",
                "The selective state update mechanism enables long sequence processing.",
                "Block-diagonal matrices provide computational efficiency while maintaining expressiveness."
            ]
        except Exception as e:
            print(f"❌ Error loading WikiText dataset: {e}")
            print("🔄 Using fallback texts")
            test_texts = [
                "Mamba models demonstrate efficient recursive processing through state space models.",
                "The selective state update mechanism enables long sequence processing.",
                "Block-diagonal matrices provide computational efficiency while maintaining expressiveness."
            ]
        
        print(f"📝 Analyzing {len(test_texts)} test texts")
        
        # Run analysis
        layer_indices = [0, 3, 6, 9, 12]
        results = analyzer.analyze_recursive_ssm_attention_effects(test_texts, layer_indices)
        
        # Create visualizations
        print("\n📊 Creating visualizations...")
        analyzer.visualize_analysis_results()
        
        # Save results
        print("\n💾 Saving analysis results...")
        analyzer.save_analysis_results()
        
        print("\n✅ Corrected recursive SSM-attention-neuron analysis complete!")
        print("📁 Check the generated files for detailed results and visualizations")
        
        return analyzer, results
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required dependencies: pip install transformers torch")
        return None, None
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return None, None


if __name__ == "__main__":
    demonstrate_corrected_analysis()
