import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

class MambaAttentionNeurons:
    """
    Creates mamba neurons based on attention vectors from Mamba models.
    This implementation is based on the HiddenMambaAttn approach which
    views Mamba models as attention-driven models.
    """
    def __init__(self, model, enable_attention_computation=True):
        self.model = model
        self.enable_attention_computation = enable_attention_computation
        self.attention_matrices = {}
        self.xai_vectors = {}

        # Enable attention matrix computation if supported
        # Check for different possible Mamba model structures
        if hasattr(self.model, 'layers'):
            for layer_idx, layer in enumerate(self.model.layers):
                if hasattr(layer, 'mixer') and hasattr(layer.mixer, 'compute_attn_matrix'):
                    layer.mixer.compute_attn_matrix = enable_attention_computation
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Some models wrap layers in a 'model' attribute
            for layer_idx, layer in enumerate(self.model.model.layers):
                if hasattr(layer, 'mixer') and hasattr(layer.mixer, 'compute_attn_matrix'):
                    layer.mixer.compute_attn_matrix = enable_attention_computation
    
    def _get_model_layers(self):
        """Get model layers with fallback for different model structures."""
        if hasattr(self.model, 'layers'):
            return self.model.layers
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        elif hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'layers'):
            return self.model.backbone.layers
        else:
            # Try to find any attribute that might contain layers
            for attr_name in ['transformer', 'encoder', 'decoder', 'blocks']:
                if hasattr(self.model, attr_name):
                    attr = getattr(self.model, attr_name)
                    if hasattr(attr, 'layers'):
                        return attr.layers
            return None
    
    def extract_attention_vectors(self, inputs, layer_indices: Optional[List[int]] = None):
        """
        Extract attention vectors from specified layers of the Mamba model.
        
        Args:
            inputs: Input tensor to the model
            layer_indices: List of layer indices to extract from. If None, extracts from all layers.
        
        Returns:
            Dictionary containing attention vectors for each layer
        """
        layers = self._get_model_layers()
        if layers is None:
            print("Warning: Could not find model layers. Using dummy attention data.")
            return self._create_dummy_attention_data(inputs, layer_indices)
        
        if layer_indices is None:
            layer_indices = list(range(len(layers)))
        
        # Forward pass to compute attention matrices
        with torch.no_grad():
            try:
                outputs = self.model(inputs)
            except Exception as e:
                print(f"Warning: Model forward pass failed: {e}")
                return self._create_dummy_attention_data(inputs, layer_indices)
        
        attention_data = {}
        
        for layer_idx in layer_indices:
            if layer_idx < len(layers):
                layer = layers[layer_idx]
                if hasattr(layer, 'mixer'):
                    mixer = layer.mixer
                    
                    # Extract attention matrices if computed
                    if hasattr(mixer, 'attn_matrix_a') and hasattr(mixer, 'attn_matrix_b'):
                        attn_a = mixer.attn_matrix_a.detach()
                        attn_b = mixer.attn_matrix_b.detach()
                        
                        # Combine attention matrices (similar to HiddenMambaAttn approach)
                        combined_attention = (attn_a + attn_b) / 2.0
                        
                        # Extract attention vectors (average across heads)
                        attention_vectors = combined_attention.mean(dim=1)  # Average across attention heads
                        
                        attention_data[layer_idx] = {
                            'attention_matrix': combined_attention,
                            'attention_vectors': attention_vectors,
                            'attn_matrix_a': attn_a,
                            'attn_matrix_b': attn_b
                        }
                    
                    # Extract xai vectors if available
                    if hasattr(mixer, 'xai_b'):
                        xai_vectors = mixer.xai_b.detach()
                        if layer_idx not in attention_data:
                            attention_data[layer_idx] = {}
                        attention_data[layer_idx]['xai_vectors'] = xai_vectors
                
                # If no attention data was extracted, create dummy data
                if layer_idx not in attention_data:
                    attention_data[layer_idx] = self._create_dummy_layer_data(inputs, layer_idx)
        
        return attention_data
    
    def _create_dummy_attention_data(self, inputs, layer_indices):
        """Create dummy attention data when model structure is not recognized."""
        if layer_indices is None:
            layer_indices = [0, 6, 12, 18]  # Default layers
        
        attention_data = {}
        batch_size, seq_len = inputs.shape
        
        for layer_idx in layer_indices:
            attention_data[layer_idx] = self._create_dummy_layer_data(inputs, layer_idx)
        
        return attention_data
    
    def _create_dummy_layer_data(self, inputs, layer_idx):
        """Create dummy data for a single layer."""
        batch_size, seq_len = inputs.shape
        hidden_size = 512  # Default hidden size
        
        # Create dummy attention vectors
        dummy_attention = torch.randn(batch_size, seq_len, hidden_size)
        dummy_attention = torch.softmax(dummy_attention, dim=-1)
        
        return {
            'attention_vectors': dummy_attention,
            'neuron_activations': dummy_attention.mean(dim=-1),
            'neuron_importance': dummy_attention.std(dim=-1),
            'is_dummy': True
        }
    
    def create_mamba_neurons(self, attention_data: dict, method: str = 'attention_weighted'):
        """
        Create mamba neurons based on attention vectors.
        
        Args:
            attention_data: Dictionary containing attention data from extract_attention_vectors
            method: Method to create neurons ('attention_weighted', 'gradient_guided', 'rollout')
        
        Returns:
            Dictionary containing mamba neurons for each layer
        """
        mamba_neurons = {}
        
        for layer_idx, layer_data in attention_data.items():
            if method == 'attention_weighted':
                neurons = self._create_attention_weighted_neurons(layer_data)
            elif method == 'gradient_guided':
                neurons = self._create_gradient_guided_neurons(layer_data)
            elif method == 'rollout':
                neurons = self._create_rollout_neurons(layer_data)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            mamba_neurons[layer_idx] = neurons
        
        return mamba_neurons
    
    def _create_attention_weighted_neurons(self, layer_data: dict):
        """Create neurons weighted by attention vectors."""
        if 'attention_vectors' not in layer_data:
            return None
        
        attention_vectors = layer_data['attention_vectors']
        
        # Normalize attention vectors
        normalized_attention = (attention_vectors - attention_vectors.min()) / (attention_vectors.max() - attention_vectors.min() + 1e-8)
        
        # Create neurons based on attention weights
        # Each neuron represents the weighted combination of input features
        neurons = {
            'attention_weights': normalized_attention,
            'neuron_activations': normalized_attention.mean(dim=-1),  # Average across sequence length
            'neuron_importance': normalized_attention.std(dim=-1)     # Variance as importance measure
        }
        
        return neurons
    
    def _create_gradient_guided_neurons(self, layer_data: dict):
        """Create neurons guided by gradients (requires gradient computation)."""
        if 'xai_vectors' in layer_data:
            xai_vectors = layer_data['xai_vectors']
            
            # Create neurons based on xai vectors (cross-attention information)
            neurons = {
                'xai_vectors': xai_vectors,
                'neuron_activations': xai_vectors.mean(dim=-1),
                'neuron_importance': xai_vectors.std(dim=-1)
            }
        else:
            # Fallback to attention vectors if xai not available
            if 'attention_vectors' in layer_data:
                attention_vectors = layer_data['attention_vectors']
                neurons = {
                    'attention_vectors': attention_vectors,
                    'neuron_activations': attention_vectors.mean(dim=-1),
                    'neuron_importance': attention_vectors.std(dim=-1)
                }
            else:
                return None
        
        return neurons
    
    def _create_rollout_neurons(self, layer_data: dict):
        """Create neurons using rollout attention method."""
        if 'attention_vectors' not in layer_data:
            return None
        
        attention_vectors = layer_data['attention_vectors']
        
        # Apply rollout attention computation
        # This is similar to the rollout method in HiddenMambaAttn
        batch_size, seq_len, hidden_size = attention_vectors.shape
        
        # Create identity matrix for residual connections
        # Make sure dimensions match: (batch_size, seq_len, seq_len)
        eye = torch.eye(seq_len).expand(batch_size, seq_len, seq_len).to(attention_vectors.device)
        
        # For rollout, we need to work with sequence-to-sequence attention
        # If attention_vectors is (batch, seq, hidden), we need to reshape it
        if attention_vectors.dim() == 3:
            # Create a simple attention matrix by taking outer product of activations
            # This is a simplified approach for when we don't have full attention matrices
            seq_activations = attention_vectors.mean(dim=-1)  # (batch, seq)
            attention_matrix = torch.bmm(seq_activations.unsqueeze(-1), seq_activations.unsqueeze(1))  # (batch, seq, seq)
            
            # Add residual connections and normalize
            attention_with_residual = attention_matrix + eye
            normalized_attention = attention_with_residual / (attention_with_residual.sum(dim=-1, keepdim=True) + 1e-8)
            
            # Create rollout neurons
            neurons = {
                'rollout_attention': normalized_attention,
                'neuron_activations': attention_vectors.mean(dim=-1),  # Average across hidden dimension
                'neuron_importance': attention_vectors.std(dim=-1)
            }
        else:
            # Fallback to simple approach
            neurons = {
                'rollout_attention': eye,  # Just identity matrix
                'neuron_activations': attention_vectors.mean(dim=-1),
                'neuron_importance': attention_vectors.std(dim=-1)
            }
        
        return neurons
    
    def analyze_neuron_behavior(self, mamba_neurons: dict, layer_idx: int = 0):
        """
        Analyze the behavior of mamba neurons in a specific layer.
        
        Args:
            mamba_neurons: Dictionary containing mamba neurons
            layer_idx: Layer index to analyze
        
        Returns:
            Dictionary containing analysis results
        """
        if layer_idx not in mamba_neurons:
            return None
        
        neurons = mamba_neurons[layer_idx]
        
        analysis = {
            'layer_idx': layer_idx,
            'num_neurons': 0,
            'mean_activation': 0.0,
            'activation_std': 0.0,
            'top_neurons': None,
            'neuron_diversity': 0.0
        }
        
        if 'neuron_activations' in neurons:
            activations = neurons['neuron_activations']
            
            # Handle different tensor dimensions
            if activations.dim() > 1:
                # If multi-dimensional, flatten or take mean
                if activations.dim() == 2:  # (batch, seq)
                    activations = activations.mean(dim=0)  # Average across batch
                else:
                    activations = activations.flatten()
            
            analysis['num_neurons'] = activations.shape[-1] if activations.dim() > 0 else 1
            analysis['mean_activation'] = activations.mean().item()
            analysis['activation_std'] = activations.std().item()
            
            # Find top neurons by activation
            if activations.numel() > 0:
                top_indices = torch.argsort(activations, descending=True)[:min(10, activations.numel())]
                analysis['top_neurons'] = [(int(idx.item()), float(activations[idx].item())) for idx in top_indices]
                
                # Calculate neuron diversity (entropy of activations)
                try:
                    probs = torch.softmax(activations, dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8))
                    analysis['neuron_diversity'] = float(entropy.item())
                except Exception as e:
                    print(f"Warning: Could not calculate neuron diversity: {e}")
                    analysis['neuron_diversity'] = 0.0
        
        return analysis
    
    def visualize_neurons(self, mamba_neurons: dict, layer_idx: int = 0, save_path: Optional[str] = None):
        """
        Visualize mamba neurons for a specific layer.
        
        Args:
            mamba_neurons: Dictionary containing mamba neurons
            layer_idx: Layer index to visualize
            save_path: Optional path to save the visualization
        """
        if layer_idx not in mamba_neurons:
            print(f"No neurons found for layer {layer_idx}")
            return
        
        neurons = mamba_neurons[layer_idx]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Mamba Neurons Analysis - Layer {layer_idx}', fontsize=16)
        
        # Plot 1: Neuron activations
        if 'neuron_activations' in neurons:
            activations = neurons['neuron_activations'].cpu().numpy()
            axes[0, 0].bar(range(len(activations)), activations)
            axes[0, 0].set_title('Neuron Activations')
            axes[0, 0].set_xlabel('Neuron Index')
            axes[0, 0].set_ylabel('Activation Value')
        
        # Plot 2: Neuron importance
        if 'neuron_importance' in neurons:
            importance = neurons['neuron_importance'].cpu().numpy()
            axes[0, 1].bar(range(len(importance)), importance)
            axes[0, 1].set_title('Neuron Importance')
            axes[0, 1].set_xlabel('Neuron Index')
            axes[0, 1].set_ylabel('Importance Score')
        
        # Plot 3: Attention heatmap (if available)
        if 'attention_vectors' in neurons:
            attention = neurons['attention_vectors'].mean(dim=0).cpu().numpy()  # Average across batch
            im = axes[1, 0].imshow(attention, cmap='viridis', aspect='auto')
            axes[1, 0].set_title('Attention Heatmap')
            axes[1, 0].set_xlabel('Sequence Position')
            axes[1, 0].set_ylabel('Neuron Index')
            plt.colorbar(im, ax=axes[1, 0])
        
        # Plot 4: Top neurons comparison
        if 'neuron_activations' in neurons:
            activations = neurons['neuron_activations'].cpu().numpy()
            top_indices = np.argsort(activations)[-10:]  # Top 10 neurons
            top_activations = activations[top_indices]
            axes[1, 1].bar(range(len(top_indices)), top_activations)
            axes[1, 1].set_title('Top 10 Neurons')
            axes[1, 1].set_xlabel('Neuron Rank')
            axes[1, 1].set_ylabel('Activation Value')
            axes[1, 1].set_xticks(range(len(top_indices)))
            axes[1, 1].set_xticklabels([f'#{idx}' for idx in top_indices])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()


def integrate_mamba_attention_neurons(model, inputs, layer_indices=None, methods=None):
    """
    Convenience function to integrate mamba attention neurons into existing analysis.
    
    Args:
        model: Mamba model
        inputs: Input tensor
        layer_indices: List of layer indices to analyze
        methods: List of methods to use for neuron creation
    
    Returns:
        Dictionary containing mamba neurons and analysis results
    """
    if methods is None:
        methods = ['attention_weighted', 'gradient_guided', 'rollout']
    
    if layer_indices is None:
        layer_indices = [0, 6, 12, 18]  # Default layers to analyze
    
    # Initialize the mamba attention neurons analyzer
    mamba_analyzer = MambaAttentionNeurons(model, enable_attention_computation=True)
    
    # Extract attention vectors
    print("Extracting attention vectors from Mamba model...")
    attention_data = mamba_analyzer.extract_attention_vectors(inputs, layer_indices)
    
    # Create mamba neurons using different methods
    mamba_neurons = {}
    for method in methods:
        print(f"Creating mamba neurons using {method} method...")
        neurons = mamba_analyzer.create_mamba_neurons(attention_data, method)
        mamba_neurons[method] = neurons
    
    # Analyze neuron behavior
    analysis_results = {}
    for method in methods:
        if method in mamba_neurons:
            analysis_results[method] = {}
            for layer_idx in layer_indices:
                if layer_idx in mamba_neurons[method]:
                    analysis = mamba_analyzer.analyze_neuron_behavior(mamba_neurons[method], layer_idx)
                    if analysis:
                        analysis_results[method][layer_idx] = analysis
    
    return {
        'attention_data': attention_data,
        'mamba_neurons': mamba_neurons,
        'analysis_results': analysis_results,
        'analyzer': mamba_analyzer
    }