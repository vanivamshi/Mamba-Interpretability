"""
SSM Component Extractor for Mamba Models

This module extracts the core State Space Model components from Mamba layers
to understand the recursive properties and how they affect successive layers.

Key components extracted:
- A matrix: Controls recursive state transitions
- B matrix: Controls input influence on state
- C matrix: Projects hidden state to output
- Delta (Δ): Time-varying parameter modulating recursion
- Hidden states: Actual recursive state evolution

Run instructions
# Run individual components
python ssm_component_extractor.py
python layer_correlation_analyzer.py
python recursive_visualizer.py
python recursive_analysis_report.py

# Or use the classes directly
from ssm_component_extractor import SSMComponentExtractor
from layer_correlation_analyzer import LayerCorrelationAnalyzer
from recursive_visualizer import RecursiveVisualizer
from recursive_analysis_report import RecursiveAnalysisReporter
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from utils import get_model_layers

# Optional imports for visualization
import matplotlib.pyplot as plt
import seaborn as sns
HAS_PLOTTING = True


class SSMComponentExtractor:
    """
    Extracts State Space Model components from Mamba layers.
    
    This class hooks into Mamba model layers to capture the core SSM parameters
    that define the recursive behavior: A, B, C matrices, delta parameters,
    and hidden state evolution.
    """
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        
        # Storage for extracted components
        self.ssm_components = {}
        self.hidden_states = {}
        self.delta_parameters = {}
        
    def extract_ssm_components(self, layer_indices: List[int], input_text: str) -> Dict:
        """
        Extract SSM components from specified layers.
        
        Args:
            layer_indices: List of layer indices to analyze
            input_text: Input text to process
            
        Returns:
            Dictionary containing extracted SSM components for each layer
        """
        print(f"🔍 Extracting SSM components from layers: {layer_indices}")
        print(f"📝 Input text: '{input_text[:50]}...'")
        
        # Tokenize input
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model layers
        layers = get_model_layers(self.model)
        if not layers:
            raise ValueError("Could not find model layers")
        
        # Extract components from each layer
        for layer_idx in layer_indices:
            if layer_idx >= len(layers):
                print(f"⚠️ Layer {layer_idx} not found, skipping")
                continue
                
            print(f"\n📊 Extracting from layer {layer_idx}...")
            layer_components = self._extract_layer_components(layers[layer_idx], inputs, layer_idx)
            self.ssm_components[layer_idx] = layer_components
            
        return self.ssm_components
    
    def _extract_layer_components(self, layer, inputs, layer_idx):
        """Extract SSM components from a single layer."""
        components = {
            'A_matrix': None,
            'B_matrix': None, 
            'C_matrix': None,
            'delta_parameters': None,
            'hidden_states': None,
            'input_embeddings': None,
            'output_embeddings': None
        }
        
        # Storage for captured data
        captured_data = {}
        
        def ssm_hook(module, input, output):
            """Hook to capture SSM components during forward pass."""
            try:
                print(f"    🔍 Hooking module: {type(module).__name__}")
                print(f"    📏 Input shape: {[inp.shape if hasattr(inp, 'shape') else type(inp) for inp in input]}")
                print(f"    📏 Output shape: {output.shape if hasattr(output, 'shape') else type(output)}")
                
                # Try to access SSM components from the module
                if hasattr(module, 'A'):
                    captured_data['A_matrix'] = module.A.detach().clone()
                    print(f"    ✅ Found A matrix: {module.A.shape}")
                if hasattr(module, 'B'):
                    captured_data['B_matrix'] = module.B.detach().clone()
                    print(f"    ✅ Found B matrix: {module.B.shape}")
                if hasattr(module, 'C'):
                    captured_data['C_matrix'] = module.C.detach().clone()
                    print(f"    ✅ Found C matrix: {module.C.shape}")
                if hasattr(module, 'D'):
                    captured_data['D_matrix'] = module.D.detach().clone()
                    print(f"    ✅ Found D matrix: {module.D.shape}")
                    
                # Capture delta parameters if available
                if hasattr(module, 'delta_proj'):
                    print(f"    ✅ Found delta_proj: {module.delta_proj}")
                if hasattr(module, 'delta'):
                    captured_data['delta_parameters'] = module.delta.detach().clone()
                    print(f"    ✅ Found delta: {module.delta.shape}")
                    
                # Capture hidden states
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                    
                captured_data['hidden_states'] = hidden_states.detach().clone()
                captured_data['input_embeddings'] = input[0].detach().clone() if input else None
                print(f"    ✅ Captured hidden states: {hidden_states.shape}")
                
            except Exception as e:
                print(f"    ⚠️ Error in SSM hook: {e}")
        
        # Register hooks on different possible SSM modules
        hooks = []
        
        # Try to find SSM module within the layer
        ssm_module = None
        if hasattr(layer, 'mixer') and hasattr(layer.mixer, 'ssm'):
            ssm_module = layer.mixer.ssm
        elif hasattr(layer, 'ssm'):
            ssm_module = layer.ssm
        elif hasattr(layer, 'mixer'):
            ssm_module = layer.mixer
            
        if ssm_module:
            hook = ssm_module.register_forward_hook(ssm_hook)
            hooks.append(hook)
        else:
            # Fallback: hook the entire layer
            hook = layer.register_forward_hook(ssm_hook)
            hooks.append(hook)
        
        # Run forward pass
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Store captured components
            components.update(captured_data)
            
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        # If we didn't capture SSM matrices directly, try to extract them
        if components['A_matrix'] is None:
            print(f"    🔍 No A matrix found, trying to extract from layer weights...")
            components = self._extract_matrices_from_weights(layer, components)
        
        # Print what we actually captured
        print(f"    📋 Captured components:")
        for key, value in components.items():
            if value is not None:
                if hasattr(value, 'shape'):
                    print(f"      {key}: {value.shape}")
                else:
                    print(f"      {key}: {type(value)}")
            else:
                print(f"      {key}: None")
        
        return components
    
    def _extract_matrices_from_weights(self, layer, components):
        """Try to extract A, B, C matrices from layer weights."""
        try:
            print(f"    🔍 Searching for SSM parameters in layer...")
            # Look for SSM-related parameters in the layer
            for name, param in layer.named_parameters():
                print(f"      Found parameter: {name} -> {param.shape}")
                if 'A' in name and param.dim() >= 2:
                    components['A_matrix'] = param.detach().clone()
                    print(f"      ✅ Extracted A matrix: {param.shape}")
                elif 'B' in name and param.dim() >= 2:
                    components['B_matrix'] = param.detach().clone()
                    print(f"      ✅ Extracted B matrix: {param.shape}")
                elif 'C' in name and param.dim() >= 2:
                    components['C_matrix'] = param.detach().clone()
                    print(f"      ✅ Extracted C matrix: {param.shape}")
                elif 'delta' in name.lower():
                    components['delta_parameters'] = param.detach().clone()
                    print(f"      ✅ Extracted delta: {param.shape}")
                elif 'D' in name:
                    components['D_matrix'] = param.detach().clone()
                    print(f"      ✅ Extracted D matrix: {param.shape}")
                    
        except Exception as e:
            print(f"    ⚠️ Could not extract matrices from weights: {e}")
            
        return components
    
    def analyze_recursive_dynamics(self, layer_idx: int) -> Dict:
        """
        Analyze the recursive dynamics of a specific layer.
        
        Args:
            layer_idx: Layer index to analyze
            
        Returns:
            Dictionary containing recursive dynamics analysis
        """
        if layer_idx not in self.ssm_components:
            raise ValueError(f"No components extracted for layer {layer_idx}")
            
        components = self.ssm_components[layer_idx]
        analysis = {}
        
        # Analyze A matrix (state transition)
        if components['A_matrix'] is not None:
            A = components['A_matrix']
            analysis['A_matrix_analysis'] = {
                'shape': A.shape,
                'frobenius_norm': torch.norm(A, 'fro').item(),
                'mean_value': A.mean().item(),
                'std_value': A.std().item()
            }
            
            # Only compute eigenvalues if A is square
            if A.shape[0] == A.shape[1]:
                eigenvals = torch.linalg.eigvals(A).real
                analysis['A_matrix_analysis']['eigenvalues'] = eigenvals
                analysis['A_matrix_analysis']['spectral_radius'] = torch.max(torch.abs(eigenvals)).item()
                
                # Recursive stability analysis
                max_eigenval = torch.max(eigenvals).item()
                analysis['recursive_stability'] = {
                    'is_stable': max_eigenval < 1.0,
                    'max_eigenvalue': max_eigenval,
                    'stability_margin': 1.0 - max_eigenval
                }
            else:
                print(f"  ⚠️ A matrix is not square ({A.shape}), skipping eigenvalue analysis")
                analysis['recursive_stability'] = {
                    'is_stable': 'unknown',
                    'max_eigenvalue': 'N/A',
                    'stability_margin': 'N/A'
                }
        
        # Analyze B matrix (input influence)
        if components['B_matrix'] is not None:
            B = components['B_matrix']
            analysis['B_matrix_analysis'] = {
                'shape': B.shape,
                'frobenius_norm': torch.norm(B, 'fro').item(),
                'mean_value': B.mean().item(),
                'std_value': B.std().item(),
                'input_sensitivity': torch.norm(B, 'fro').item()
            }
        
        # Analyze C matrix (output projection)
        if components['C_matrix'] is not None:
            C = components['C_matrix']
            analysis['C_matrix_analysis'] = {
                'shape': C.shape,
                'frobenius_norm': torch.norm(C, 'fro').item(),
                'mean_value': C.mean().item(),
                'std_value': C.std().item(),
                'output_sensitivity': torch.norm(C, 'fro').item()
            }
        
        # Analyze hidden state evolution
        if components['hidden_states'] is not None:
            hidden_states = components['hidden_states']
            analysis['hidden_state_analysis'] = self._analyze_hidden_state_evolution(hidden_states)
        
        return analysis
    
    def _analyze_hidden_state_evolution(self, hidden_states):
        """Analyze how hidden states evolve over time."""
        # hidden_states shape: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Take first batch for analysis
        h = hidden_states[0]  # [seq_len, hidden_dim]
        
        analysis = {
            'sequence_length': seq_len,
            'hidden_dimension': hidden_dim,
            'state_magnitude': {
                'mean': torch.norm(h, dim=1).mean().item(),
                'std': torch.norm(h, dim=1).std().item(),
                'max': torch.norm(h, dim=1).max().item(),
                'min': torch.norm(h, dim=1).min().item()
            },
            'state_variation': {
                'temporal_variance': torch.var(h, dim=0).mean().item(),
                'spatial_variance': torch.var(h, dim=1).mean().item()
            },
            'state_correlation': {
                'autocorrelation': self._compute_autocorrelation(h),
                'cross_correlation': self._compute_cross_correlation(h)
            }
        }
        
        return analysis
    
    def _compute_autocorrelation(self, h):
        """Compute temporal autocorrelation of hidden states."""
        # Compute correlation between consecutive states
        h_shifted = h[1:]  # h_{t+1}
        h_original = h[:-1]  # h_t
        
        # Flatten for correlation computation
        h_flat = h_original.flatten()
        h_shifted_flat = h_shifted.flatten()
        
        # Compute correlation coefficient
        correlation = torch.corrcoef(torch.stack([h_flat, h_shifted_flat]))[0, 1]
        return correlation.item() if not torch.isnan(correlation) else 0.0
    
    def _compute_cross_correlation(self, h):
        """Compute cross-correlation between different hidden dimensions."""
        # Compute correlation between different dimensions at each time step
        correlations = []
        for t in range(h.shape[0]):
            h_t = h[t]  # [hidden_dim]
            if h_t.shape[0] > 1:
                try:
                    # Compute correlation matrix for this time step
                    corr_matrix = torch.corrcoef(h_t.unsqueeze(0))
                    if corr_matrix.numel() > 1 and not torch.isnan(corr_matrix).any():
                        # Take mean of off-diagonal elements
                        if corr_matrix.dim() == 2 and corr_matrix.shape[0] > 1:
                            mask = ~torch.eye(corr_matrix.shape[0], dtype=bool)
                            mean_corr = corr_matrix[mask].mean()
                            correlations.append(mean_corr.item())
                except Exception as e:
                    # Skip this time step if correlation computation fails
                    continue
        
        return np.mean(correlations) if correlations else 0.0
    
    def visualize_ssm_components(self, layer_idx: int, save_path: str = None):
        """Visualize the extracted SSM components."""
        if layer_idx not in self.ssm_components:
            raise ValueError(f"No components extracted for layer {layer_idx}")
            
        components = self.ssm_components[layer_idx]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'SSM Components - Layer {layer_idx}', fontsize=16)
        
        # Plot A matrix
        if components['A_matrix'] is not None:
            A = components['A_matrix'].cpu().numpy()
            if A.ndim > 2:
                A = A.reshape(A.shape[-2], A.shape[-1])
            sns.heatmap(A, ax=axes[0, 0], cmap='RdBu_r', center=0)
            axes[0, 0].set_title('A Matrix (State Transition)')
            axes[0, 0].set_xlabel('State Dimension')
            axes[0, 0].set_ylabel('State Dimension')
        
        # Plot B matrix
        if components['B_matrix'] is not None:
            B = components['B_matrix'].cpu().numpy()
            if B.ndim > 2:
                B = B.reshape(B.shape[-2], B.shape[-1])
            sns.heatmap(B, ax=axes[0, 1], cmap='RdBu_r', center=0)
            axes[0, 1].set_title('B Matrix (Input Influence)')
            axes[0, 1].set_xlabel('Input Dimension')
            axes[0, 1].set_ylabel('State Dimension')
        
        # Plot C matrix
        if components['C_matrix'] is not None:
            C = components['C_matrix'].cpu().numpy()
            if C.ndim > 2:
                C = C.reshape(C.shape[-2], C.shape[-1])
            sns.heatmap(C, ax=axes[1, 0], cmap='RdBu_r', center=0)
            axes[1, 0].set_title('C Matrix (Output Projection)')
            axes[1, 0].set_xlabel('State Dimension')
            axes[1, 0].set_ylabel('Output Dimension')
        
        # Plot hidden state evolution
        if components['hidden_states'] is not None:
            h = components['hidden_states'][0].cpu().numpy()  # [seq_len, hidden_dim]
            # Plot first few dimensions over time
            for i in range(min(5, h.shape[1])):
                axes[1, 1].plot(h[:, i], label=f'Dim {i}', alpha=0.7)
            axes[1, 1].set_title('Hidden State Evolution')
            axes[1, 1].set_xlabel('Time Step')
            axes[1, 1].set_ylabel('Activation Value')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 SSM components visualization saved to: {save_path}")
        else:
            plt.show()
        
        return fig


def demonstrate_ssm_extraction():
    """Demonstrate SSM component extraction on a Mamba model."""
    print("🚀 SSM Component Extraction Demo")
    print("=" * 50)
    
    # Load model
    from transformers import AutoModelForCausalLM
    model_name = "state-spaces/mamba-130m-hf"
    print(f"📥 Loading model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Initialize extractor
    extractor = SSMComponentExtractor(model, device)
    
    # Test input
    test_text = "The recursive nature of Mamba models allows them to process sequences efficiently."
    print(f"📝 Test input: '{test_text}'")
    
    # Extract components from first few layers
    layer_indices = [0, 1, 2]
    ssm_components = extractor.extract_ssm_components(layer_indices, test_text)
    
    # Analyze recursive dynamics
    print("\n🔬 Recursive Dynamics Analysis")
    print("-" * 40)
    
    for layer_idx in layer_indices:
        if layer_idx in ssm_components:
            print(f"\n📊 Layer {layer_idx}:")
            analysis = extractor.analyze_recursive_dynamics(layer_idx)
            
            # Print key findings
            if 'A_matrix_analysis' in analysis:
                A_analysis = analysis['A_matrix_analysis']
                print(f"  A Matrix: {A_analysis['shape']}")
                if 'spectral_radius' in A_analysis:
                    print(f"  Spectral Radius: {A_analysis['spectral_radius']:.4f}")
                else:
                    print(f"  Frobenius Norm: {A_analysis['frobenius_norm']:.4f}")
                
            if 'recursive_stability' in analysis:
                stability = analysis['recursive_stability']
                if stability['is_stable'] != 'unknown':
                    print(f"  Recursive Stability: {'✅ Stable' if stability['is_stable'] else '❌ Unstable'}")
                    print(f"  Stability Margin: {stability['stability_margin']:.4f}")
                else:
                    print(f"  Recursive Stability: Unknown (A matrix not square)")
                
            if 'hidden_state_analysis' in analysis:
                h_analysis = analysis['hidden_state_analysis']
                print(f"  Hidden States: {h_analysis['sequence_length']} timesteps, {h_analysis['hidden_dimension']} dims")
                print(f"  State Magnitude: {h_analysis['state_magnitude']['mean']:.4f} ± {h_analysis['state_magnitude']['std']:.4f}")
    
    # Visualize components
    print("\n🎨 Creating visualizations...")
    for layer_idx in layer_indices:
        if layer_idx in ssm_components:
            save_path = f"ssm_components_layer_{layer_idx}.png"
            extractor.visualize_ssm_components(layer_idx, save_path)
    
    print("\n✅ SSM component extraction complete!")
    return extractor, ssm_components


if __name__ == "__main__":
    demonstrate_ssm_extraction()
