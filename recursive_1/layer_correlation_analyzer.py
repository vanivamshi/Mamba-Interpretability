"""
Layer-wise Activation Correlation Analyzer for Mamba Models

This module analyzes how activations correlate across successive layers in Mamba,
studying the recursive effects and how information flows through the model.

Key analyses:
- Cross-layer activation correlations
- Recursive state evolution patterns
- Information flow between layers
- Temporal dependencies in hidden states
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from ssm_component_extractor import SSMComponentExtractor
from utils import get_model_layers


class LayerCorrelationAnalyzer:
    """
    Analyzes correlations between activations across Mamba layers.
    
    This class studies how the recursive properties of Mamba affect
    the flow of information between successive layers.
    """
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        
        # Storage for analysis results
        self.layer_activations = {}
        self.correlation_matrices = {}
        self.recursive_patterns = {}
        
    def extract_layer_activations(self, layer_indices: List[int], input_text: str) -> Dict:
        """
        Extract activations from multiple layers simultaneously.
        
        Args:
            layer_indices: List of layer indices to analyze
            input_text: Input text to process
            
        Returns:
            Dictionary containing activations for each layer
        """
        print(f"🔍 Extracting activations from layers: {layer_indices}")
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
        
        # Storage for activations from each layer
        layer_activations = {layer_idx: [] for layer_idx in layer_indices}
        
        def activation_hook_factory(layer_idx):
            """Create a hook function for a specific layer."""
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                # Store activation for this layer
                layer_activations[layer_idx].append(hidden_states.detach().clone())
                print(f"    📊 Layer {layer_idx}: captured activation {hidden_states.shape}")
                
            return hook
        
        # Register hooks for each layer
        hooks = []
        for layer_idx in layer_indices:
            if layer_idx < len(layers):
                layer = layers[layer_idx]
                hook = layer.register_forward_hook(activation_hook_factory(layer_idx))
                hooks.append(hook)
                print(f"  ✅ Registered hook for layer {layer_idx}")
        
        # Run forward pass
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
        finally:
            # Remove all hooks
            for hook in hooks:
                hook.remove()
        
        # Process and store activations
        for layer_idx in layer_indices:
            if layer_activations[layer_idx]:
                # Take the first (and only) activation for this layer
                activation = layer_activations[layer_idx][0]  # [batch, seq_len, hidden_dim]
                self.layer_activations[layer_idx] = activation
                print(f"  📋 Layer {layer_idx}: stored activation {activation.shape}")
            else:
                print(f"  ⚠️ No activation captured for layer {layer_idx}")
        
        return self.layer_activations
    
    def compute_cross_layer_correlations(self, layer_pairs: List[Tuple[int, int]] = None) -> Dict:
        """
        Compute correlations between activations across layer pairs.
        
        Args:
            layer_pairs: List of (layer_i, layer_j) pairs to analyze.
                        If None, analyzes consecutive layers.
        
        Returns:
            Dictionary containing correlation matrices for each layer pair
        """
        if not self.layer_activations:
            raise ValueError("No layer activations available. Run extract_layer_activations first.")
        
        if layer_pairs is None:
            # Analyze consecutive layers
            layer_indices = sorted(self.layer_activations.keys())
            layer_pairs = [(layer_indices[i], layer_indices[i+1]) 
                          for i in range(len(layer_indices)-1)]
        
        print(f"🔗 Computing correlations for layer pairs: {layer_pairs}")
        
        correlations = {}
        
        for layer_i, layer_j in layer_pairs:
            if layer_i not in self.layer_activations or layer_j not in self.layer_activations:
                print(f"  ⚠️ Skipping pair ({layer_i}, {layer_j}) - missing activations")
                continue
            
            print(f"  📊 Analyzing pair ({layer_i}, {layer_j})...")
            
            # Get activations for both layers
            act_i = self.layer_activations[layer_i]  # [batch, seq_len, hidden_dim]
            act_j = self.layer_activations[layer_j]
            
            # Flatten to [batch * seq_len, hidden_dim] for correlation computation
            batch_size, seq_len, hidden_dim = act_i.shape
            act_i_flat = act_i.view(-1, hidden_dim)  # [batch*seq_len, hidden_dim]
            act_j_flat = act_j.view(-1, hidden_dim)
            
            # Compute correlation matrix between the two layers
            correlation_matrix = self._compute_correlation_matrix(act_i_flat, act_j_flat)
            
            # Store results
            pair_key = f"layer_{layer_i}_to_{layer_j}"
            correlations[pair_key] = {
                'correlation_matrix': correlation_matrix,
                'mean_correlation': correlation_matrix.mean().item(),
                'max_correlation': correlation_matrix.max().item(),
                'min_correlation': correlation_matrix.min().item(),
                'std_correlation': correlation_matrix.std().item(),
                'layer_i_shape': act_i.shape,
                'layer_j_shape': act_j.shape
            }
            
            print(f"    Mean correlation: {correlations[pair_key]['mean_correlation']:.4f}")
            print(f"    Max correlation: {correlations[pair_key]['max_correlation']:.4f}")
        
        self.correlation_matrices = correlations
        return correlations
    
    def _compute_correlation_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute correlation matrix between two sets of activations.
        
        Args:
            X: First set of activations [N, D]
            Y: Second set of activations [N, D]
        
        Returns:
            Correlation matrix [D, D]
        """
        # Center the data
        X_centered = X - X.mean(dim=0, keepdim=True)
        Y_centered = Y - Y.mean(dim=0, keepdim=True)
        
        # Compute covariance
        cov_xy = torch.mm(X_centered.T, Y_centered) / (X.shape[0] - 1)
        
        # Compute standard deviations
        std_x = X_centered.std(dim=0, keepdim=True)
        std_y = Y_centered.std(dim=0, keepdim=True)
        
        # Avoid division by zero
        std_x = torch.clamp(std_x, min=1e-8)
        std_y = torch.clamp(std_y, min=1e-8)
        
        # Compute correlation matrix
        correlation_matrix = cov_xy / (std_x.T @ std_y)
        
        return correlation_matrix
    
    def analyze_recursive_patterns(self, layer_idx: int) -> Dict:
        """
        Analyze recursive patterns within a single layer's activations.
        
        Args:
            layer_idx: Layer index to analyze
        
        Returns:
            Dictionary containing recursive pattern analysis
        """
        if layer_idx not in self.layer_activations:
            raise ValueError(f"No activations available for layer {layer_idx}")
        
        print(f"🔄 Analyzing recursive patterns in layer {layer_idx}...")
        
        activations = self.layer_activations[layer_idx]  # [batch, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = activations.shape
        
        # Take first batch for analysis
        h = activations[0]  # [seq_len, hidden_dim]
        
        patterns = {
            'temporal_autocorrelation': self._compute_temporal_autocorrelation(h),
            'spatial_correlation': self._compute_spatial_correlation(h),
            'state_evolution': self._analyze_state_evolution(h),
            'recursive_memory': self._analyze_recursive_memory(h)
        }
        
        self.recursive_patterns[layer_idx] = patterns
        return patterns
    
    def _compute_temporal_autocorrelation(self, h: torch.Tensor) -> Dict:
        """Compute temporal autocorrelation of hidden states."""
        seq_len, hidden_dim = h.shape
        
        # Compute autocorrelation for different lags
        autocorrelations = {}
        for lag in range(1, min(seq_len, 10)):  # Up to lag 10
            if seq_len > lag:
                h_lag = h[lag:]  # h_{t+lag}
                h_orig = h[:-lag]  # h_t
                
                # Compute correlation for each dimension
                correlations = []
                for d in range(hidden_dim):
                    if h_orig[:, d].std() > 1e-8 and h_lag[:, d].std() > 1e-8:
                        corr = torch.corrcoef(torch.stack([h_orig[:, d], h_lag[:, d]]))[0, 1]
                        if not torch.isnan(corr):
                            correlations.append(corr.item())
                
                if correlations:
                    autocorrelations[f'lag_{lag}'] = {
                        'mean': np.mean(correlations),
                        'std': np.std(correlations),
                        'count': len(correlations)
                    }
        
        return autocorrelations
    
    def _compute_spatial_correlation(self, h: torch.Tensor) -> Dict:
        """Compute spatial correlation between different hidden dimensions."""
        seq_len, hidden_dim = h.shape
        
        # Compute correlation between dimensions at each time step
        spatial_correlations = []
        for t in range(seq_len):
            h_t = h[t]  # [hidden_dim]
            if h_t.std() > 1e-8:
                # Compute correlation matrix for this time step
                try:
                    corr_matrix = torch.corrcoef(h_t.unsqueeze(0))
                    if corr_matrix.numel() > 1 and not torch.isnan(corr_matrix).any():
                        # Take mean of off-diagonal elements
                        if corr_matrix.dim() == 2 and corr_matrix.shape[0] > 1:
                            mask = ~torch.eye(corr_matrix.shape[0], dtype=bool)
                            mean_corr = corr_matrix[mask].mean()
                            spatial_correlations.append(mean_corr.item())
                except:
                    continue
        
        return {
            'mean_spatial_correlation': np.mean(spatial_correlations) if spatial_correlations else 0.0,
            'std_spatial_correlation': np.std(spatial_correlations) if spatial_correlations else 0.0,
            'temporal_variation': np.std(spatial_correlations) if spatial_correlations else 0.0
        }
    
    def _analyze_state_evolution(self, h: torch.Tensor) -> Dict:
        """Analyze how the state evolves over time."""
        seq_len, hidden_dim = h.shape
        
        # Compute state magnitude over time
        state_magnitudes = torch.norm(h, dim=1)  # [seq_len]
        
        # Compute state changes
        state_changes = torch.norm(h[1:] - h[:-1], dim=1)  # [seq_len-1]
        
        # Compute state direction changes (curvature)
        if seq_len > 2:
            directions = h[1:] - h[:-1]  # [seq_len-1, hidden_dim]
            direction_changes = torch.norm(directions[1:] - directions[:-1], dim=1)  # [seq_len-2]
        else:
            direction_changes = torch.tensor([])
        
        return {
            'state_magnitude': {
                'mean': state_magnitudes.mean().item(),
                'std': state_magnitudes.std().item(),
                'trend': self._compute_trend(state_magnitudes.cpu().numpy())
            },
            'state_changes': {
                'mean': state_changes.mean().item(),
                'std': state_changes.std().item(),
                'max': state_changes.max().item()
            },
            'direction_changes': {
                'mean': direction_changes.mean().item() if len(direction_changes) > 0 else 0.0,
                'std': direction_changes.std().item() if len(direction_changes) > 0 else 0.0
            }
        }
    
    def _analyze_recursive_memory(self, h: torch.Tensor) -> Dict:
        """Analyze how much the model 'remembers' from early states."""
        seq_len, hidden_dim = h.shape
        
        # Compute how much later states correlate with early states
        memory_analysis = {}
        
        for early_t in range(min(5, seq_len)):  # Check first 5 time steps
            early_state = h[early_t]  # [hidden_dim]
            correlations = []
            
            for later_t in range(early_t + 1, seq_len):
                later_state = h[later_t]  # [hidden_dim]
                
                # Compute correlation between early and later states
                if early_state.std() > 1e-8 and later_state.std() > 1e-8:
                    corr = torch.corrcoef(torch.stack([early_state, later_state]))[0, 1]
                    if not torch.isnan(corr):
                        correlations.append(corr.item())
            
            if correlations:
                memory_analysis[f'from_t{early_t}'] = {
                    'mean_correlation': np.mean(correlations),
                    'max_correlation': np.max(correlations),
                    'memory_decay': np.mean(correlations) if correlations else 0.0
                }
        
        return memory_analysis
    
    def _compute_trend(self, values: np.ndarray) -> str:
        """Compute trend of a sequence of values."""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def generate_analysis_report(self) -> Dict:
        """Generate a comprehensive analysis report."""
        report = {
            'layer_activations': {
                layer_idx: {
                    'shape': list(activation.shape),
                    'mean_magnitude': torch.norm(activation).item() / activation.numel(),
                    'std_magnitude': activation.std().item()
                }
                for layer_idx, activation in self.layer_activations.items()
            },
            'correlation_matrices': self.correlation_matrices,
            'recursive_patterns': self.recursive_patterns,
            'summary': self._generate_summary()
        }
        
        return report
    
    def _generate_summary(self) -> Dict:
        """Generate a summary of the analysis."""
        summary = {
            'num_layers_analyzed': len(self.layer_activations),
            'correlation_analysis': {
                'num_pairs': len(self.correlation_matrices),
                'mean_correlation': np.mean([
                    data['mean_correlation'] for data in self.correlation_matrices.values()
                ]) if self.correlation_matrices else 0.0
            },
            'recursive_analysis': {
                'num_layers': len(self.recursive_patterns),
                'layers_with_patterns': list(self.recursive_patterns.keys())
            }
        }
        
        return summary


def demonstrate_layer_correlation_analysis():
    """Demonstrate layer correlation analysis on a Mamba model."""
    print("🚀 Layer Correlation Analysis Demo")
    print("=" * 50)
    
    # Load model
    from transformers import AutoModelForCausalLM
    model_name = "state-spaces/mamba-130m-hf"
    print(f"📥 Loading model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Initialize analyzer
    analyzer = LayerCorrelationAnalyzer(model, device)
    
    # Test input
    test_text = "The recursive nature of Mamba models allows them to process sequences efficiently through state space models."
    print(f"📝 Test input: '{test_text}'")
    
    # Extract activations from multiple layers
    layer_indices = [0, 1, 2, 3]
    activations = analyzer.extract_layer_activations(layer_indices, test_text)
    
    # Compute cross-layer correlations
    print("\n🔗 Computing cross-layer correlations...")
    correlations = analyzer.compute_cross_layer_correlations()
    
    # Analyze recursive patterns for each layer
    print("\n🔄 Analyzing recursive patterns...")
    for layer_idx in layer_indices:
        if layer_idx in activations:
            patterns = analyzer.analyze_recursive_patterns(layer_idx)
            print(f"  📊 Layer {layer_idx}: analyzed recursive patterns")
    
    # Generate comprehensive report
    print("\n📋 Generating analysis report...")
    report = analyzer.generate_analysis_report()
    
    # Print key findings
    print("\n" + "="*60)
    print("📊 KEY FINDINGS")
    print("="*60)
    
    print(f"\n🔍 Layer Activations:")
    for layer_idx, data in report['layer_activations'].items():
        print(f"  Layer {layer_idx}: {data['shape']} (magnitude: {data['mean_magnitude']:.4f})")
    
    print(f"\n🔗 Cross-Layer Correlations:")
    for pair, data in report['correlation_matrices'].items():
        print(f"  {pair}: mean={data['mean_correlation']:.4f}, max={data['max_correlation']:.4f}")
    
    print(f"\n🔄 Recursive Patterns:")
    for layer_idx, patterns in report['recursive_patterns'].items():
        print(f"  Layer {layer_idx}:")
        if 'temporal_autocorrelation' in patterns:
            lags = list(patterns['temporal_autocorrelation'].keys())
            if lags:
                print(f"    Temporal autocorrelation: {len(lags)} lags analyzed")
        if 'spatial_correlation' in patterns:
            spatial = patterns['spatial_correlation']
            print(f"    Spatial correlation: {spatial['mean_spatial_correlation']:.4f}")
        if 'state_evolution' in patterns:
            evolution = patterns['state_evolution']
            print(f"    State evolution: {evolution['state_magnitude']['trend']} trend")
    
    # Save report
    with open('layer_correlation_analysis.json', 'w') as f:
        # Convert tensors to lists for JSON serialization
        def convert_tensors(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(v) for v in obj]
            else:
                return obj
        
        json.dump(convert_tensors(report), f, indent=2)
    
    print(f"\n💾 Analysis report saved to: layer_correlation_analysis.json")
    print("\n✅ Layer correlation analysis complete!")
    
    return analyzer, report


if __name__ == "__main__":
    demonstrate_layer_correlation_analysis()
