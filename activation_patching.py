"""
Activation Patching Framework for Causal Testing

This module implements activation patching for necessity and sufficiency testing
of candidate circuits in Mamba models, following the mechanistic interpretability framework.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PatchResult:
    """Results from activation patching experiment."""
    original_logits: torch.Tensor
    patched_logits: torch.Tensor
    effect_size: float
    p_value: float
    patch_type: str
    circuit_indices: List[int]
    timestep: Optional[int] = None

class ActivationPatcher:
    """
    Implements activation patching for causal testing of candidate circuits.
    
    This class provides methods for:
    1. Necessity testing: Ablate/zero circuit activations
    2. Sufficiency testing: Patch circuit activations from reference to target
    """
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Store original model state for restoration
        self.original_state = None
        self.hooks = []

    def _safe_model_forward(self, inputs):
        """
        Safely call the model with either tensor or dict inputs.
        Returns model outputs (logits or full output object).
        """
        with torch.no_grad():
            try:
                logger.debug(f"🧩 Safe forward called with input type {type(inputs)}")
                # Case 1: Hugging Face-style dict input
                if isinstance(inputs, dict):
                    return self.model(**inputs)
                # Case 2: Plain tensor input
                elif isinstance(inputs, torch.Tensor):
                    return self.model(inputs)
                # Case 3: Object with input_ids
                elif hasattr(inputs, "input_ids"):
                    return self.model(input_ids=inputs.input_ids)
                else:
                    raise TypeError(f"Unsupported input type: {type(inputs)}")
            except Exception as e:
                try:
                    return self.model(input_ids=inputs if not isinstance(inputs, dict) else inputs.get("input_ids"))
                except Exception as e2:
                    logger.error(f"Safe forward failed: {e2}")
                    raise
        
    def save_model_state(self):
        """Save the current model state."""
        self.original_state = {name: param.clone() for name, param in self.model.named_parameters()}
        logger.info("Model state saved")
    
    def restore_model_state(self):
        """Restore the original model state."""
        if self.original_state is not None:
            for name, param in self.model.named_parameters():
                param.data = self.original_state[name].data
            logger.info("Model state restored")
    
    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        logger.info("All hooks cleared")
    
    def project_to_subspace(self, activations: torch.Tensor, circuit_indices: List[int]) -> torch.Tensor:
        """
        Project activations to circuit subspace.
        
        Args:
            activations: Input activations [batch_size, seq_len, hidden_size]
            circuit_indices: Indices of circuit dimensions
            
        Returns:
            Projected activations
        """
        projected = torch.zeros_like(activations)
        projected[..., circuit_indices] = activations[..., circuit_indices]
        return projected
    
    def necessity_test(self, 
                      inputs: torch.Tensor,
                      circuit_indices: List[int],
                      layer_idx: int,
                      mode: str = "zero") -> PatchResult:
        """
        Test necessity of circuit by ablating/zeroing circuit activations.
        
        Args:
            inputs: Input tokens [batch_size, seq_len]
            circuit_indices: Indices of circuit dimensions to ablate
            layer_idx: Layer index to patch
            mode: Ablation mode ("zero", "noise", "mean")
            
        Returns:
            PatchResult containing necessity test results
        """
        logger.info(f"🔍 Running necessity test: ablating {len(circuit_indices)} dimensions in layer {layer_idx}")
        logger.info(f"🔍 Input type: {type(inputs)}, shape: {inputs.shape if hasattr(inputs, 'shape') else 'no shape'}")
        
        # Get original logits
        original_outputs = self._safe_model_forward(inputs)
        original_logits = original_outputs.logits if hasattr(original_outputs, 'logits') else original_outputs
        
        # Register ablation hook
        def ablation_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Apply ablation to circuit dimensions
            ablated_hidden = hidden_states.clone()
            
            # ✅ FIX: Handle shape mismatches by ensuring indices are within bounds
            valid_indices = [idx for idx in circuit_indices if idx < ablated_hidden.shape[-1]]
            
            if mode == "zero":
                ablated_hidden[..., valid_indices] = 0
            elif mode == "noise":
                noise = torch.randn_like(ablated_hidden[..., valid_indices]) * 0.1
                ablated_hidden[..., valid_indices] = noise
            elif mode == "mean":
                mean_val = ablated_hidden[..., valid_indices].mean()
                ablated_hidden[..., valid_indices] = mean_val
            
            if isinstance(output, tuple):
                return (ablated_hidden, *output[1:])
            else:
                return ablated_hidden
        
        # Register hook on target layer
        from utils import get_model_layers
        layers = get_model_layers(self.model)
        if layers and layer_idx < len(layers):
            hook = layers[layer_idx].register_forward_hook(ablation_hook)
            self.hooks.append(hook)
        
        # Get ablated logits
        ablated_outputs = self._safe_model_forward(inputs)
        ablated_logits = ablated_outputs.logits if hasattr(ablated_outputs, 'logits') else ablated_outputs
        
        # Compute effect size
        effect_size = self._compute_effect_size(original_logits, ablated_logits)
        
        # Compute statistical significance
        p_value = self._compute_p_value(original_logits, ablated_logits)
        
        return PatchResult(
            original_logits=original_logits,
            patched_logits=ablated_logits,
            effect_size=effect_size,
            p_value=p_value,
            patch_type=f"necessity_{mode}",
            circuit_indices=circuit_indices,
            timestep=None
        )
    
    def sufficiency_test(self,
                        target_inputs: torch.Tensor,
                        reference_inputs: torch.Tensor,
                        circuit_indices: List[int],
                        layer_idx: int,
                        timestep: Optional[int] = None) -> PatchResult:
        """
        Test sufficiency by patching circuit activations from reference to target.
        
        Args:
            target_inputs: Target input tokens [batch_size, seq_len]
            reference_inputs: Reference input tokens [batch_size, seq_len]
            circuit_indices: Indices of circuit dimensions to patch
            layer_idx: Layer index to patch
            timestep: Specific timestep to patch (None for all timesteps)
            
        Returns:
            PatchResult containing sufficiency test results
        """
        logger.info(f"Running sufficiency test: patching {len(circuit_indices)} dimensions from reference to target")
        
        # Get reference activations
        reference_activations = self._extract_activations(reference_inputs, layer_idx)
        
        # Get original target logits
        original_outputs = self._safe_model_forward(target_inputs)
        original_logits = original_outputs.logits if hasattr(original_outputs, 'logits') else original_outputs
        
        # Register patching hook
        def patching_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Patch circuit dimensions
            patched_hidden = hidden_states.clone()
            
            # ✅ FIX: Handle shape mismatches between reference and target
            target_seq_len = patched_hidden.shape[1]
            ref_seq_len = reference_activations.shape[1]
            min_seq_len = min(target_seq_len, ref_seq_len)
            
            logger.debug(f"🔧 Patching shapes: target={patched_hidden.shape}, ref={reference_activations.shape}, min_seq={min_seq_len}")
            
            # ✅ FIX: Ensure indices are within bounds
            valid_indices = [idx for idx in circuit_indices if idx < patched_hidden.shape[-1]]
            logger.debug(f"🔧 Valid indices: {len(valid_indices)}/{len(circuit_indices)}")
            
            if timestep is not None:
                # Patch specific timestep
                if timestep < min_seq_len:
                    patched_hidden[:, timestep, valid_indices] = reference_activations[:, timestep, valid_indices]
            else:
                # Patch all timesteps (up to minimum sequence length)
                patched_hidden[:, :min_seq_len, valid_indices] = reference_activations[:, :min_seq_len, valid_indices]
            
            if isinstance(output, tuple):
                return (patched_hidden, *output[1:])
            else:
                return patched_hidden
        
        # Register hook on target layer
        from utils import get_model_layers
        layers = get_model_layers(self.model)
        if layers and layer_idx < len(layers):
            hook = layers[layer_idx].register_forward_hook(patching_hook)
            self.hooks.append(hook)
        
        # Get patched logits
        patched_outputs = self._safe_model_forward(target_inputs)
        patched_logits = patched_outputs.logits if hasattr(patched_outputs, 'logits') else patched_outputs
        
        # Compute effect size
        effect_size = self._compute_effect_size(original_logits, patched_logits)
        
        # Compute statistical significance
        p_value = self._compute_p_value(original_logits, patched_logits)
        
        return PatchResult(
            original_logits=original_logits,
            patched_logits=patched_logits,
            effect_size=effect_size,
            p_value=p_value,
            patch_type="sufficiency",
            circuit_indices=circuit_indices,
            timestep=timestep
        )
    
    def _extract_activations(self, inputs: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Extract activations from a specific layer."""
        activations = []
        
        def activation_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            activations.append(hidden_states.detach().clone())
        
        # Register temporary hook
        from utils import get_model_layers
        layers = get_model_layers(self.model)
        if layers and layer_idx < len(layers):
            hook = layers[layer_idx].register_forward_hook(activation_hook)
            
            _ = self._safe_model_forward(inputs)
            
            hook.remove()
            
            if activations:
                return activations[0]
        
        return torch.zeros_like(inputs)
    
    def _compute_effect_size(self, original_logits: torch.Tensor, patched_logits: torch.Tensor) -> float:
        """Compute effect size (Cohen's d) between original and patched logits."""
        # Flatten logits for comparison
        orig_flat = original_logits.view(-1)
        patched_flat = patched_logits.view(-1)
        
        # Compute Cohen's d
        pooled_std = torch.sqrt((torch.var(orig_flat) + torch.var(patched_flat)) / 2)
        if pooled_std > 0:
            effect_size = torch.abs(torch.mean(orig_flat) - torch.mean(patched_flat)) / pooled_std
            return effect_size.item()
        else:
            return 0.0
    
    def _compute_p_value(self, original_logits: torch.Tensor, patched_logits: torch.Tensor) -> float:
        """Compute p-value using paired t-test."""
        # Flatten logits for comparison
        orig_flat = original_logits.view(-1).cpu().numpy()
        patched_flat = patched_logits.view(-1).cpu().numpy()
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(orig_flat, patched_flat)
        return p_value
    
    def run_control_tests(self,
                         inputs: torch.Tensor,
                         circuit_indices: List[int],
                         layer_idx: int,
                         num_random_tests: int = 10) -> Dict[str, List[float]]:
        """
        Run control tests with random subspaces.
        
        Args:
            inputs: Input tokens
            circuit_indices: Circuit dimensions
            layer_idx: Layer index
            num_random_tests: Number of random control tests
            
        Returns:
            Dictionary containing control test results
        """
        logger.info(f"Running {num_random_tests} random control tests")
        
        control_effects = []
        
        # ✅ FIX: Handle dict inputs for control tests
        if isinstance(inputs, dict):
            # Extract input_ids from dict to get proper shape
            input_ids = inputs.get('input_ids', inputs)
            if hasattr(input_ids, 'shape'):
                hidden_size = input_ids.shape[-1] if len(input_ids.shape) > 2 else 768
            else:
                hidden_size = 768  # Default fallback
        else:
            hidden_size = inputs.shape[-1] if len(inputs.shape) > 2 else 768  # Default hidden size
        
        for i in range(num_random_tests):
            # Generate random circuit indices
            random_indices = np.random.choice(hidden_size, size=len(circuit_indices), replace=False)
            
            # Run necessity test with random indices
            result = self.necessity_test(inputs, random_indices.tolist(), layer_idx)
            control_effects.append(result.effect_size)
        
        return {
            'random_effects': control_effects,
            'mean_random_effect': np.mean(control_effects),
            'std_random_effect': np.std(control_effects),
            'max_random_effect': np.max(control_effects)
        }

class CircuitTester:
    """
    High-level interface for testing candidate circuits.
    """
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.patcher = ActivationPatcher(model, device)
        self.test_results = {}
    
    def test_circuit(self,
                    inputs: torch.Tensor,
                    circuit_indices: List[int],
                    layer_idx: int,
                    reference_inputs: Optional[torch.Tensor] = None,
                    run_controls: bool = True) -> Dict[str, Any]:
        """
        Test a candidate circuit comprehensively.
        
        Args:
            inputs: Input tokens for testing
            circuit_indices: Circuit dimension indices
            layer_idx: Layer index
            reference_inputs: Reference inputs for sufficiency testing
            run_controls: Whether to run control tests
            
        Returns:
            Dictionary containing all test results
        """
        logger.info(f"🔍 CircuitTester.test_circuit called with {len(circuit_indices)} dimensions in layer {layer_idx}")
        logger.info(f"🔍 Input type: {type(inputs)}, shape: {inputs.shape if hasattr(inputs, 'shape') else 'no shape'}")
        
        results = {
            'circuit_indices': circuit_indices,
            'layer_idx': layer_idx,
            'necessity_tests': {},
            'sufficiency_tests': {},
            'control_tests': {}
        }
        
        # Necessity tests
        for mode in ["zero", "noise", "mean"]:
            try:
                necessity_result = self.patcher.necessity_test(inputs, circuit_indices, layer_idx, mode)
                results['necessity_tests'][mode] = {
                    'effect_size': necessity_result.effect_size,
                    'p_value': necessity_result.p_value,
                    'significant': necessity_result.p_value < 0.05
                }
                self.patcher.clear_hooks()
            except Exception as e:
                logger.error(f"Necessity test failed for mode {mode}: {e}")
                results['necessity_tests'][mode] = {'error': str(e)}
        
        # Sufficiency tests
        if reference_inputs is not None:
            try:
                sufficiency_result = self.patcher.sufficiency_test(
                    inputs, reference_inputs, circuit_indices, layer_idx
                )
                results['sufficiency_tests']['full_patch'] = {
                    'effect_size': sufficiency_result.effect_size,
                    'p_value': sufficiency_result.p_value,
                    'significant': sufficiency_result.p_value < 0.05
                }
                self.patcher.clear_hooks()
            except Exception as e:
                logger.error(f"Sufficiency test failed: {e}")
                results['sufficiency_tests']['full_patch'] = {'error': str(e)}
        
        # Control tests
        if run_controls:
            try:
                control_results = self.patcher.run_control_tests(inputs, circuit_indices, layer_idx)
                results['control_tests'] = control_results
            except Exception as e:
                logger.error(f"Control tests failed: {e}")
                results['control_tests'] = {'error': str(e)}
        
        # Store results
        circuit_id = f"layer_{layer_idx}_dims_{len(circuit_indices)}"
        self.test_results[circuit_id] = results
        
        return results
    
    def visualize_results(self, circuit_id: str, save_path: Optional[str] = None):
        """Visualize circuit test results."""
        if circuit_id not in self.test_results:
            logger.error(f"Circuit {circuit_id} not found in test results")
            return
        
        results = self.test_results[circuit_id]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Circuit Test Results: {circuit_id}', fontsize=16)
        
        # Plot 1: Necessity test effects
        necessity_modes = list(results['necessity_tests'].keys())
        necessity_effects = []
        necessity_significant = []
        
        for mode in necessity_modes:
            if 'effect_size' in results['necessity_tests'][mode]:
                necessity_effects.append(results['necessity_tests'][mode]['effect_size'])
                necessity_significant.append(results['necessity_tests'][mode]['significant'])
            else:
                necessity_effects.append(0)
                necessity_significant.append(False)
        
        colors = ['red' if sig else 'blue' for sig in necessity_significant]
        axes[0, 0].bar(necessity_modes, necessity_effects, color=colors)
        axes[0, 0].set_title('Necessity Test Effects')
        axes[0, 0].set_ylabel('Effect Size')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Sufficiency test effects
        if results['sufficiency_tests']:
            sufficiency_modes = list(results['sufficiency_tests'].keys())
            sufficiency_effects = []
            sufficiency_significant = []
            
            for mode in sufficiency_modes:
                if 'effect_size' in results['sufficiency_tests'][mode]:
                    sufficiency_effects.append(results['sufficiency_tests'][mode]['effect_size'])
                    sufficiency_significant.append(results['sufficiency_tests'][mode]['significant'])
                else:
                    sufficiency_effects.append(0)
                    sufficiency_significant.append(False)
            
            colors = ['red' if sig else 'blue' for sig in sufficiency_significant]
            axes[0, 1].bar(sufficiency_modes, sufficiency_effects, color=colors)
            axes[0, 1].set_title('Sufficiency Test Effects')
            axes[0, 1].set_ylabel('Effect Size')
            axes[0, 1].tick_params(axis='x', rotation=45)
        else:
            axes[0, 1].text(0.5, 0.5, 'No Sufficiency\nTests Run', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Sufficiency Test Effects')
        
        # Plot 3: Control test distribution
        if 'random_effects' in results['control_tests']:
            random_effects = results['control_tests']['random_effects']
            axes[1, 0].hist(random_effects, bins=20, alpha=0.7, color='gray')
            axes[1, 0].axvline(np.mean(random_effects), color='red', linestyle='--', 
                             label=f'Mean: {np.mean(random_effects):.3f}')
            axes[1, 0].set_title('Random Control Effects Distribution')
            axes[1, 0].set_xlabel('Effect Size')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].legend()
        else:
            axes[1, 0].text(0.5, 0.5, 'No Control\nTests Run', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Control Test Distribution')
        
        # Plot 4: Summary comparison
        all_effects = []
        all_labels = []
        
        # Add necessity effects
        for mode, effect in zip(necessity_modes, necessity_effects):
            all_effects.append(effect)
            all_labels.append(f'Necessity_{mode}')
        
        # Add sufficiency effects
        if results['sufficiency_tests']:
            for mode, effect in zip(sufficiency_modes, sufficiency_effects):
                all_effects.append(effect)
                all_labels.append(f'Sufficiency_{mode}')
        
        # Add control mean
        if 'mean_random_effect' in results['control_tests']:
            all_effects.append(results['control_tests']['mean_random_effect'])
            all_labels.append('Control_Mean')
        
        if all_effects:
            axes[1, 1].bar(range(len(all_effects)), all_effects)
            axes[1, 1].set_title('All Effects Comparison')
            axes[1, 1].set_ylabel('Effect Size')
            axes[1, 1].set_xticks(range(len(all_labels)))
            axes[1, 1].set_xticklabels(all_labels, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Circuit test visualization saved to {save_path}")
        
        plt.show()
    
    def get_significant_circuits(self, p_threshold: float = 0.05) -> Dict[str, Dict[str, Any]]:
        """Get circuits with significant effects."""
        significant_circuits = {}
        
        for circuit_id, results in self.test_results.items():
            significant_tests = []
            
            # Check necessity tests
            for mode, test_result in results['necessity_tests'].items():
                if 'significant' in test_result and test_result['significant']:
                    significant_tests.append(f"necessity_{mode}")
            
            # Check sufficiency tests
            for mode, test_result in results['sufficiency_tests'].items():
                if 'significant' in test_result and test_result['significant']:
                    significant_tests.append(f"sufficiency_{mode}")
            
            if significant_tests:
                significant_circuits[circuit_id] = {
                    'significant_tests': significant_tests,
                    'results': results
                }
        
        return significant_circuits

def run_activation_patching_analysis(model: nn.Module,
                                    inputs: torch.Tensor,
                                    candidate_circuits: List[List[int]],
                                    layer_idx: int,
                                    reference_inputs: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    """
    Run comprehensive activation patching analysis.
    
    Args:
        model: Mamba model to test
        inputs: Input tokens for testing
        candidate_circuits: List of candidate circuit dimension lists
        layer_idx: Layer index to test
        reference_inputs: Reference inputs for sufficiency testing
        
    Returns:
        Dictionary containing analysis results
    """
    logger.info(f"Running activation patching analysis on {len(candidate_circuits)} circuits")
    
    tester = CircuitTester(model)
    analysis_results = {
        'circuit_results': {},
        'summary': {},
        'significant_circuits': {}
    }
    
    # Test each candidate circuit
    for i, circuit_indices in enumerate(candidate_circuits):
        circuit_id = f"circuit_{i}"
        logger.info(f"Testing circuit {i+1}/{len(candidate_circuits)}: {len(circuit_indices)} dimensions")
        
        try:
            results = tester.test_circuit(
                inputs=inputs,
                circuit_indices=circuit_indices,
                layer_idx=layer_idx,
                reference_inputs=reference_inputs,
                run_controls=True
            )
            analysis_results['circuit_results'][circuit_id] = results
        except Exception as e:
            logger.error(f"Failed to test circuit {i}: {e}")
            analysis_results['circuit_results'][circuit_id] = {'error': str(e)}
    
    # Get significant circuits
    significant_circuits = tester.get_significant_circuits()
    analysis_results['significant_circuits'] = significant_circuits
    
    # Compute summary statistics
    all_effects = []
    for circuit_id, results in analysis_results['circuit_results'].items():
        if 'error' not in results:
            # Collect all effect sizes
            for test_type in ['necessity_tests', 'sufficiency_tests']:
                for mode, test_result in results.get(test_type, {}).items():
                    if 'effect_size' in test_result:
                        all_effects.append(test_result['effect_size'])
    
    if all_effects:
        analysis_results['summary'] = {
            'total_circuits_tested': len(candidate_circuits),
            'significant_circuits': len(significant_circuits),
            'mean_effect_size': np.mean(all_effects),
            'max_effect_size': np.max(all_effects),
            'effect_size_std': np.std(all_effects)
        }
    
    return analysis_results

if __name__ == "__main__":
    # Example usage
    logger.info("Activation patching framework implementation complete!")
    
    print("Activation patching framework ready for causal testing!")
    print("Key features:")
    print("- Necessity testing (ablation)")
    print("- Sufficiency testing (patching)")
    print("- Control tests with random subspaces")
    print("- Statistical significance testing")
    print("- Comprehensive visualization")
