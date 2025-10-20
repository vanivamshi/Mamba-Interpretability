"""
Activation Collection Validation Framework

This module provides comprehensive validation tools to ensure activation collection
is working correctly in Mamba mechanistic analysis.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

class ActivationValidator:
    """
    Comprehensive validation framework for activation collection.
    
    This class provides multiple validation methods to ensure:
    1. Correct hook registration and removal
    2. Proper activation shapes and values
    3. Consistent activation collection across runs
    4. Model layer access reliability
    5. Memory and performance considerations
    """
    
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.validation_results = {}
        
    def validate_model_structure(self) -> Dict[str, Any]:
        """
        Validate that we can access model layers correctly.
        
        Returns:
            Dictionary with layer access validation results
        """
        logger.info("Validating model structure and layer access...")
        
        results = {
            'model_type': type(self.model).__name__,
            'has_layers': False,
            'has_backbone_layers': False,
            'layer_count': 0,
            'layer_types': [],
            'access_methods': []
        }
        
        # Test different layer access methods
        access_methods = [
            ('model.layers', lambda m: m.layers),
            ('model.backbone.layers', lambda m: m.backbone.layers),
            ('model.model.layers', lambda m: m.model.layers),
            ('model.transformer.h', lambda m: m.transformer.h),
        ]
        
        for method_name, access_fn in access_methods:
            try:
                layers = access_fn(self.model)
                if layers is not None and len(layers) > 0:
                    results['access_methods'].append({
                        'method': method_name,
                        'success': True,
                        'layer_count': len(layers),
                        'layer_types': [type(layer).__name__ for layer in layers[:3]]  # First 3 layers
                    })
                    
                    if method_name == 'model.layers':
                        results['has_layers'] = True
                        results['layer_count'] = len(layers)
                        results['layer_types'] = [type(layer).__name__ for layer in layers]
                    elif method_name == 'model.backbone.layers':
                        results['has_backbone_layers'] = True
                        
            except AttributeError:
                results['access_methods'].append({
                    'method': method_name,
                    'success': False,
                    'error': 'AttributeError'
                })
        
        # Test layer access for specific indices
        if results['has_layers']:
            test_indices = [0, min(5, results['layer_count']-1), results['layer_count']-1]
            for idx in test_indices:
                try:
                    layer = self.model.layers[idx]
                    results[f'layer_{idx}_accessible'] = True
                    results[f'layer_{idx}_type'] = type(layer).__name__
                except (IndexError, AttributeError) as e:
                    results[f'layer_{idx}_accessible'] = False
                    results[f'layer_{idx}_error'] = str(e)
        
        self.validation_results['model_structure'] = results
        logger.info(f"Model structure validation complete: {len(results['access_methods'])} methods tested")
        return results
    
    def validate_hook_registration(self, layer_indices: List[int]) -> Dict[str, Any]:
        """
        Validate that hooks can be registered and removed correctly.
        
        Args:
            layer_indices: List of layer indices to test
            
        Returns:
            Dictionary with hook registration validation results
        """
        logger.info(f"Validating hook registration for layers: {layer_indices}")
        
        results = {
            'layers_tested': layer_indices,
            'registration_success': {},
            'hook_removal_success': {},
            'activation_capture_success': {}
        }
        
        from experimental_framework import ActivationHook
        
        for layer_idx in layer_indices:
            layer_results = {
                'registration': False,
                'removal': False,
                'activation_capture': False,
                'error': None
            }
            
            try:
                # Test hook registration
                hook = ActivationHook(layer_idx)
                
                # Try to find the target layer
                target_layer = None
                if hasattr(self.model, 'layers') and layer_idx < len(self.model.layers):
                    target_layer = self.model.layers[layer_idx]
                elif hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'layers'):
                    if layer_idx < len(self.model.backbone.layers):
                        target_layer = self.model.backbone.layers[layer_idx]
                
                if target_layer is None:
                    layer_results['error'] = f"Could not find layer {layer_idx}"
                    results['registration_success'][layer_idx] = layer_results
                    continue
                
                # Register hook
                hook.register_hook(self.model, target_layer)
                layer_results['registration'] = True
                
                # Test activation capture with a simple forward pass
                test_input = torch.tensor([[1, 2, 3]], device=self.device)  # Simple test input
                try:
                    with torch.no_grad():
                        _ = self.model(test_input)
                    
                    activations = hook.get_activations()
                    if activations and len(activations) > 0:
                        layer_results['activation_capture'] = True
                        layer_results['activation_shape'] = activations[0].shape
                        layer_results['activation_dtype'] = str(activations[0].dtype)
                except Exception as e:
                    layer_results['error'] = f"Activation capture failed: {str(e)}"
                
                # Test hook removal
                hook.remove_hook()
                layer_results['removal'] = True
                
            except Exception as e:
                layer_results['error'] = str(e)
            
            results['registration_success'][layer_idx] = layer_results
        
        self.validation_results['hook_registration'] = results
        logger.info(f"Hook registration validation complete for {len(layer_indices)} layers")
        return results
    
    def validate_activation_consistency(self, texts: List[str], layer_indices: List[int], 
                                      num_runs: int = 3) -> Dict[str, Any]:
        """
        Validate that activation collection is consistent across multiple runs.
        
        Args:
            texts: List of input texts
            layer_indices: Layers to test
            num_runs: Number of runs to test consistency
            
        Returns:
            Dictionary with consistency validation results
        """
        logger.info(f"Validating activation consistency across {num_runs} runs...")
        
        results = {
            'texts_tested': len(texts),
            'layers_tested': layer_indices,
            'num_runs': num_runs,
            'consistency_results': {}
        }
        
        from experimental_framework import ActivationCollector
        
        all_activations = []
        
        for run_idx in range(num_runs):
            logger.info(f"Run {run_idx + 1}/{num_runs}")
            
            # Setup collector
            collector = ActivationCollector(self.model, None)  # No config needed for validation
            collector.register_hooks(layer_indices)
            
            run_activations = {}
            
            for text in texts:
                # Tokenize
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Collect activations
                activations = collector.collect_activations(inputs["input_ids"])
                run_activations[text] = activations
            
            collector.remove_all_hooks()
            all_activations.append(run_activations)
        
        # Analyze consistency
        for layer_idx in layer_indices:
            layer_consistency = {
                'shape_consistency': True,
                'value_consistency': True,
                'mean_differences': [],
                'std_differences': [],
                'max_differences': []
            }
            
            # Check shape consistency
            shapes = []
            for run_activations in all_activations:
                for text, activations in run_activations.items():
                    if layer_idx in activations:
                        shapes.append(activations[layer_idx].shape)
            
            if len(set(str(s) for s in shapes)) > 1:
                layer_consistency['shape_consistency'] = False
                layer_consistency['shape_variations'] = list(set(str(s) for s in shapes))
            
            # Check value consistency for each text
            for text in texts:
                text_activations = []
                for run_activations in all_activations:
                    if text in run_activations and layer_idx in run_activations[text]:
                        text_activations.append(run_activations[text][layer_idx])
                
                if len(text_activations) >= 2:
                    # Compare activations pairwise
                    for i in range(len(text_activations)):
                        for j in range(i + 1, len(text_activations)):
                            act1, act2 = text_activations[i], text_activations[j]
                            
                            # Ensure same shape for comparison
                            if act1.shape == act2.shape:
                                diff = torch.abs(act1 - act2)
                                layer_consistency['mean_differences'].append(float(torch.mean(diff)))
                                layer_consistency['std_differences'].append(float(torch.std(diff)))
                                layer_consistency['max_differences'].append(float(torch.max(diff)))
                                
                                # Check if differences are significant
                                if torch.max(diff) > 1e-6:
                                    layer_consistency['value_consistency'] = False
            
            results['consistency_results'][layer_idx] = layer_consistency
        
        self.validation_results['activation_consistency'] = results
        logger.info("Activation consistency validation complete")
        return results
    
    def validate_activation_properties(self, activations: Dict[int, torch.Tensor]) -> Dict[str, Any]:
        """
        Validate that collected activations have expected properties.
        
        Args:
            activations: Dictionary of collected activations
            
        Returns:
            Dictionary with activation property validation results
        """
        logger.info("Validating activation properties...")
        
        results = {
            'layers_analyzed': list(activations.keys()),
            'property_checks': {}
        }
        
        for layer_idx, activation_tensor in activations.items():
            layer_properties = {
                'shape': list(activation_tensor.shape),
                'dtype': str(activation_tensor.dtype),
                'device': str(activation_tensor.device),
                'has_nan': bool(torch.isnan(activation_tensor).any()),
                'has_inf': bool(torch.isinf(activation_tensor).any()),
                'is_finite': bool(torch.isfinite(activation_tensor).all()),
                'memory_usage_mb': activation_tensor.element_size() * activation_tensor.nelement() / (1024**2)
            }
            
            # Statistical properties
            activation_np = activation_tensor.cpu().numpy()
            layer_properties.update({
                'mean': float(np.mean(activation_np)),
                'std': float(np.std(activation_np)),
                'min': float(np.min(activation_np)),
                'max': float(np.max(activation_np)),
                'sparsity_rate': float(np.mean(np.abs(activation_np) < 1e-6)),
                'variance': float(np.var(activation_np)),
                'kurtosis': float(scipy_stats.kurtosis(activation_np.flatten())),
                'skewness': float(scipy_stats.skew(activation_np.flatten()))
            })
            
            # Validation flags
            layer_properties['validation_flags'] = {
                'reasonable_shape': len(activation_tensor.shape) >= 2,
                'no_nan_inf': not layer_properties['has_nan'] and not layer_properties['has_inf'],
                'reasonable_range': -10 < layer_properties['min'] < 10 and -10 < layer_properties['max'] < 10,
                'reasonable_sparsity': 0 < layer_properties['sparsity_rate'] < 0.9,
                'finite_memory': layer_properties['memory_usage_mb'] < 1000  # Less than 1GB
            }
            
            results['property_checks'][layer_idx] = layer_properties
        
        self.validation_results['activation_properties'] = results
        logger.info(f"Activation properties validation complete for {len(activations)} layers")
        return results
    
    def generate_validation_report(self, save_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive validation report.
        
        Args:
            save_path: Optional path to save the report
            
        Returns:
            Complete validation report
        """
        logger.info("Generating comprehensive validation report...")
        
        report = {
            'timestamp': torch.datetime.now().isoformat(),
            'model_name': type(self.model).__name__,
            'device': str(self.device),
            'validation_results': self.validation_results
        }
        
        # Summary statistics
        summary = {
            'total_validations': len(self.validation_results),
            'model_structure_valid': False,
            'hook_registration_valid': False,
            'activation_consistency_valid': False,
            'activation_properties_valid': False
        }
        
        if 'model_structure' in self.validation_results:
            ms_results = self.validation_results['model_structure']
            summary['model_structure_valid'] = (
                ms_results['has_layers'] or ms_results['has_backbone_layers']
            )
        
        if 'hook_registration' in self.validation_results:
            hr_results = self.validation_results['hook_registration']
            summary['hook_registration_valid'] = all(
                result['registration'] and result['removal'] 
                for result in hr_results['registration_success'].values()
            )
        
        if 'activation_consistency' in self.validation_results:
            ac_results = self.validation_results['activation_consistency']
            summary['activation_consistency_valid'] = all(
                result['shape_consistency'] and result['value_consistency']
                for result in ac_results['consistency_results'].values()
            )
        
        if 'activation_properties' in self.validation_results:
            ap_results = self.validation_results['activation_properties']
            summary['activation_properties_valid'] = all(
                all(flags.values()) for flags in [
                    props['validation_flags'] 
                    for props in ap_results['property_checks'].values()
                ]
            )
        
        report['summary'] = summary
        
        # Save report if path provided
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Validation report saved to {save_path}")
        
        return report
    
    def visualize_activation_distributions(self, activations: Dict[int, torch.Tensor], 
                                        save_path: Optional[Path] = None):
        """
        Create visualizations of activation distributions.
        
        Args:
            activations: Dictionary of collected activations
            save_path: Optional path to save plots
        """
        logger.info("Creating activation distribution visualizations...")
        
        num_layers = len(activations)
        fig, axes = plt.subplots(2, num_layers, figsize=(4*num_layers, 8))
        if num_layers == 1:
            axes = axes.reshape(2, 1)
        
        for i, (layer_idx, activation_tensor) in enumerate(activations.items()):
            activation_np = activation_tensor.cpu().numpy().flatten()
            
            # Histogram
            axes[0, i].hist(activation_np, bins=50, alpha=0.7, density=True)
            axes[0, i].set_title(f'Layer {layer_idx} Distribution')
            axes[0, i].set_xlabel('Activation Value')
            axes[0, i].set_ylabel('Density')
            
            # Box plot
            axes[1, i].boxplot(activation_np)
            axes[1, i].set_title(f'Layer {layer_idx} Box Plot')
            axes[1, i].set_ylabel('Activation Value')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Activation distributions saved to {save_path}")
        
        plt.show()

def run_comprehensive_validation(model, tokenizer, texts: List[str], 
                               layer_indices: List[int] = None,
                               device: str = "cuda") -> Dict[str, Any]:
    """
    Run comprehensive activation collection validation.
    
    Args:
        model: The model to validate
        tokenizer: The tokenizer
        texts: List of test texts
        layer_indices: Layers to test (default: [0, 6, 12, 18])
        device: Device to use
        
    Returns:
        Complete validation report
    """
    if layer_indices is None:
        layer_indices = [0, 6, 12, 18]
    
    logger.info("Starting comprehensive activation collection validation...")
    
    validator = ActivationValidator(model, tokenizer, device)
    
    # Step 1: Validate model structure
    validator.validate_model_structure()
    
    # Step 2: Validate hook registration
    validator.validate_hook_registration(layer_indices)
    
    # Step 3: Validate activation consistency
    validator.validate_activation_consistency(texts, layer_indices)
    
    # Step 4: Collect activations and validate properties
    from experimental_framework import ActivationCollector
    collector = ActivationCollector(model, None)
    collector.register_hooks(layer_indices)
    
    all_activations = {}
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        activations = collector.collect_activations(inputs["input_ids"])
        
        for layer_idx, activation in activations.items():
            if layer_idx not in all_activations:
                all_activations[layer_idx] = []
            all_activations[layer_idx].append(activation)
    
    collector.remove_all_hooks()
    
    # Concatenate activations
    final_activations = {}
    for layer_idx, activation_list in all_activations.items():
        if activation_list:
            final_activations[layer_idx] = torch.cat(activation_list, dim=0)
    
    validator.validate_activation_properties(final_activations)
    
    # Generate report
    report = validator.generate_validation_report()
    
    # Create visualizations
    validator.visualize_activation_distributions(final_activations)
    
    logger.info("Comprehensive validation complete!")
    return report

if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    model_name = "state-spaces/mamba-130m-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming industries worldwide.",
        "Machine learning models require large amounts of training data."
    ]
    
    report = run_comprehensive_validation(model, tokenizer, test_texts)
    print("Validation complete! Check the report for details.")

