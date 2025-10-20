"""
Activation Collection Debugging Guide

This guide provides step-by-step instructions for debugging and ensuring
correct activation collection in Mamba mechanistic analysis.
"""

import torch
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class ActivationDebugger:
    """
    Debugging utilities for activation collection issues.
    """
    
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def debug_model_layers(self) -> Dict[str, Any]:
        """
        Debug model layer access - this is the most common issue.
        """
        logger.info("🔍 Debugging model layer access...")
        
        debug_info = {
            'model_type': type(self.model).__name__,
            'model_attributes': [attr for attr in dir(self.model) if not attr.startswith('_')],
            'layer_access_methods': {}
        }
        
        # Test all possible layer access methods
        access_methods = [
            ('model.layers', lambda m: getattr(m, 'layers', None)),
            ('model.backbone.layers', lambda m: getattr(getattr(m, 'backbone', None), 'layers', None)),
            ('model.model.layers', lambda m: getattr(getattr(m, 'model', None), 'layers', None)),
            ('model.transformer.h', lambda m: getattr(getattr(m, 'transformer', None), 'h', None)),
            ('model.transformer.layers', lambda m: getattr(getattr(m, 'transformer', None), 'layers', None)),
        ]
        
        for method_name, access_fn in access_methods:
            try:
                layers = access_fn(self.model)
                if layers is not None:
                    debug_info['layer_access_methods'][method_name] = {
                        'success': True,
                        'layer_count': len(layers),
                        'layer_types': [type(layer).__name__ for layer in layers[:3]],
                        'first_layer_attributes': [attr for attr in dir(layers[0]) if not attr.startswith('_')][:10]
                    }
                    logger.info(f"✅ {method_name}: Found {len(layers)} layers")
                else:
                    debug_info['layer_access_methods'][method_name] = {'success': False, 'reason': 'None returned'}
            except Exception as e:
                debug_info['layer_access_methods'][method_name] = {'success': False, 'error': str(e)}
                logger.warning(f"❌ {method_name}: {str(e)}")
        
        return debug_info
    
    def debug_hook_registration(self, layer_idx: int = 0) -> Dict[str, Any]:
        """
        Debug hook registration for a specific layer.
        """
        logger.info(f"🔍 Debugging hook registration for layer {layer_idx}...")
        
        debug_info = {
            'layer_idx': layer_idx,
            'target_layer_found': False,
            'target_layer_type': None,
            'hook_registration_success': False,
            'activation_capture_success': False,
            'errors': []
        }
        
        try:
            # Find target layer
            target_layer = None
            if hasattr(self.model, 'layers') and layer_idx < len(self.model.layers):
                target_layer = self.model.layers[layer_idx]
                debug_info['target_layer_found'] = True
                debug_info['target_layer_type'] = type(target_layer).__name__
                logger.info(f"✅ Found layer {layer_idx} using model.layers")
            elif hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'layers'):
                if layer_idx < len(self.model.backbone.layers):
                    target_layer = self.model.backbone.layers[layer_idx]
                    debug_info['target_layer_found'] = True
                    debug_info['target_layer_type'] = type(target_layer).__name__
                    logger.info(f"✅ Found layer {layer_idx} using backbone.layers")
            
            if target_layer is None:
                debug_info['errors'].append(f"Could not find layer {layer_idx}")
                return debug_info
            
            # Test hook registration
            from experimental_framework import ActivationHook
            hook = ActivationHook(layer_idx)
            
            try:
                hook.register_hook(self.model, target_layer)
                debug_info['hook_registration_success'] = True
                logger.info(f"✅ Hook registered successfully")
                
                # Test activation capture
                test_input = torch.tensor([[1, 2, 3, 4, 5]], device=self.device)
                with torch.no_grad():
                    _ = self.model(test_input)
                
                activations = hook.get_activations()
                if activations and len(activations) > 0:
                    debug_info['activation_capture_success'] = True
                    debug_info['activation_shape'] = activations[0].shape
                    debug_info['activation_dtype'] = str(activations[0].dtype)
                    logger.info(f"✅ Activation captured: {activations[0].shape}")
                else:
                    debug_info['errors'].append("No activations captured")
                
                # Clean up
                hook.remove_hook()
                logger.info(f"✅ Hook removed successfully")
                
            except Exception as e:
                debug_info['errors'].append(f"Hook registration failed: {str(e)}")
                logger.error(f"❌ Hook registration failed: {str(e)}")
        
        except Exception as e:
            debug_info['errors'].append(f"Debug process failed: {str(e)}")
            logger.error(f"❌ Debug process failed: {str(e)}")
        
        return debug_info
    
    def debug_activation_collection(self, text: str, layer_indices: List[int]) -> Dict[str, Any]:
        """
        Debug the complete activation collection process.
        """
        logger.info(f"🔍 Debugging activation collection for text: '{text[:50]}...'")
        
        debug_info = {
            'text': text,
            'layer_indices': layer_indices,
            'tokenization_success': False,
            'collector_setup_success': False,
            'activation_collection_success': False,
            'activations': {},
            'errors': []
        }
        
        try:
            # Test tokenization
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            debug_info['tokenization_success'] = True
            debug_info['input_shape'] = inputs['input_ids'].shape
            logger.info(f"✅ Tokenization successful: {inputs['input_ids'].shape}")
            
            # Test collector setup
            from experimental_framework import ActivationCollector
            collector = ActivationCollector(self.model, None)
            
            try:
                success = collector.register_hooks(layer_indices)
                debug_info['collector_setup_success'] = success
                if success:
                    logger.info(f"✅ Collector setup successful for layers: {layer_indices}")
                else:
                    debug_info['errors'].append("Collector setup failed")
                    return debug_info
            except Exception as e:
                debug_info['errors'].append(f"Collector setup failed: {str(e)}")
                return debug_info
            
            # Test activation collection
            try:
                activations = collector.collect_activations(inputs["input_ids"])
                debug_info['activation_collection_success'] = True
                debug_info['activations'] = {
                    layer_idx: {
                        'shape': list(activation.shape),
                        'dtype': str(activation.dtype),
                        'device': str(activation.device),
                        'mean': float(torch.mean(activation)),
                        'std': float(torch.std(activation))
                    }
                    for layer_idx, activation in activations.items()
                }
                logger.info(f"✅ Activation collection successful for {len(activations)} layers")
                
            except Exception as e:
                debug_info['errors'].append(f"Activation collection failed: {str(e)}")
                logger.error(f"❌ Activation collection failed: {str(e)}")
            
            # Clean up
            collector.remove_all_hooks()
            logger.info(f"✅ Hooks removed")
            
        except Exception as e:
            debug_info['errors'].append(f"Debug process failed: {str(e)}")
            logger.error(f"❌ Debug process failed: {str(e)}")
        
        return debug_info

def create_improved_activation_collector():
    """
    Create an improved ActivationCollector with better error handling.
    """
    from experimental_framework import ActivationCollector
    
    class ImprovedActivationCollector(ActivationCollector):
        """
        Improved ActivationCollector with better error handling and debugging.
        """
        
        def register_hooks(self, layer_indices: List[int]) -> bool:
            """
            Register hooks with improved error handling and debugging.
            """
            logger.info(f"Registering hooks for layers: {layer_indices}")
            
            # Clear existing hooks
            self.remove_all_hooks()
            
            # Find layers using multiple strategies
            layers = self._find_model_layers()
            if layers is None:
                logger.error("Could not find model layers using any method")
                return False
            
            logger.info(f"Found {len(layers)} layers in model")
            
            success_count = 0
            for layer_idx in layer_indices:
                if layer_idx < len(layers):
                    try:
                        from experimental_framework import ActivationHook
                        hook = ActivationHook(layer_idx)
                        hook.register_hook(self.model, layers[layer_idx])
                        self.hooks[layer_idx] = hook
                        success_count += 1
                        logger.info(f"✅ Registered hook for layer {layer_idx}")
                    except Exception as e:
                        logger.error(f"❌ Failed to register hook for layer {layer_idx}: {str(e)}")
                else:
                    logger.warning(f"⚠️ Layer index {layer_idx} out of range (max: {len(layers)-1})")
            
            logger.info(f"Successfully registered {success_count}/{len(layer_indices)} hooks")
            return success_count > 0
        
        def _find_model_layers(self):
            """
            Find model layers using multiple strategies with detailed logging.
            """
            strategies = [
                ('model.layers', lambda m: getattr(m, 'layers', None)),
                ('model.backbone.layers', lambda m: getattr(getattr(m, 'backbone', None), 'layers', None)),
                ('model.model.layers', lambda m: getattr(getattr(m, 'model', None), 'layers', None)),
                ('model.transformer.h', lambda m: getattr(getattr(m, 'transformer', None), 'h', None)),
            ]
            
            for strategy_name, strategy_fn in strategies:
                try:
                    layers = strategy_fn(self.model)
                    if layers is not None and len(layers) > 0:
                        logger.info(f"✅ Found layers using {strategy_name}: {len(layers)} layers")
                        return layers
                except Exception as e:
                    logger.debug(f"Strategy {strategy_name} failed: {str(e)}")
            
            logger.error("❌ Could not find model layers using any strategy")
            return None
        
        def collect_activations(self, inputs: torch.Tensor) -> Dict[int, torch.Tensor]:
            """
            Collect activations with improved error handling.
            """
            logger.info(f"Collecting activations for input shape: {inputs.shape}")
            
            # Clear previous activations
            for hook in self.hooks.values():
                hook.clear_activations()
            
            # Forward pass with error handling
            try:
                with torch.no_grad():
                    _ = self.model(inputs)
                logger.info("✅ Forward pass completed successfully")
            except Exception as e:
                logger.error(f"❌ Forward pass failed: {str(e)}")
                return {}
            
            # Collect activations
            activations = {}
            for layer_idx, hook in self.hooks.items():
                try:
                    hook_activations = hook.get_activations()
                    if hook_activations:
                        # Concatenate all activations from this layer
                        activations[layer_idx] = torch.cat(hook_activations, dim=0)
                        logger.info(f"✅ Collected activations for layer {layer_idx}: {activations[layer_idx].shape}")
                    else:
                        logger.warning(f"⚠️ No activations collected for layer {layer_idx}")
                except Exception as e:
                    logger.error(f"❌ Failed to collect activations for layer {layer_idx}: {str(e)}")
            
            return activations
    
    return ImprovedActivationCollector

def run_activation_debugging_suite(model, tokenizer, device="cuda"):
    """
    Run a comprehensive debugging suite for activation collection.
    """
    logger.info("🚀 Starting comprehensive activation debugging suite...")
    
    debugger = ActivationDebugger(model, tokenizer, device)
    
    # Step 1: Debug model structure
    logger.info("Step 1: Debugging model structure...")
    model_debug = debugger.debug_model_layers()
    
    # Step 2: Debug hook registration
    logger.info("Step 2: Debugging hook registration...")
    hook_debug = debugger.debug_hook_registration(0)
    
    # Step 3: Debug activation collection
    logger.info("Step 3: Debugging activation collection...")
    test_text = "The quick brown fox jumps over the lazy dog."
    collection_debug = debugger.debug_activation_collection(test_text, [0, 6])
    
    # Step 4: Test improved collector
    logger.info("Step 4: Testing improved activation collector...")
    ImprovedCollector = create_improved_activation_collector()
    improved_collector = ImprovedCollector(model, None)
    
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming industries worldwide."
    ]
    
    improved_results = {}
    for i, text in enumerate(test_texts):
        logger.info(f"Testing improved collector with text {i+1}: '{text[:30]}...'")
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        success = improved_collector.register_hooks([0, 6])
        if success:
            activations = improved_collector.collect_activations(inputs["input_ids"])
            improved_results[f'text_{i}'] = {
                'success': True,
                'layers_collected': list(activations.keys()),
                'shapes': {k: list(v.shape) for k, v in activations.items()}
            }
        else:
            improved_results[f'text_{i}'] = {'success': False}
        
        improved_collector.remove_all_hooks()
    
    # Generate summary report
    summary = {
        'model_structure_debug': model_debug,
        'hook_registration_debug': hook_debug,
        'activation_collection_debug': collection_debug,
        'improved_collector_test': improved_results,
        'overall_status': 'PASS' if all([
            model_debug['layer_access_methods'],
            hook_debug['target_layer_found'],
            collection_debug['activation_collection_success']
        ]) else 'FAIL'
    }
    
    logger.info(f"🎯 Debugging suite complete! Overall status: {summary['overall_status']}")
    return summary

if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    model_name = "state-spaces/mamba-130m-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Run debugging suite
    debug_results = run_activation_debugging_suite(model, tokenizer)
    
    print("🔍 Debugging Results:")
    print(f"Model Structure: {'✅' if debug_results['model_structure_debug']['layer_access_methods'] else '❌'}")
    print(f"Hook Registration: {'✅' if debug_results['hook_registration_debug']['target_layer_found'] else '❌'}")
    print(f"Activation Collection: {'✅' if debug_results['activation_collection_debug']['activation_collection_success'] else '❌'}")
    print(f"Overall Status: {debug_results['overall_status']}")

