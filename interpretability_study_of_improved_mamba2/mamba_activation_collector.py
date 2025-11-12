"""
Mamba-Specific Activation Collection Utilities

This module provides specialized activation collection for Mamba models,
handling the unique aspects of SSM architecture and state management.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Tuple, Optional, Any
from transformers import MambaForCausalLM

logger = logging.getLogger(__name__)

class MambaActivationHook:
    """Specialized hook for Mamba models that handles SSM-specific outputs."""
    
    def __init__(self, layer_idx: int, hook_type: str = "forward"):
        self.layer_idx = layer_idx
        self.hook_type = hook_type
        self.activations = []
        self.hook_handle = None
        
    def register_hook(self, model, target_layer):
        """Register the hook on the target layer."""
        if self.hook_type == "forward":
            self.hook_handle = target_layer.register_forward_hook(self._forward_hook)
        elif self.hook_type == "backward":
            self.hook_handle = target_layer.register_backward_hook(self._backward_hook)
        
        logger.info(f"Registered {self.hook_type} hook on Mamba layer {self.layer_idx}")
    
    def _forward_hook(self, module, input, output):
        """Forward hook to capture Mamba activations, handling SSM-specific outputs."""
        # Mamba layers typically output a tuple: (hidden_states, [optional states])
        if isinstance(output, tuple):
            # For Mamba, we want the hidden states (first element)
            # The second element might be recurrent states or cache
            hidden_states = output[0]
            
            # Ensure we're getting the sequence-level hidden states
            if hidden_states.dim() == 3:  # [batch, seq, hidden]
                activation = hidden_states.detach().clone()
            elif hidden_states.dim() == 2:  # [seq, hidden] or [batch*seq, hidden]
                # Reshape to [batch, seq, hidden] if needed
                # This might happen if the model processes sequences differently
                activation = hidden_states.detach().clone()
            else:
                logger.warning(f"Unexpected hidden states shape: {hidden_states.shape}")
                activation = hidden_states.detach().clone()
        else:
            # Single tensor output
            activation = output.detach().clone()
        
        self.activations.append(activation)
        logger.debug(f"Captured activation for layer {self.layer_idx}: {activation.shape}")
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Backward hook to capture gradients."""
        if grad_output and grad_output[0] is not None:
            gradient = grad_output[0].detach().clone()
            self.activations.append(gradient)
    
    def remove_hook(self):
        """Remove the registered hook."""
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None
            logger.info(f"Removed hook from Mamba layer {self.layer_idx}")
    
    def get_activations(self) -> List[torch.Tensor]:
        """Get collected activations."""
        return self.activations
    
    def clear_activations(self):
        """Clear stored activations."""
        self.activations = []

class MambaActivationCollector:
    """Specialized activation collector for Mamba models."""
    
    def __init__(self, model: MambaForCausalLM, config):
        self.model = model
        self.config = config
        self.hooks = {}
        self.activation_data = {}
        
        # Verify this is a Mamba model
        if not isinstance(model, MambaForCausalLM):
            logger.warning("Model is not MambaForCausalLM, but using MambaActivationCollector")
        
    def register_hooks(self, layer_indices: List[int]):
        """Register hooks for specified Mamba layers."""
        from utils import get_model_layers
        
        layers = get_model_layers(self.model)
        if layers is None:
            logger.error("Could not find model layers")
            return False
        
        for layer_idx in layer_indices:
            if layer_idx < len(layers):
                hook = MambaActivationHook(layer_idx)
                hook.register_hook(self.model, layers[layer_idx])
                self.hooks[layer_idx] = hook
                logger.info(f"Registered Mamba hook for layer {layer_idx}")
            else:
                logger.warning(f"Layer index {layer_idx} out of range")
        
        return True
    
    def collect_activations(self, inputs: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[int, torch.Tensor]:
        """
        Collect activations for given inputs, handling Mamba-specific considerations.
        
        Args:
            inputs: Input token IDs [batch, seq_len]
            attention_mask: Optional attention mask (though Mamba handles causality differently)
            
        Returns:
            Dictionary mapping layer indices to activation tensors
        """
        # Clear previous activations
        for hook in self.hooks.values():
            hook.clear_activations()
        
        # Prepare inputs for Mamba model
        # Mamba models typically expect input_ids and optionally attention_mask
        model_inputs = {"input_ids": inputs}
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        
        # Forward pass with Mamba-specific handling
        with torch.no_grad():
            try:
                # Mamba models may have different forward signatures
                outputs = self.model(**model_inputs)
                
                # Log model output structure for debugging
                if isinstance(outputs, tuple):
                    logger.debug(f"Mamba model output tuple with {len(outputs)} elements")
                    logger.debug(f"First element shape: {outputs[0].shape if hasattr(outputs[0], 'shape') else type(outputs[0])}")
                else:
                    logger.debug(f"Mamba model output: {type(outputs)}")
                    
            except Exception as e:
                logger.error(f"Error during Mamba forward pass: {e}")
                raise
        
        # Collect activations from hooks
        activations = {}
        for layer_idx, hook in self.hooks.items():
            hook_activations = hook.get_activations()
            if hook_activations:
                # For Mamba, we expect each layer to produce one activation per forward pass
                if len(hook_activations) == 1:
                    activation = hook_activations[0]
                else:
                    # If multiple activations, concatenate them
                    activation = torch.cat(hook_activations, dim=0)
                
                # Ensure proper shape for Mamba activations
                if activation.dim() == 3:  # [batch, seq, hidden]
                    activations[layer_idx] = activation
                elif activation.dim() == 2:  # [seq, hidden] or [batch*seq, hidden]
                    # Reshape to [batch, seq, hidden] if we know the sequence length
                    batch_size = inputs.shape[0]
                    seq_len = inputs.shape[1]
                    hidden_size = activation.shape[-1]
                    
                    if activation.shape[0] == batch_size * seq_len:
                        # Reshape from [batch*seq, hidden] to [batch, seq, hidden]
                        activation = activation.view(batch_size, seq_len, hidden_size)
                    else:
                        # Assume it's [seq, hidden] and add batch dimension
                        activation = activation.unsqueeze(0)  # [1, seq, hidden]
                    
                    activations[layer_idx] = activation
                else:
                    logger.warning(f"Unexpected activation shape for layer {layer_idx}: {activation.shape}")
                    activations[layer_idx] = activation
                
                logger.info(f"Collected Mamba activations for layer {layer_idx}: {activations[layer_idx].shape}")
        
        return activations
    
    def remove_all_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks.values():
            hook.remove_hook()
        self.hooks.clear()
        logger.info("Removed all Mamba activation hooks")

def create_mamba_aware_collector(model, config):
    """
    Factory function to create the appropriate activation collector based on model type.
    
    Args:
        model: The loaded model
        config: Experiment configuration
        
    Returns:
        Appropriate activation collector (MambaActivationCollector or standard ActivationCollector)
    """
    if isinstance(model, MambaForCausalLM):
        logger.info("Creating Mamba-aware activation collector")
        return MambaActivationCollector(model, config)
    else:
        logger.info("Creating standard activation collector")
        from experimental_framework import ActivationCollector
        return ActivationCollector(model, config)

def analyze_mamba_output_structure(model: MambaForCausalLM, sample_input: torch.Tensor) -> Dict[str, Any]:
    """
    Analyze the output structure of a Mamba model to understand its behavior.
    
    Args:
        model: The Mamba model
        sample_input: Sample input tensor
        
    Returns:
        Dictionary with analysis results
    """
    logger.info("Analyzing Mamba model output structure...")
    
    with torch.no_grad():
        try:
            outputs = model(sample_input)
            
            analysis = {
                "output_type": type(outputs).__name__,
                "is_tuple": isinstance(outputs, tuple),
                "num_outputs": len(outputs) if isinstance(outputs, tuple) else 1,
                "output_shapes": [],
                "output_types": []
            }
            
            if isinstance(outputs, tuple):
                for i, output in enumerate(outputs):
                    if hasattr(output, 'shape'):
                        analysis["output_shapes"].append(output.shape)
                        analysis["output_types"].append(type(output).__name__)
                    else:
                        analysis["output_shapes"].append(str(output))
                        analysis["output_types"].append(type(output).__name__)
            else:
                if hasattr(outputs, 'shape'):
                    analysis["output_shapes"].append(outputs.shape)
                    analysis["output_types"].append(type(outputs).__name__)
                else:
                    analysis["output_shapes"].append(str(outputs))
                    analysis["output_types"].append(type(outputs).__name__)
            
            logger.info(f"Mamba output analysis: {analysis}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing Mamba output structure: {e}")
            return {"error": str(e)}
