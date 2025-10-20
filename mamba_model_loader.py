"""
Mamba Model Loading Utilities

This module provides utilities for properly loading Mamba models with the correct
architecture initialization for mechanistic interpretability analysis.
"""

import torch
import logging
from typing import Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, MambaForCausalLM

logger = logging.getLogger(__name__)

def load_mamba_model_and_tokenizer(
    model_name: str, 
    device: str = "cuda",
    use_mamba_class: bool = True,
    fallback_to_auto: bool = True
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load Mamba model and tokenizer with proper architecture initialization.
    
    Args:
        model_name: Name of the model to load (e.g., "state-spaces/mamba-130m-hf")
        device: Device to load the model on
        use_mamba_class: Whether to try MambaForCausalLM first
        fallback_to_auto: Whether to fallback to AutoModelForCausalLM if MambaForCausalLM fails
        
    Returns:
        Tuple of (model, tokenizer)
        
    Raises:
        Exception: If model loading fails completely
    """
    logger.info(f"Loading Mamba model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with proper Mamba class and memory optimization
    model = None
    
    # Memory-efficient loading configuration
    load_config = {
        'torch_dtype': torch.float16 if device == "cuda" else torch.float32,  # Use half precision on CUDA
        'low_cpu_mem_usage': True,  # Reduce CPU memory usage during loading
        'device_map': None,  # We'll handle device placement manually
    }
    
    if use_mamba_class:
        try:
            model = MambaForCausalLM.from_pretrained(model_name, **load_config)
            logger.info("✅ Successfully loaded model using MambaForCausalLM with memory optimization")
        except Exception as e:
            logger.warning(f"Failed to load with MambaForCausalLM: {e}")
            if fallback_to_auto:
                logger.info("Falling back to AutoModelForCausalLM...")
                model = AutoModelForCausalLM.from_pretrained(model_name, **load_config)
                logger.info("✅ Successfully loaded model using AutoModelForCausalLM with memory optimization")
            else:
                raise e
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_config)
        logger.info("✅ Successfully loaded model using AutoModelForCausalLM with memory optimization")
    
    if model is None:
        raise Exception("Failed to load model with any method")
    
    # Memory optimization and device handling
    try:
        # Clear CUDA cache before loading
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 Cleared CUDA cache")
        
        # Try to move to device with memory optimization
        if device == "cuda" and torch.cuda.is_available():
            # Check available memory
            available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            logger.info(f"💾 Available CUDA memory: {available_memory / 1024**3:.2f} GB")
            
            # Estimate model memory usage (rough approximation)
            model_params = sum(p.numel() for p in model.parameters())
            estimated_memory = model_params * 4 / 1024**3  # 4 bytes per float32 parameter
            logger.info(f"📊 Estimated model memory: {estimated_memory:.2f} GB")
            
            if estimated_memory > available_memory * 0.8:  # Use 80% of available memory as threshold
                logger.warning("⚠️  Model too large for available CUDA memory, falling back to CPU")
                device = "cpu"
        
        # Move to device
        model = model.to(device)
        model.eval()
        
        # Additional memory optimization for CUDA
        if device == "cuda":
            # Enable memory efficient attention if available
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                logger.info("✅ Enabled gradient checkpointing for memory efficiency")
            
            # Set memory fraction to prevent OOM
            torch.cuda.set_per_process_memory_fraction(0.9)
            logger.info("✅ Set CUDA memory fraction to 90%")
        
        logger.info(f"✅ Model loaded successfully on {device}")
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"❌ CUDA out of memory: {e}")
        logger.info("🔄 Falling back to CPU...")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Move to CPU
        device = "cpu"
        model = model.to(device)
        model.eval()
        logger.info("✅ Model loaded successfully on CPU (fallback)")
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        raise
    
    return model, tokenizer

def get_model_info(model: torch.nn.Module) -> dict:
    """
    Get information about the loaded model.
    
    Args:
        model: The loaded model
        
    Returns:
        Dictionary with model information
    """
    info = {
        "model_type": type(model).__name__,
        "is_mamba_class": isinstance(model, MambaForCausalLM),
        "device": next(model.parameters()).device,
        "dtype": next(model.parameters()).dtype,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    # Try to get layer information
    try:
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'layers'):
            info["num_layers"] = len(model.backbone.layers)
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            info["num_layers"] = len(model.transformer.h)
        else:
            info["num_layers"] = "unknown"
    except Exception:
        info["num_layers"] = "unknown"
    
    return info

def verify_mamba_architecture(model: torch.nn.Module) -> bool:
    """
    Verify that the model has proper Mamba architecture components.
    
    Args:
        model: The loaded model
        
    Returns:
        True if model appears to have Mamba architecture, False otherwise
    """
    try:
        # Check for Mamba-specific components
        has_backbone = hasattr(model, 'backbone')
        has_layers = hasattr(model, 'backbone') and hasattr(model.backbone, 'layers')
        
        if has_layers:
            # Check if layers have Mamba-specific components
            first_layer = model.backbone.layers[0]
            has_mamba_block = hasattr(first_layer, 'mixer') or hasattr(first_layer, 'ssm')
            return has_mamba_block
        
        return False
    except Exception as e:
        logger.warning(f"Could not verify Mamba architecture: {e}")
        return False
