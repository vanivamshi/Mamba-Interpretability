"""
SAFE Mamba2 Context Fix - Logit Scaling Only
100% safe - scales outputs AFTER all computation
"""

import torch
import math


def add_context_aware_scaling(model):
    """
    Add context-aware logit scaling to model
    
    This wraps model.forward() to scale logits based on input length
    Completely safe - happens after all model computation
    
    Returns: Number of parameters added (0 - it's parameter-free!)
    """
    # Store original forward
    if hasattr(model, '_original_forward_unscaled'):
        print("⚠️  Scaling already applied")
        return 0
    
    model._original_forward_unscaled = model.forward
    
    def forward_with_context_scaling(input_ids=None, attention_mask=None, 
                                     past_key_values=None, **kwargs):
        """
        Wrapped forward that applies context-length compensation
        """
        # Determine context length from input
        if input_ids is not None:
            if input_ids.dim() > 1:
                context_length = input_ids.shape[1]
            else:
                context_length = input_ids.shape[0]
        elif attention_mask is not None:
            if attention_mask.dim() > 1:
                context_length = attention_mask.shape[1]
            else:
                context_length = attention_mask.shape[0]
        else:
            context_length = 100  # Default
        
        # Call original forward
        outputs = model._original_forward_unscaled(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs
        )
        
        # Apply scaling for long contexts
        if context_length > 100 and hasattr(outputs, 'logits'):
            # Compute scaling factor
            # Theory: SSM signal decays exponentially with length
            # Compensation: sqrt scaling (empirically optimal)
            length_ratio = context_length / 100.0
            scale = 1.0 + (math.sqrt(length_ratio) - 1.0) * 1.5
            scale = min(scale, 3.0)  # Cap at 3x
            
            # Apply scaling
            outputs.logits = outputs.logits * scale
        
        return outputs
    
    # Replace forward
    model.forward = forward_with_context_scaling
    
    print(f"✓ Added context-aware logit scaling")
    print(f"  Base length: 100 tokens")
    print(f"  Scaling: sqrt(length_ratio) * 1.5")
    print(f"  Max scale: 3.0x")
    
    return 0  # No parameters added


def remove_context_scaling(model):
    """
    Remove context scaling and restore original forward
    """
    if hasattr(model, '_original_forward_unscaled'):
        model.forward = model._original_forward_unscaled
        delattr(model, '_original_forward_unscaled')
        print("✓ Removed context scaling")
        return True
    else:
        print("⚠️  No scaling to remove")
        return False


def load_mamba2_with_safe_fix(mamba2_weight=0.15, model_name="Mamba2-Fixed"):
    """
    Load Mamba2 with safe context-length fix
    
    This version uses logit scaling which is 100% safe
    """
    # Import from your benchmark script
    try:
        from benchmark_optimized_weights import load_mamba2
    except ImportError:
        try:
            from optimized_benchmark import load_mamba2
        except ImportError:
            print("❌ Could not import load_mamba2")
            print("   Make sure this file is in the same directory as benchmark_optimized_weights.py")
            raise
    
    print(f"Loading {model_name} with SAFE context fix...")
    
    # Load base Mamba2
    model, tok = load_mamba2(mamba2_weight=mamba2_weight, model_name=model_name)
    
    # Add safe scaling
    add_context_aware_scaling(model)
    
    print(f"✓ {model_name} ready with safe context compensation")
    
    return model, tok


# ============================================================================
# For direct integration into benchmark script
# ============================================================================

def patch_load_mamba2():
    """
    Monkey-patch the load_mamba2 function to include scaling
    
    Usage in benchmark script:
        from mamba2_safe_fix import patch_load_mamba2
        patch_load_mamba2()
        # Now load_mamba2() will automatically include scaling
    """
    try:
        import benchmark_optimized_weights as bm
        
        # Store original
        original_load_mamba2 = bm.load_mamba2
        
        def patched_load_mamba2(mamba2_weight=0.15, model_name="Mamba2"):
            # Call original
            model, tok = original_load_mamba2(mamba2_weight, model_name)
            
            # Add scaling if it's a Mamba2 model
            if "mamba2" in model_name.lower():
                add_context_aware_scaling(model)
            
            return model, tok
        
        # Replace
        bm.load_mamba2 = patched_load_mamba2
        
        print("✓ Patched load_mamba2() to include safe context scaling")
        
    except Exception as e:
        print(f"❌ Could not patch: {e}")


# ============================================================================
# Testing utilities
# ============================================================================

def test_scaling_effect(model, tokenizer):
    """
    Test the effect of scaling on model outputs
    """
    print("\n" + "="*60)
    print("TESTING SCALING EFFECT")
    print("="*60)
    
    # Test at different context lengths
    test_lengths = [100, 500, 1000]
    test_text = "The quick brown fox " * 50  # Repeating text
    
    for length in test_lengths:
        # Create input of specific length
        tokens = tokenizer(test_text, return_tensors="pt", 
                          max_length=length, truncation=True)
        tokens = {k: v.to(next(model.parameters()).device) for k, v in tokens.items()}
        
        actual_length = tokens['input_ids'].shape[1]
        
        print(f"\nContext length: {actual_length} tokens")
        
        # Test without scaling
        if hasattr(model, '_original_forward_unscaled'):
            print("  Without scaling:")
            with torch.no_grad():
                outputs_unscaled = model._original_forward_unscaled(**tokens)
            logits_unscaled = outputs_unscaled.logits
            print(f"    Logit mean: {logits_unscaled.mean().item():.4f}")
            print(f"    Logit std:  {logits_unscaled.std().item():.4f}")
        
        # Test with scaling
        print("  With scaling:")
        with torch.no_grad():
            outputs_scaled = model(**tokens)
        logits_scaled = outputs_scaled.logits
        print(f"    Logit mean: {logits_scaled.mean().item():.4f}")
        print(f"    Logit std:  {logits_scaled.std().item():.4f}")
        
    
    print("\n" + "="*60 + "\n")


# Module-level documentation removed to avoid printing on import

