"""
Context-Aware Post-Processing Layer for Mamba2 - FIXED VERSION
Properly detects and wraps Mamba2 layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleLengthCompensator(nn.Module):
    """
    Minimal fix: Rescale by sqrt(context_length) to counter SSM decay
    """
    def __init__(self, d_model, base_length=100):
        super().__init__()
        self.d_model = d_model
        self.base_length = base_length
        
        # Learnable compensation strength
        self.compensation_strength = nn.Parameter(torch.tensor(1.5))  # Start at 1.5x
        
        # Small refinement network
        self.refine = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, x, context_length):
        """
        Simple compensation: scale by sqrt(length_ratio)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Compute scaling factor
        length_ratio = max(context_length, self.base_length) / self.base_length
        scale = torch.sqrt(torch.tensor(length_ratio, device=x.device)) * self.compensation_strength
        scale = torch.clamp(scale, 0.8, 3.0)  # Reasonable bounds
        
        # Apply scaling
        scaled = x * scale
        
        # Refine
        refined = self.refine(scaled)
        
        # Mix with original (preserve signal)
        return refined * 0.7 + x * 0.3


def attach_simple_compensator_fixed(model, device='cuda', dtype=torch.float32):
    """
    FIXED: Properly detect Mamba2-wrapped layers and attach compensators
    
    Detection strategy:
    1. Look for layers with 'mamba2_layer' attribute (from your wrapper)
    2. Look for layers with callable 'mamba2' attribute
    3. Check if layer class name contains 'Mamba2'
    """
    if not hasattr(model, 'transformer') or not hasattr(model.transformer, 'h'):
        print("⚠️  Model doesn't have transformer.h structure")
        return 0
    
    layers = model.transformer.h
    count = 0
    d_model = 768  # GPT-2 hidden dimension
    
    print(f"Scanning {len(layers)} layers for Mamba2...")
    
    for idx, layer in enumerate(layers):
        # Debug: Check what attributes this layer has
        has_mamba2 = False
        
        # Strategy 1: Check for mamba2_layer attribute (from Mamba2ActiveLayer wrapper)
        if hasattr(layer, 'mamba2_layer'):
            has_mamba2 = True
            print(f"  Layer {idx}: Found mamba2_layer attribute ✓")
        
        # Strategy 2: Check for mamba2 attribute (callable)
        elif hasattr(layer, 'mamba2') and callable(getattr(layer, 'mamba2', None)):
            has_mamba2 = True
            print(f"  Layer {idx}: Found mamba2 callable ✓")
        
        # Strategy 3: Check class name
        elif 'Mamba2' in type(layer).__name__:
            has_mamba2 = True
            print(f"  Layer {idx}: Class name contains 'Mamba2' ✓")
        
        if not has_mamba2:
            continue
        
        # Create compensator
        compensator = SimpleLengthCompensator(d_model).to(device=device, dtype=dtype)
        
        # Store original forward function
        original_forward = layer.forward
        
        # Create wrapped forward function
        def make_compensated_forward(orig_forward, comp, layer_idx):
            def compensated_forward(hidden_states, past_key_value=None, cache_position=None,
                                   attention_mask=None, head_mask=None, encoder_hidden_states=None,
                                   encoder_attention_mask=None, use_cache=False, 
                                   output_attentions=False, **kwargs):
                
                # Determine context length from input
                if hidden_states.dim() == 3:
                    batch_size, context_length, _ = hidden_states.shape
                elif hidden_states.dim() == 2:
                    context_length, _ = hidden_states.shape
                else:
                    context_length = 100  # Fallback
                
                # Call original forward (includes Mamba2 processing)
                output = orig_forward(
                    hidden_states,
                    past_key_value=past_key_value,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    **kwargs
                )
                
                # Extract hidden states
                if isinstance(output, tuple):
                    hidden = output[0]
                    rest = output[1:]
                else:
                    hidden = output
                    rest = ()
                
                # Apply compensation
                needs_unsqueeze = False
                if hidden.dim() == 2:
                    hidden = hidden.unsqueeze(0)
                    needs_unsqueeze = True
                
                # Apply compensator
                compensated = comp(hidden, context_length)
                
                if needs_unsqueeze:
                    compensated = compensated.squeeze(0)
                
                # Return in original format
                if isinstance(output, tuple):
                    return (compensated,) + rest
                else:
                    return compensated
            
            return compensated_forward
        
        # Replace forward function
        layer.forward = make_compensated_forward(original_forward, compensator, idx)
        layer.length_compensator = compensator  # Store reference for inspection
        
        count += 1
        print(f"  ✓ Attached compensator to layer {idx}")
    
    print(f"✓ Total: Attached compensators to {count} Mamba2 layers")
    return count


def load_mamba2_with_context_fix(mamba2_weight=0.15, model_name="Mamba2-Fixed"):
    """
    Load Mamba2 with automatic context-length fix
    
    This wraps your existing load_mamba2 function and adds compensation
    """
    print(f"Loading {model_name} with context-length fix (simple=True)...")
    
    # Import your existing function
    import sys
    import os
    
    # Try to import from the benchmark script
    try:
        # Try importing from benchmark_optimized_weights (actual file name)
        from benchmark_optimized_weights import load_mamba2, DEVICE
    except ImportError:
        try:
            # Fallback: try optimized_benchmark
            from optimized_benchmark import load_mamba2, DEVICE
        except ImportError:
            print("⚠️  Could not import load_mamba2 from benchmark_optimized_weights or optimized_benchmark")
            print("⚠️  Make sure to run this from the same directory")
            raise
    
    # Load base Mamba2
    print(f"Loading {model_name} (GPT-2 + Mamba2 enhancements @ {mamba2_weight:.0%} influence)...")
    model, tok = load_mamba2(mamba2_weight=mamba2_weight, model_name=model_name)
    
    # Get device and dtype from model
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    # Attach context fix with proper device/dtype
    num_fixed = attach_simple_compensator_fixed(model, device=device, dtype=dtype)
    
    print(f"✓ Context-length fix applied to {num_fixed} layers")
    
    if num_fixed == 0:
        print(f"⚠️  WARNING: No layers were fixed! Check layer structure.")
    
    return model, tok


# ============================================================================
# DEBUG: Layer Structure Inspector
# ============================================================================

def inspect_model_structure(model, max_depth=3):
    """
    Debug function: Print model structure to understand layer organization
    """
    print("\n" + "="*60)
    print("MODEL STRUCTURE INSPECTION")
    print("="*60)
    
    def print_structure(obj, name, depth=0, max_depth=3):
        if depth > max_depth:
            return
        
        indent = "  " * depth
        
        # Print object type
        print(f"{indent}{name}: {type(obj).__name__}")
        
        # If it's a module, check for interesting attributes
        if isinstance(obj, nn.Module):
            # Check for mamba2-related attributes
            mamba2_attrs = [attr for attr in dir(obj) if 'mamba2' in attr.lower()]
            if mamba2_attrs:
                print(f"{indent}  └─ Mamba2 attributes: {mamba2_attrs}")
            
            # Recurse into children
            if depth < max_depth:
                for child_name, child_module in obj.named_children():
                    print_structure(child_module, child_name, depth + 1, max_depth)
    
    print_structure(model, "model", 0, max_depth)
    print("="*60 + "\n")


# ============================================================================
# ALTERNATIVE: Direct Wrapper Modification
# ============================================================================

def inject_compensator_into_wrapper(model, device='cuda', dtype=torch.float32):
    """
    ALTERNATIVE APPROACH: Directly modify the Mamba2ActiveLayer wrapper class
    
    This finds your Mamba2ActiveLayer instances and adds compensation logic
    """
    if not hasattr(model, 'transformer') or not hasattr(model.transformer, 'h'):
        return 0
    
    layers = model.transformer.h
    count = 0
    d_model = 768
    
    for idx, layer in enumerate(layers):
        # Look for the wrapper pattern from your code
        if hasattr(layer, 'gpt2_layer') and hasattr(layer, 'mamba2_layer'):
            print(f"Found Mamba2ActiveLayer at position {idx}")
            
            # Create compensator
            compensator = SimpleLengthCompensator(d_model).to(device=device, dtype=dtype)
            
            # Store original forward
            original_forward = layer.forward
            
            # Create new forward that adds compensation
            def make_compensated_forward(orig_forward, comp, layer_idx):
                def compensated_forward(hidden_states, past_key_value=None, cache_position=None,
                                       attention_mask=None, head_mask=None, encoder_hidden_states=None,
                                       encoder_attention_mask=None, use_cache=False, 
                                       output_attentions=False, **kwargs):
                    
                    # Get context length
                    if hidden_states.dim() == 3:
                        context_length = hidden_states.shape[1]
                    else:
                        context_length = hidden_states.shape[0]
                    
                    # Original Mamba2ActiveLayer forward
                    output = orig_forward(
                        hidden_states, past_key_value, cache_position,
                        attention_mask, head_mask, encoder_hidden_states,
                        encoder_attention_mask, use_cache, output_attentions, **kwargs
                    )
                    
                    # Apply compensation
                    if isinstance(output, tuple):
                        hidden = output[0]
                        rest = output[1:]
                    else:
                        hidden = output
                        rest = ()
                    
                    # Compensate
                    if hidden.dim() == 2:
                        hidden = hidden.unsqueeze(0)
                        squeeze = True
                    else:
                        squeeze = False
                    
                    compensated = comp(hidden, context_length)
                    
                    if squeeze:
                        compensated = compensated.squeeze(0)
                    
                    if isinstance(output, tuple):
                        return (compensated,) + rest
                    else:
                        return compensated
                
                return compensated_forward
            
            # Replace forward
            layer.forward = make_compensated_forward(original_forward, compensator, idx)
            layer.compensator = compensator
            
            count += 1
            print(f"  ✓ Injected compensator into layer {idx}")
    
    print(f"✓ Injected compensators into {count} Mamba2ActiveLayer wrappers")
    return count


# ============================================================================
# BACKWARD COMPATIBILITY: Keep old function names
# ============================================================================

def attach_simple_compensator(model, device='cuda', dtype=torch.float32):
    """
    Backward compatibility wrapper for attach_simple_compensator_fixed
    """
    return attach_simple_compensator_fixed(model, device=device, dtype=dtype)


# Keep the old SimpleLengthCompensator and other classes for backward compatibility
# (They're already defined above)


# ============================================================================
# MINIMAL NON-BREAKING VERSION (SAFEST APPROACH)
# ============================================================================

class MinimalLengthCompensator(nn.Module):
    """
    Ultra-minimal compensator: Just applies a learned scaling factor
    No refinement networks that could break gradients
    """
    def __init__(self, d_model, base_length=100):
        super().__init__()
        self.d_model = d_model
        self.base_length = base_length
        
        # Single learnable parameter: compensation strength
        # Start at 1.0 (no change) and let it learn
        self.log_strength = nn.Parameter(torch.zeros(1))
    
    def get_scale(self, context_length):
        """Compute scale factor for given context length"""
        if context_length <= self.base_length:
            return 1.0
        
        # Scale factor increases with context length
        length_ratio = context_length / self.base_length
        
        # Learnable strength controls how aggressive the compensation is
        strength = torch.exp(self.log_strength)  # Always positive
        strength = torch.clamp(strength, 0.5, 2.0)  # Bounded
        
        # sqrt scaling (empirically works for SSM decay)
        scale = 1.0 + (torch.sqrt(torch.tensor(length_ratio, device=strength.device, dtype=strength.dtype)) - 1.0) * strength
        
        return torch.clamp(scale, 0.9, 3.0)
    
    def forward(self, x, context_length):
        """
        Apply minimal scaling - no other transformations
        x: [batch, seq, d_model]
        """
        scale = self.get_scale(context_length)
        
        # Simple scaling - preserves everything else
        return x * scale


def attach_minimal_compensator(model, device='cuda', dtype=torch.float32):
    """
    Minimal attachment: Only scale outputs, don't touch forward logic
    
    Key: We monkey-patch at the RETURN point, not the input
    """
    if not hasattr(model, 'transformer') or not hasattr(model.transformer, 'h'):
        print("⚠️  Model doesn't have transformer.h structure")
        return 0
    
    layers = model.transformer.h
    count = 0
    d_model = 768
    
    print(f"Attaching minimal compensators to {len(layers)} layers...")
    
    for idx, layer in enumerate(layers):
        # Only process Mamba2ActiveLayer
        if not hasattr(layer, 'mamba2_layer'):
            continue
        
        # Create minimal compensator
        compensator = MinimalLengthCompensator(d_model).to(device=device, dtype=dtype)
        
        # Store ORIGINAL forward (very important!)
        if not hasattr(layer, '_original_forward_uncompensated'):
            layer._original_forward_uncompensated = layer.forward
        
        # Create wrapper that ONLY compensates output
        def make_minimal_wrapper(original_fn, comp, layer_idx):
            def minimal_compensated_forward(hidden_states, *args, **kwargs):
                # Infer context length
                if hidden_states.dim() == 3:
                    context_length = hidden_states.shape[1]
                elif hidden_states.dim() == 2:
                    context_length = hidden_states.shape[0]
                else:
                    context_length = 100
                
                # Call ORIGINAL forward - completely unchanged
                try:
                    output = original_fn(hidden_states, *args, **kwargs)
                except Exception as e:
                    print(f"ERROR in original forward at layer {layer_idx}: {e}")
                    # Return input as-is on error
                    return hidden_states
                
                # Only compensate if context is long
                if context_length <= 100:
                    # Short context - no compensation needed
                    return output
                
                # Apply minimal compensation to output
                try:
                    # Extract hidden states
                    if isinstance(output, tuple):
                        hidden = output[0]
                        rest = output[1:]
                    else:
                        hidden = output
                        rest = ()
                    
                    # Ensure 3D for compensator
                    original_shape = hidden.shape
                    if hidden.dim() == 2:
                        hidden = hidden.unsqueeze(0)
                    
                    # Apply compensation (just scaling, nothing fancy)
                    compensated = comp(hidden, context_length)
                    
                    # Restore shape
                    if len(original_shape) == 2:
                        compensated = compensated.squeeze(0)
                    
                    # Return in same format
                    if isinstance(output, tuple):
                        return (compensated,) + rest
                    else:
                        return compensated
                
                except Exception as e:
                    # If compensation fails, return original output
                    print(f"WARNING: Compensation failed at layer {layer_idx}: {e}")
                    return output
            
            return minimal_compensated_forward
        
        # Apply wrapper
        layer.forward = make_minimal_wrapper(layer._original_forward_uncompensated, compensator, idx)
        layer.compensator = compensator
        
        count += 1
    
    print(f"✓ Attached minimal compensators to {count} layers")
    return count


def remove_compensators(model):
    """
    Remove compensators and restore original forward functions
    Useful for debugging
    """
    if not hasattr(model, 'transformer') or not hasattr(model.transformer, 'h'):
        return 0
    
    count = 0
    for layer in model.transformer.h:
        if hasattr(layer, '_original_forward_uncompensated'):
            layer.forward = layer._original_forward_uncompensated
            delattr(layer, '_original_forward_uncompensated')
            if hasattr(layer, 'compensator'):
                delattr(layer, 'compensator')
            count += 1
    
    print(f"✓ Removed compensators from {count} layers")
    return count


# ============================================================================
# ALTERNATIVE: Post-hoc compensation (safest approach)
# ============================================================================

class PostHocCompensator:
    """
    Apply compensation AFTER model forward, not during
    This is the safest approach - can't break the model
    """
    def __init__(self, base_length=100):
        self.base_length = base_length
        self.strength = 1.5  # Fixed strength
    
    def compensate(self, logits, context_length):
        """
        Compensate model outputs for long context
        
        Args:
            logits: [batch, seq, vocab] - raw model outputs
            context_length: int
        
        Returns:
            compensated_logits: [batch, seq, vocab]
        """
        if context_length <= self.base_length:
            return logits
        
        # Compute scale
        length_ratio = context_length / self.base_length
        scale = 1.0 + (math.sqrt(length_ratio) - 1.0) * self.strength
        scale = min(scale, 3.0)
        
        # Apply to logits (increases confidence)
        return logits * scale


# ============================================================================
# DEBUGGING: Test if compensation is the problem
# ============================================================================

def test_compensator_impact(model, test_input):
    """
    Test if compensator breaks the model
    
    Usage:
        model, tok = load_mamba2()
        test_input = tok("Hello world", return_tensors="pt")
        test_compensator_impact(model, test_input)
    """
    print("\n=== TESTING COMPENSATOR IMPACT ===")
    
    # Test 1: Baseline (no compensation)
    print("\n1. Testing baseline (no compensators)...")
    with torch.no_grad():
        try:
            baseline_output = model(**test_input)
            baseline_logits = baseline_output.logits
            print(f"   ✓ Baseline works")
            print(f"   Logits shape: {baseline_logits.shape}")
            print(f"   Logits mean: {baseline_logits.mean().item():.4f}")
            print(f"   Logits std: {baseline_logits.std().item():.4f}")
        except Exception as e:
            print(f"   ✗ Baseline FAILED: {e}")
            return
    
    # Test 2: With compensators
    print("\n2. Attaching compensators...")
    device = test_input['input_ids'].device if isinstance(test_input, dict) else test_input.device
    num_attached = attach_minimal_compensator(model, device=device)
    
    print("\n3. Testing with compensators...")
    with torch.no_grad():
        try:
            compensated_output = model(**test_input)
            compensated_logits = compensated_output.logits
            print(f"   ✓ Compensated works")
            print(f"   Logits shape: {compensated_logits.shape}")
            print(f"   Logits mean: {compensated_logits.mean().item():.4f}")
            print(f"   Logits std: {compensated_logits.std().item():.4f}")
            
            # Check difference
            diff = (compensated_logits - baseline_logits).abs().mean().item()
            print(f"\n   Difference from baseline: {diff:.6f}")
            
            if diff > 1.0:
                print(f"   ⚠️  Large difference - compensation may be too aggressive")
            elif diff < 0.001:
                print(f"   ⚠️  Tiny difference - compensation may not be working")
            else:
                print(f"   ✓ Reasonable difference")
                
        except Exception as e:
            print(f"   ✗ Compensated FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    # Test 3: Remove compensators
    print("\n4. Removing compensators...")
    remove_compensators(model)
    
    print("\n5. Testing after removal...")
    with torch.no_grad():
        try:
            restored_output = model(**test_input)
            restored_logits = restored_output.logits
            print(f"   ✓ Restored works")
            
            # Should match baseline
            diff = (restored_logits - baseline_logits).abs().mean().item()
            print(f"   Difference from original baseline: {diff:.6f}")
            
            if diff < 0.0001:
                print(f"   ✓ Successfully restored to baseline")
            else:
                print(f"   ⚠️  Didn't fully restore")
                
        except Exception as e:
            print(f"   ✗ Restore FAILED: {e}")
    
    print("\n=== END TEST ===\n")


# Module-level documentation removed to avoid printing on import
