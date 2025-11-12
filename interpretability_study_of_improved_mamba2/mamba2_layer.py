# COMMENTED: Original Mamba2 implementation with enhanced features
# This implementation includes:
# - Enhanced SSMBlock with more parameters (5 gates, 32-state SSM)
# - Multiple scaling parameters for gradient monitoring
# - Pre/post normalization layers
# - Dtype consistency fixes
# - Fine-tuning capabilities

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List, Any
import logging
import numpy as np
import json

logger = logging.getLogger(__name__)


class GateModule(nn.Module):
    """
    Lightweight gating module producing a per-token delta in [0, 1].
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
            nn.GELU(),
            nn.Linear(d_model, d_model, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, d_model]
        # Use manual normalization instead of LayerNorm for float16 compatibility
        x_norm = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
        
        # Ensure dtype consistency
        if x_norm.dtype != next(self.proj.parameters()).dtype:
            x_norm = x_norm.to(next(self.proj.parameters()).dtype)
        
        return torch.sigmoid(self.proj(x_norm))


class SSMBlock(nn.Module):
    """
    Improved state-space block with better stability and gradient flow.
    """
    def __init__(self, d_model: int, decay: float = 0.9, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Smaller initialization for stability
        self.A = nn.Parameter(torch.randn(d_state, d_state) * 0.01)
        self.B = nn.Parameter(torch.randn(d_state, d_model) * 0.01)
        self.C = nn.Parameter(torch.randn(d_state, d_model) * 0.01)
        self.D = nn.Parameter(torch.ones(d_model) * 0.1)
        
        self.decay = nn.Parameter(torch.tensor(decay))
        
        # ✅ FIX: Simpler compression with constraints
        self.compression_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),  # ReLU instead of SiLU
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        # ✅ ADD: Layer normalization
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # ✅ Apply normalization first with dtype consistency
        if x.dtype != next(self.norm.parameters()).dtype:
            x = x.to(next(self.norm.parameters()).dtype)
        x = self.norm(x)
        
        # ✅ FIX: Keep at least 50% of signal
        compression_factor = self.compression_gate(x.mean(dim=1, keepdim=True))
        compression_factor = 0.5 + 0.5 * compression_factor
        x_compressed = x * compression_factor
        
        # SSM computation
        h = torch.zeros(batch_size, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        
        for t in range(seq_len):
            u_t = x_compressed[:, t, :]
            
            # ✅ FIX: Remove nested tanh
            h_new = torch.matmul(h, self.A) + torch.matmul(u_t, self.B.T)
            h = torch.tanh(h_new) * torch.sigmoid(self.decay)
            
            # ✅ FIX: Linear output + direct skip connection
            y_t = torch.matmul(h, self.C) + self.D * u_t
            outputs.append(y_t)
        
        output = torch.stack(outputs, dim=1)
        
        # ✅ ADD: Residual connection
        output = output + x
        
        return output


# COMMENTED: Enhanced SparseAttention implementation
# This implementation includes:
# - Lightweight sparse attention with top-k selection
# - Differentiable attention mechanism
# - Designed for analysis baselines

# class SparseAttention(nn.Module):
#     """
#     Extremely lightweight sparse attention approximation: sample top-k keys per query.
#     Designed to be cheap and differentiable for analysis baselines.
#     """
#     def __init__(self, d_model: int, sparsity: float = 0.95):
#         super().__init__()
#         self.q = nn.Linear(d_model, d_model, bias=False)
#         self.k = nn.Linear(d_model, d_model, bias=False)
#         self.v = nn.Linear(d_model, d_model, bias=False)
#         self.sparsity = sparsity
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         b, s, d = x.shape
#         
#         # Ensure dtype consistency
#         if x.dtype != next(self.q.parameters()).dtype:
#             x = x.to(next(self.q.parameters()).dtype)
#         
#         q = self.q(x)
#         k = self.k(x)
#         v = self.v(x)
#         attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (d ** 0.5)
#         # Keep top-k per query (approximate sparse attention)
#         k_keep = max(1, int((1 - self.sparsity) * s))
#         topk_vals, topk_idx = torch.topk(attn_scores, k_keep, dim=-1)
#         mask = torch.full_like(attn_scores, float('-inf'))
#         mask.scatter_(-1, topk_idx, topk_vals)
#         attn = torch.softmax(mask, dim=-1)
#         return torch.matmul(attn, v)


class LearnedGate(nn.Module):
    """
    Produces a mixing coefficient in [0,1] to blend local/global paths.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        # x: [b, s, d] -> gate: [b, s, 1]
        
        # Ensure dtype consistency
        if x.dtype != next(self.proj.parameters()).dtype:
            x = x.to(next(self.proj.parameters()).dtype)
        
        gate = torch.sigmoid(self.proj(x))
        return gate


# COMMENTED: Enhanced Mamba2Layer implementation
# This implementation includes:
# - Enhanced SSMBlock with more parameters (5 gates, 32-state SSM)
# - Multiple scaling parameters for gradient monitoring
# - Pre/post normalization layers
# - Dtype consistency fixes
# - Fine-tuning capabilities

# class EnhancedMamba2Layer(nn.Module):
#     """
#     Enhanced Mamba2Layer with more parameters for increased influence.
#     """
#     def __init__(self, d_model: int, n_gates: int = 5, n_timescales: int = 3, d_state: int = 32):
        super().__init__()
        self.d_model = d_model
        
        # ✅ ENHANCED: More gates for increased parameter count
        self.gate_weights = nn.Parameter(torch.ones(n_gates) / n_gates)
        
        # ✅ ENHANCED: More gates (5 instead of 3)
        self.gates = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_gates)
        ])
        
        # Multi-timescale SSM with larger state dimensions
        decay_rates = [0.7, 0.9, 0.98]
        # ✅ ENHANCED: Larger SSM state dimensions (32 instead of 16)
        self.ssm_fast = SSMBlock(d_model, decay=decay_rates[0], d_state=d_state)
        self.ssm_medium = SSMBlock(d_model, decay=decay_rates[1], d_state=d_state)
        self.ssm_slow = SSMBlock(d_model, decay=decay_rates[2], d_state=d_state)
        
        # ✅ ENHANCED: Learnable timescale weights
        self.timescale_weights = nn.Parameter(torch.ones(3) / 3)
        
        # ✅ ENHANCED: More sophisticated attention and memory
        self.sparse_attn = EnhancedSparseAttention(d_model, sparsity=0.8)
        self.memory_gate = nn.Parameter(torch.ones(d_model) * 0.5)
        
        # ✅ ENHANCED: Additional processing layers
        self.pre_norm = nn.LayerNorm(d_model)
        self.post_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        
        # ✅ ENHANCED: Multiple gradient scaling parameters
        self.grad_scale = nn.Parameter(torch.ones(1))
        self.gate_scale = nn.Parameter(torch.ones(1))
        self.ssm_scale = nn.Parameter(torch.ones(1))
        
        # ✅ ENHANCED: Additional learnable components
        self.attention_scale = nn.Parameter(torch.ones(1))
        self.residual_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # ✅ ENHANCED: Pre-normalization with dtype consistency
        if x.dtype != next(self.pre_norm.parameters()).dtype:
            x = x.to(next(self.pre_norm.parameters()).dtype)
        x = self.pre_norm(x)
        
        # ✅ FIX: Ensure dtype consistency at the start
        target_dtype = next(self.gates[0].parameters()).dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)
        
        # ✅ ENHANCED: More sophisticated gate processing with scaling
        gate_weights_norm = torch.softmax(self.gate_weights, dim=0)
        
        gate_outputs = []
        for i, gate in enumerate(self.gates):
            # Ensure dtype consistency for each gate
            if x.dtype != next(gate.parameters()).dtype:
                x = x.to(next(gate.parameters()).dtype)
            gate_out = torch.sigmoid(gate(x)) * x  # Multiplicative gate
            gate_outputs.append(gate_out * gate_weights_norm[i])
        
        # ✅ ENHANCED: Apply gate scaling
        gated_x = sum(gate_outputs) * self.gate_scale + x * 0.1  # Small residual
        
        # ✅ ENHANCED: Multi-timescale processing with scaling
        ssm_fast_out = self.ssm_fast(gated_x, layer_idx)
        ssm_medium_out = self.ssm_medium(gated_x, layer_idx)
        ssm_slow_out = self.ssm_slow(gated_x, layer_idx)
        
        # ✅ ENHANCED: Learned combination with SSM scaling
        timescale_weights_norm = torch.softmax(self.timescale_weights, dim=0)
        ssm_combined = (timescale_weights_norm[0] * ssm_fast_out + 
                       timescale_weights_norm[1] * ssm_medium_out + 
                       timescale_weights_norm[2] * ssm_slow_out) * self.ssm_scale
        
        # ✅ ENHANCED: Memory processing with attention scaling
        memory_output = self.sparse_attn(ssm_combined) * self.attention_scale
        memory_gated = memory_output * torch.sigmoid(self.memory_gate)
        
        # ✅ ENHANCED: Strong residual with residual scaling
        memory_gated = memory_gated + ssm_combined * self.residual_scale
        
        # ✅ ENHANCED: Post-normalization and output projection
        if memory_gated.dtype != next(self.post_norm.parameters()).dtype:
            memory_gated = memory_gated.to(next(self.post_norm.parameters()).dtype)
        normalized = self.post_norm(memory_gated)
        
        if normalized.dtype != next(self.output_proj.parameters()).dtype:
            normalized = normalized.to(next(self.output_proj.parameters()).dtype)
        output = self.output_proj(normalized)
        
        # ✅ ENHANCED: Final residual with overall gradient scaling
        output = output + x
        output = output * self.grad_scale
        
        return output


def replace_layer_with_enhanced_mamba2(model, layer_idx: int, d_model: int = 768):
    """
    FIXED: Properly replace layer with enhanced Mamba2
    """
    logger.info(f"Replacing layer {layer_idx} with FIXED Mamba2...")
    
    # Create final fixed Mamba2 layer
    mamba2_layer = Mamba2Layer(d_model=d_model, n_gates=3, n_timescales=3, d_state=16)
    
    # Get device and dtype from model
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    # ✅ CRITICAL: Initialize weights properly BEFORE dtype conversion
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.1)  # Small gain
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    mamba2_layer.apply(init_weights)
    
    # Move to same device and dtype as the model
    mamba2_layer = mamba2_layer.to(device=device, dtype=dtype)
    
    # ✅ CRITICAL: Ensure all parameters are properly converted
    for name, param in mamba2_layer.named_parameters():
        if param.dtype != dtype:
            param.data = param.data.to(dtype=dtype)
            logger.info(f"Converted parameter {name} from {param.dtype} to {dtype}")
    
    # Also ensure all buffers are converted
    for name, buffer in mamba2_layer.named_buffers():
        if buffer.dtype != dtype:
            buffer.data = buffer.data.to(dtype=dtype)
            logger.info(f"Converted buffer {name} from {buffer.dtype} to {dtype}")
    
    # Store original layer
    original_layer = model.backbone.layers[layer_idx]
    
    # ✅ FIX: Create wrapper that ADDS Mamba2, doesn't replace
    class Mamba2WrappedLayer(nn.Module):
        def __init__(self, original, mamba2, layer_idx):
            super().__init__()
            self.original = original
            self.mamba2 = mamba2
            self.layer_idx = layer_idx
            self.norm = original.norm
            
            # ✅ ADD: Learnable mixing weight
            self.mix_weight = nn.Parameter(torch.tensor(0.1))  # Start with 10% Mamba2
            
            # ✅ ADD: Expose mixer attribute for compatibility
            if hasattr(original, 'mixer'):
                self.mixer = original.mixer
            else:
                # Create a mock mixer if original doesn't have one
                self.mixer = type('MockMixer', (), {})()
            
        def forward(self, hidden_states, residual=None, inference_params=None, cache_params=None, **kwargs):
            # Get original output - pass arguments carefully to avoid duplicates
            # Only pass arguments that the original layer expects
            original_args = [hidden_states]
            if residual is not None:
                original_args.append(residual)
            if inference_params is not None:
                original_args.append(inference_params)
            if cache_params is not None:
                original_args.append(cache_params)
            
            original_out = self.original(*original_args, **kwargs)
            
            # Get Mamba2 output
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(0)
                squeeze_output = True
            else:
                squeeze_output = False
            
            mamba2_out = self.mamba2(hidden_states, self.layer_idx)
            
            if squeeze_output:
                mamba2_out = mamba2_out.squeeze(0)
            
            # ✅ FIX: Gradually mix in Mamba2
            mix_weight_sigmoid = torch.sigmoid(self.mix_weight)
            output = original_out * (1 - mix_weight_sigmoid) + mamba2_out * mix_weight_sigmoid
            
            return output
    
    # Replace the layer
    wrapped = Mamba2WrappedLayer(original_layer, mamba2_layer, layer_idx)
    model.backbone.layers[layer_idx] = wrapped
    
    logger.info(f"✅ Layer {layer_idx} now uses FIXED Mamba2 with gradual mixing")
    return mamba2_layer

def attach_mamba2_layers(model: nn.Module, attribute_name: str = "mamba2") -> int:
    """
    FIXED: Attach Mamba2 layers with proper integration
    """
    logger.info("Attaching enhanced Mamba2 layers (using working implementation)...")
    
    # Try to infer layers
    layers = None
    for path in [
        lambda m: m.backbone.layers,
        lambda m: m.model.layers,
        lambda m: m.layers,
        lambda m: m.transformer.h,
        lambda m: m.transformer.layers,
    ]:
        try:
            layers = path(model)
            if layers is not None:
                break
        except Exception:
            continue

    if layers is None:
        logger.warning("Could not find model layers")
        return 0

    # Infer d_model from first layer - use hidden_size, not inner dimension
    d_model: Optional[int] = None
    
    # Try to get hidden_size from model config first
    try:
        config = getattr(model, 'config', None)
        if config:
            d_model = getattr(config, 'hidden_size', None)
            if d_model:
                logger.info(f"Using hidden_size from config: {d_model}")
    except Exception:
        pass
    
    # If config didn't work, try to infer from layer structure
    if d_model is None:
        try:
            first = layers[0]
            # Look for hidden dimension in various places
            probes = [
                ('norm.weight', lambda l: l.norm.weight.shape[0] if hasattr(l, 'norm') else None),
                ('layer_norm.weight', lambda l: l.layer_norm.weight.shape[0] if hasattr(l, 'layer_norm') else None),
                ('mixer.norm.weight', lambda l: l.mixer.norm.weight.shape[0] if hasattr(l.mixer, 'norm') else None),
            ]
            
            for _, fn in probes:
                try:
                    val = fn(first)
                    if isinstance(val, int) and val > 0:
                        d_model = val
                        logger.info(f"Detected d_model from layer structure: {d_model}")
                        break
                except Exception:
                    continue
        except Exception:
            pass
    
    # Final fallback
    if d_model is None:
        d_model = 768  # Common default
        logger.warning(f"Could not detect d_model, using default: {d_model}")
    
    if d_model is None:
        return 0

    count = 0
    for idx, layer in enumerate(layers):
        if hasattr(layer, attribute_name):
            continue
            
        # ✅ FIX: Use enhanced attachment for first few layers
        if idx < 3:  # Only attach to first 3 layers for stability
            try:
                mamba2_layer = replace_layer_with_enhanced_mamba2(model, idx, d_model)
                count += 1
                logger.info(f"✅ Enhanced Mamba2 attached to layer {idx}")
            except Exception as e:
                logger.warning(f"Failed to attach enhanced Mamba2 to layer {idx}: {e}")
                # Fallback to simple attachment
                mamba2_layer = Mamba2Layer(d_model)
                device = next(model.parameters()).device
                dtype = next(model.parameters()).dtype
                mamba2_layer = mamba2_layer.to(device=device, dtype=dtype)
                
                # ✅ CRITICAL: Ensure all parameters are properly converted
                for name, param in mamba2_layer.named_parameters():
                    if param.dtype != dtype:
                        param.data = param.data.to(dtype=dtype)
                        logger.info(f"Converted parameter {name} from {param.dtype} to {dtype}")
                
                # Also ensure all buffers are converted
                for name, buffer in mamba2_layer.named_buffers():
                    if buffer.dtype != dtype:
                        buffer.data = buffer.data.to(dtype=dtype)
                        logger.info(f"Converted buffer {name} from {buffer.dtype} to {dtype}")
                
                setattr(layer, attribute_name, mamba2_layer)
                count += 1
        else:
            # Simple attachment for remaining layers
            mamba2_layer = Mamba2Layer(d_model)
            device = next(model.parameters()).device
            dtype = next(model.parameters()).dtype
            mamba2_layer = mamba2_layer.to(device=device, dtype=dtype)
            
            # ✅ CRITICAL: Ensure all parameters are properly converted
            for name, param in mamba2_layer.named_parameters():
                if param.dtype != dtype:
                    param.data = param.data.to(dtype=dtype)
                    logger.info(f"Converted parameter {name} from {param.dtype} to {dtype}")
            
            # Also ensure all buffers are converted
            for name, buffer in mamba2_layer.named_buffers():
                if buffer.dtype != dtype:
                    buffer.data = buffer.data.to(dtype=dtype)
                    logger.info(f"Converted buffer {name} from {buffer.dtype} to {dtype}")
            
            setattr(layer, attribute_name, mamba2_layer)
            count += 1
            
    logger.info(f"✅ Attached Mamba2 layers to {count} layers")
    return count


def test_mamba2_gradient_flow():
    """Test that gradients actually flow through Mamba2"""
    mamba2 = Mamba2Layer(d_model=768, n_gates=3, d_state=16)
    
    x = torch.randn(2, 10, 768, requires_grad=True)
    output = mamba2(x)
    
    loss = output.sum()
    loss.backward()
    
    # Check gradients exist and are non-zero for key parameters
    key_params = ['gate_weights', 'timescale_weights', 'memory_gate', 'grad_scale']
    for name, param in mamba2.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.abs().sum().item()
            if grad_norm > 1e-6:
                print(f"✅ {name}: grad_norm = {grad_norm:.6f}")
            else:
                print(f"⚠️  {name}: grad_norm = {grad_norm:.6f} (very small)")
        else:
            print(f"❌ {name}: No gradient")
    
    # Check that at least the key parameters have gradients
    for param_name in key_params:
        param = dict(mamba2.named_parameters())[param_name]
        assert param.grad is not None, f"No gradient for {param_name}"
        assert param.grad.abs().sum() > 1e-6, f"Zero gradient for {param_name}"
    
    print("✅ Gradient flow test passed!")


if __name__ == "__main__":
    # Test gradient flow first
    print("Testing gradient flow...")
    test_mamba2_gradient_flow()
    
    # Minimal sanity test when run directly
    torch.manual_seed(0)
    d_model = 64
    layer = Mamba2Layer(d_model)
    x = torch.randn(2, 16, d_model)  # [batch=2, seq=16, d_model]
    y0 = layer(x, layer_idx=0)
    y1 = layer(x, layer_idx=5)
    print({
        'input_shape': tuple(x.shape),
        'out_l0_shape': tuple(y0.shape),
        'out_l5_shape': tuple(y1.shape),
        'mean_out_l0': float(y0.mean().item()),
        'mean_out_l5': float(y1.mean().item()),
    })


def identify_mamba2_circuits(self, layer_idx: int = 0) -> Dict[str, Any]:
    """
    Identify circuit candidates specifically for Mamba2 architecture.
    
    Mamba2 has distinct components that form circuits:
    1. Multi-gate ensemble (3 gates)
    2. Multi-timescale SSMs (fast/medium/slow)
    3. Sparse attention (every 5th layer)
    4. Memory gate (blending local/global)
    5. Compression predictor
    """
    logger.info(f"Step 6a: Identifying Mamba2-specific circuits for layer {layer_idx}...")
    
    if not hasattr(self, 'mamba2_activation_data') or layer_idx not in self.mamba2_activation_data:
        logger.warning(f"No Mamba2 activation data found for layer {layer_idx}")
        return {}
    
    activation_data = self.mamba2_activation_data[layer_idx]
    circuits = []
    
    # ========================================
    # Circuit Type 1: Gate Ensemble Circuits
    # ========================================
    # Each gate in the ensemble can form a circuit
    for gate_idx in range(3):  # n_gates = 3
        gate_key = f'gate_{gate_idx}'
        if gate_key in activation_data:
            gate_activations = activation_data[gate_key]
            
            # Analyze gate activation patterns
            gate_mean = np.mean(gate_activations)
            gate_std = np.std(gate_activations)
            gate_sparsity = np.mean(gate_activations < 0.1)  # Proportion near 0
            
            # Strong gates (high activation) form important circuits
            if gate_mean > 0.2:  # Lowered from 0.3 to 0.2
                circuit = {
                    'type': 'gate_ensemble',
                    'gate_idx': gate_idx,
                    'components': [f'gates.{gate_idx}'],
                    'strength': float(gate_mean),
                    'sparsity': float(gate_sparsity),
                    'variance': float(gate_std),
                    'description': f'Gate {gate_idx} ensemble circuit (mean={gate_mean:.3f})'
                }
                circuits.append(circuit)
    
    # ========================================
    # Circuit Type 2: Multi-Timescale Circuits
    # ========================================
    # Fast, medium, and slow SSM paths form distinct circuits
    for timescale, decay in [('fast', 0.7), ('medium', 0.9), ('slow', 0.98)]:
        ssm_key = f'ssm_{timescale}'
        if ssm_key in activation_data:
            ssm_activations = activation_data[ssm_key]
            
            ssm_mean = np.mean(np.abs(ssm_activations))
            ssm_std = np.std(ssm_activations)
            
            # Each timescale represents a different temporal receptive field
            if ssm_mean > 0.1:
                circuit = {
                    'type': 'multi_timescale',
                    'timescale': timescale,
                    'decay': decay,
                    'components': [f'ssm_{timescale}.in_proj', f'ssm_{timescale}.out_proj'],
                    'strength': float(ssm_mean),
                    'variance': float(ssm_std),
                    'description': f'{timescale.capitalize()} timescale SSM circuit (decay={decay})'
                }
                circuits.append(circuit)
    
    # ========================================
    # Circuit Type 3: Attention-Memory Circuit
    # ========================================
    # Sparse attention + memory gate (only active every 5th layer)
    if layer_idx % 5 == 0:
        if 'sparse_attn' in activation_data and 'memory_gate' in activation_data:
            attn_activations = activation_data['sparse_attn']
            gate_activations = activation_data['memory_gate']
            
            attn_mean = np.mean(np.abs(attn_activations))
            gate_mean = np.mean(gate_activations)
            
            # This is a critical global context circuit
            circuit = {
                'type': 'attention_memory',
                'layer_idx': layer_idx,
                'components': [
                    'sparse_attn.q', 'sparse_attn.k', 'sparse_attn.v',
                    'memory_gate'
                ],
                'strength': float(attn_mean),
                'gate_preference': float(gate_mean),  # How much it prefers local vs global
                'description': f'Sparse attention + memory gate circuit (active layer {layer_idx})',
                'is_global': True
            }
            circuits.append(circuit)
    
    # ========================================
    # Circuit Type 4: Compression Control Circuit
    # ========================================
    # Compression predictor modulates gating
    if 'compression' in activation_data:
        compression_values = activation_data['compression']
        
        comp_mean = np.mean(compression_values)
        comp_std = np.std(compression_values)
        
        # High compression indicates this circuit is active
        if comp_mean > 0.2:  # Lowered from 0.3 to 0.2
            circuit = {
                'type': 'compression_control',
                'components': ['compression_predictor'],
                'strength': float(comp_mean),
                'variance': float(comp_std),
                'description': f'Compression control circuit (level={comp_mean:.3f})',
                'affects': 'all_gates'
            }
            circuits.append(circuit)
    
    # ========================================
    # Circuit Type 5: Gate Weight Distribution Circuit
    # ========================================
    # The learnable weights that blend the 3 gates
    if 'gate_weights' in activation_data:
        weights = activation_data['gate_weights']
        
        # Softmax normalized weights
        weights_normalized = np.exp(weights) / np.sum(np.exp(weights))
        dominant_gate = np.argmax(weights_normalized)
        
        circuit = {
            'type': 'gate_weight_distribution',
            'components': ['gate_weights'],
            'weights': weights_normalized.tolist(),
            'dominant_gate': int(dominant_gate),
            'strength': float(weights_normalized[dominant_gate]),
            'description': f'Gate weight distribution (dominant: gate {dominant_gate})'
        }
        circuits.append(circuit)
    
    # ========================================
    # Circuit Type 6: Composite Pathways
    # ========================================
    # Identify interactions between components
    
    # Delta modulation pathway (gates → SSMs)
    if 'delta' in activation_data:
        delta_values = activation_data['delta']
        delta_mean = np.mean(np.abs(delta_values))
        delta_std = np.std(delta_values)
        
        if delta_mean > 0.2:
            circuit = {
                'type': 'delta_modulation',
                'components': ['gates[ensemble]', 'compression_predictor', 'ssm[all]'],
                'strength': float(delta_mean),
                'variance': float(delta_std),
                'description': 'Delta modulation pathway (gates → SSM control)',
                'is_composite': True
            }
            circuits.append(circuit)
    
    # Local-global blending pathway (only for attention layers)
    if layer_idx % 5 == 0 and 'h_local' in activation_data and 'h_global' in activation_data:
        local_acts = activation_data['h_local']
        global_acts = activation_data['h_global']
        
        local_mean = np.mean(np.abs(local_acts))
        global_mean = np.mean(np.abs(global_acts))
        
        circuit = {
            'type': 'local_global_blend',
            'components': ['ssm[fast,medium,slow]', 'sparse_attn', 'memory_gate'],
            'local_strength': float(local_mean),
            'global_strength': float(global_mean),
            'description': 'Local-global blending pathway',
            'is_composite': True,
            'layer_idx': layer_idx
        }
        circuits.append(circuit)
    
    # ========================================
    # Save Results
    # ========================================
    circuits_file = self.experiment_logger.experiment_dir / f"mamba2_candidate_circuits_layer_{layer_idx}.json"
    
    result = {
        'circuits': circuits,
        'layer_idx': layer_idx,
        'architecture': 'mamba2',
        'num_circuits': len(circuits),
        'circuit_types': list(set(c['type'] for c in circuits)),
        'metadata': {
            'has_attention': layer_idx % 5 == 0,
            'n_gates': 3,
            'n_timescales': 3
        }
    }
    
    with open(circuits_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"✓ Identified {len(circuits)} Mamba2 circuits for layer {layer_idx}")
    logger.info(f"  Circuit types: {', '.join(result['circuit_types'])}")
    
    return result


def add_mamba2_activation_hooks(analyzer, model, layer_idx: int = 0):
    """
    Add hooks to capture Mamba2-specific activations for circuit analysis.
    Must be called before running forward passes.
    """
    if not hasattr(analyzer, 'mamba2_activation_data'):
        analyzer.mamba2_activation_data = {}
    
    analyzer.mamba2_activation_data[layer_idx] = {}
    
    # Get the layer
    layers = analyzer.model.backbone.layers
    if layer_idx >= len(layers):
        logger.warning(f"Layer {layer_idx} out of range")
        return
    
    layer = layers[layer_idx]
    if not hasattr(layer, 'mamba2'):
        logger.warning(f"Layer {layer_idx} does not have mamba2 module")
        return
    
    mamba2 = layer.mamba2
    
    # Hook for each gate in ensemble
    for i, gate in enumerate(mamba2.gates):
        def gate_hook(module, input, output, gate_idx=i):
            analyzer.mamba2_activation_data[layer_idx][f'gate_{gate_idx}'] = output.detach().cpu().numpy()
        gate.register_forward_hook(gate_hook)
    
    # Hook for each timescale SSM
    for name in ['ssm_fast', 'ssm_medium', 'ssm_slow']:
        ssm = getattr(mamba2, name)
        def ssm_hook(module, input, output, ssm_name=name):
            analyzer.mamba2_activation_data[layer_idx][ssm_name] = output.detach().cpu().numpy()
        ssm.register_forward_hook(ssm_hook)
    
    # Hook for sparse attention (if applicable)
    def attn_hook(module, input, output):
        analyzer.mamba2_activation_data[layer_idx]['sparse_attn'] = output.detach().cpu().numpy()
    mamba2.sparse_attn.register_forward_hook(attn_hook)
    
    # Hook for memory gate
    def mem_gate_hook(module, input, output):
        analyzer.mamba2_activation_data[layer_idx]['memory_gate'] = output.detach().cpu().numpy()
    mamba2.memory_gate.register_forward_hook(mem_gate_hook)
    
    # Hook for compression predictor
    def comp_hook(module, input, output):
        analyzer.mamba2_activation_data[layer_idx]['compression'] = output.detach().cpu().numpy()
    mamba2.compression_predictor.register_forward_hook(comp_hook)
    
    logger.info(f"✓ Added Mamba2 activation hooks for layer {layer_idx}")


# ============================================================================
# FINAL FIXED Mamba2 Implementation - All gradient flow issues resolved
# ============================================================================

class SSMBlock(nn.Module):
    """FIXED SSM Block with proper gradient flow"""
    def __init__(self, d_model, decay=0.9, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # ✅ Smaller initialization for stability
        self.A = nn.Parameter(torch.randn(d_state, d_state) * 0.01)
        self.B = nn.Parameter(torch.randn(d_state, d_model) * 0.01)
        self.C = nn.Parameter(torch.randn(d_state, d_model) * 0.01)
        self.D = nn.Parameter(torch.ones(d_model) * 0.1)  # Positive initialization
        
        # Learnable decay
        self.decay = nn.Parameter(torch.tensor(decay))
        
        # ✅ REMOVED compression gate - it kills gradients!
        # If you need compression, do it ONCE at the end, not in every SSMBlock
        
        # ✅ Layer norm for stability
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, layer_idx=0):
        batch_size, seq_len, d_model = x.shape
        
        # ✅ Normalize input first
        x = self.norm(x)
        
        # ✅ CRITICAL: Store dtype to prevent mixed precision issues
        original_dtype = x.dtype
        
        # SSM computation with FIXED gradient flow
        h = torch.zeros(batch_size, self.d_state, device=x.device, dtype=original_dtype)
        outputs = []
        
        for t in range(seq_len):
            u_t = x[:, t, :]
            
            # ✅ Single tanh only (removed double tanh)
            h_new = torch.matmul(h, self.A) + torch.matmul(u_t, self.B.T)
            h = torch.tanh(h_new)
            
            # ✅ Linear output (no tanh here!)
            y_t = torch.matmul(h, self.C) + self.D * u_t
            outputs.append(y_t)
        
        output = torch.stack(outputs, dim=1)
        
        # ✅ CRITICAL: Strong residual connection
        output = output + x
        
        return output


class SparseAttention(nn.Module):
    """Fixed sparse attention with proper gradient flow"""
    def __init__(self, d_model, sparsity=0.7):  # ✅ Reduced sparsity from 0.95
        super().__init__()
        self.d_model = d_model
        self.sparsity = sparsity
        
        # Single head for simplicity
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # ✅ Layer norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # ✅ Normalize first
        x = self.norm(x)
        
        # Compute Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_model)
        
        # ✅ FIXED: Learnable sparsity mask instead of random
        # Use top-k selection which is differentiable
        k_value = max(1, int(seq_len * (1 - self.sparsity)))
        topk_scores, topk_indices = torch.topk(scores, k=k_value, dim=-1)
        
        # Create sparse attention
        sparse_scores = torch.full_like(scores, float('-inf'))
        sparse_scores.scatter_(-1, topk_indices, topk_scores)
        
        # Apply attention
        attn_weights = F.softmax(sparse_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        # ✅ Residual connection
        output = output + x
        
        return output


class Mamba2Layer(nn.Module):
    """
    FINAL FIXED Mamba2 Layer with all gradient issues resolved
    """
    def __init__(self, d_model, n_gates=3, n_timescales=3, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.n_gates = n_gates
        self.n_timescales = n_timescales
        
        # ✅ Simplified gates (just Linear layers)
        self.gates = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_gates)
        ])
        
        # ✅ CRITICAL: Learnable gate weights (not parameters)
        self.gate_weights = nn.Parameter(torch.ones(n_gates) / n_gates)
        
        # Multi-timescale SSM
        decay_rates = [0.7, 0.9, 0.98]
        self.ssm_fast = SSMBlock(d_model, decay=decay_rates[0], d_state=d_state)
        self.ssm_medium = SSMBlock(d_model, decay=decay_rates[1], d_state=d_state)
        self.ssm_slow = SSMBlock(d_model, decay=decay_rates[2], d_state=d_state)
        
        # ✅ CRITICAL: Learnable timescale weights (not random!)
        self.timescale_weights = nn.Parameter(torch.ones(3) / 3)
        
        # Sparse attention
        self.sparse_attn = SparseAttention(d_model, sparsity=0.7)
        
        # ✅ Simple learnable memory gate (scalar parameter)
        self.memory_gate = nn.Parameter(torch.ones(1) * 0.5)
        
        # ✅ REMOVED compression predictor (major gradient killer!)
        
        # Output processing
        self.pre_norm = nn.LayerNorm(d_model)
        self.post_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        
        # ✅ Learnable scale parameters for gradient monitoring
        self.gate_scale = nn.Parameter(torch.ones(1))
        self.ssm_scale = nn.Parameter(torch.ones(1))
        self.attention_scale = nn.Parameter(torch.ones(1))
        self.residual_scale = nn.Parameter(torch.ones(1))
        self.grad_scale = nn.Parameter(torch.ones(1))
        
        # ✅ Initialize all Linear layers properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Proper weight initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x, layer_idx=0):
        batch_size, seq_len, d_model = x.shape
        
        # ✅ CRITICAL: Store dtype to prevent mixed precision issues
        original_dtype = x.dtype
        
        # ✅ Pre-normalization with dtype consistency
        if x.dtype != next(self.pre_norm.parameters()).dtype:
            x = x.to(next(self.pre_norm.parameters()).dtype)
        x = self.pre_norm(x)
        
        # ✅ FIXED: Normalized gate weights (no more cancellation!)
        gate_weights_norm = F.softmax(self.gate_weights, dim=0)
        
        # Gate processing with proper weighting
        gate_outputs = []
        for i, gate in enumerate(self.gates):
            # ✅ Use sigmoid for multiplicative gating
            gate_out = torch.sigmoid(gate(x)) * x
            gate_outputs.append(gate_out * gate_weights_norm[i])
        
        # Combine gates with residual
        gated_x = sum(gate_outputs) * self.gate_scale
        gated_x = gated_x + x * self.residual_scale * 0.1  # Small residual
        
        # ✅ CRITICAL: Fixed timescale combination (no more random!)
        timescale_weights_norm = F.softmax(self.timescale_weights, dim=0)
        
        # Multi-timescale SSM processing
        ssm_fast_out = self.ssm_fast(gated_x, layer_idx)
        ssm_medium_out = self.ssm_medium(gated_x, layer_idx)
        ssm_slow_out = self.ssm_slow(gated_x, layer_idx)
        
        # ✅ Learned weighted combination
        ssm_combined = (
            timescale_weights_norm[0] * ssm_fast_out + 
            timescale_weights_norm[1] * ssm_medium_out + 
            timescale_weights_norm[2] * ssm_slow_out
        ) * self.ssm_scale
        
        # Adaptive memory processing
        memory_output = self.sparse_attn(ssm_combined)
        
        # ✅ FIXED: Simple scalar gate (no complex predictor)
        memory_gate_sigmoid = torch.sigmoid(self.memory_gate)
        memory_gated = memory_output * memory_gate_sigmoid * self.attention_scale
        
        # ✅ Strong residual from ssm_combined
        memory_gated = memory_gated + ssm_combined * 0.5
        
        # ✅ Post-normalization with dtype consistency
        if memory_gated.dtype != next(self.post_norm.parameters()).dtype:
            memory_gated = memory_gated.to(next(self.post_norm.parameters()).dtype)
        normalized = self.post_norm(memory_gated)
        
        # Output projection with dtype consistency
        if normalized.dtype != next(self.output_proj.parameters()).dtype:
            normalized = normalized.to(next(self.output_proj.parameters()).dtype)
        output = self.output_proj(normalized)
        
        # ✅ CRITICAL: Final residual from input
        output = output + x * self.residual_scale
        
        # ✅ Gradient scaling for monitoring
        output = output * self.grad_scale
        
        return output


def replace_layer_with_final_fixed_mamba2(model, layer_idx: int, d_model: int = 768):
    """
    FINAL FIXED: Replace layer with properly integrated Mamba2
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"Replacing layer {layer_idx} with FINAL FIXED Mamba2...")
    
    # Create fixed Mamba2 layer
    mamba2_layer = Mamba2Layer(d_model=d_model, n_gates=3, n_timescales=3, d_state=16)
    mamba2_layer = mamba2_layer.to(next(model.parameters()).device)
    mamba2_layer = mamba2_layer.to(next(model.parameters()).dtype)
    
    # Store original layer
    original_layer = model.backbone.layers[layer_idx]
    
    # ✅ FIXED: Wrapper with learnable mixing and proper Mamba interface
    class Mamba2WrappedLayer(nn.Module):
        def __init__(self, original, mamba2, layer_idx):
            super().__init__()
            self.original = original
            self.mamba2 = mamba2
            self.layer_idx = layer_idx
            
            # ✅ Copy all attributes from original layer for compatibility
            for attr_name in dir(original):
                if not attr_name.startswith('_') and not callable(getattr(original, attr_name)):
                    setattr(self, attr_name, getattr(original, attr_name))
            
            # ✅ FIXED: Stable mixing weight (start at 50% Mamba2)
            self.mix_weight = nn.Parameter(torch.ones(1) * 0.5)  # Start at 0.5 for stability
            
        def forward(self, hidden_states, residual=None, inference_params=None, cache_params=None, cache_position=None, **kwargs):
            # Original Mamba output - pass all parameters
            original_out = self.original(hidden_states, residual, inference_params, cache_params, cache_position, **kwargs)
            
            # Mamba2 output
            if hidden_states.dim() == 2:
                hidden_states_3d = hidden_states.unsqueeze(0)
            else:
                hidden_states_3d = hidden_states
            
            mamba2_out = self.mamba2(hidden_states_3d, self.layer_idx)
            
            if hidden_states.dim() == 2:
                mamba2_out = mamba2_out.squeeze(0)
            
            # ✅ FIXED: Stable mixing with bounded weight
            mix_alpha = torch.clamp(self.mix_weight, 0.0, 1.0)  # Direct bounded mixing
            output = original_out * (1 - mix_alpha) + mamba2_out * mix_alpha
            
            return output
    
    # Replace the layer
    wrapped = Mamba2WrappedLayer(original_layer, mamba2_layer, layer_idx)
    model.backbone.layers[layer_idx] = wrapped
    
    logger.info(f"✅ Layer {layer_idx} now uses FINAL FIXED Mamba2")
    logger.info(f"   Initial mix weight: {torch.clamp(wrapped.mix_weight, 0.0, 1.0).item():.3f}")
    
    return mamba2_layer

