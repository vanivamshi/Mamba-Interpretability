"""
Enhanced SPD Extension with Gradient Flow Fixes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import sys
from typing import List, Dict, Any, Optional, Tuple
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
import json
import logging
import psutil
import gc

logger = logging.getLogger(__name__)

# ============================================================================
# ENHANCED MAMBA2 IMPLEMENTATION FOR STEPS 17, 18, 19
# ============================================================================

class SSMBlock(nn.Module):
    """Enhanced SSM Block with distributed compression"""
    def __init__(self, d_model, decay=0.9, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # State transition matrix with learnable decay
        self.A = nn.Parameter(torch.randn(d_state, d_state) * 0.1)
        self.B = nn.Parameter(torch.randn(d_state, d_model) * 0.1)
        self.C = nn.Parameter(torch.randn(d_state, d_model) * 0.1)
        self.D = nn.Parameter(torch.randn(d_model) * 0.1)
        
        # Learnable decay rate
        self.decay = nn.Parameter(torch.tensor(decay))
        
        # Compression controller
        self.compression_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, layer_idx=0):
        batch_size, seq_len, d_model = x.shape
        
        # Apply compression gate
        compression_factor = self.compression_gate(x.mean(dim=1, keepdim=True))
        x_compressed = x * compression_factor
        
        # SSM computation
        h = torch.zeros(batch_size, self.d_state, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            # State update
            u_t = x_compressed[:, t, :]  # Input at time t (batch_size, d_model)
            # h: (batch_size, d_state), A: (d_state, d_state), B: (d_state, d_model)
            h = torch.tanh(torch.matmul(h, self.A)) + torch.tanh(torch.matmul(u_t, self.B.T))
            
            # Output
            # h: (batch_size, d_state), C: (d_state, d_model)
            y_t = torch.tanh(torch.matmul(h, self.C)) + self.D * u_t
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1)

class GateModule(nn.Module):
    """Multi-gate module for redundancy"""
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.gate(x)

class LearnedGate(nn.Module):
    """Learned gating mechanism"""
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        return x * self.gate

class SparseAttention(nn.Module):
    """Sparse attention fallback for memory"""
    def __init__(self, d_model, sparsity=0.95):
        super().__init__()
        self.d_model = d_model
        self.sparsity = sparsity
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_model)
        
        # Apply sparsity mask
        mask = torch.rand_like(scores) > self.sparsity
        sparse_attn = scores.masked_fill(~mask, float('-inf'))
        
        # Apply attention
        attn_output = torch.matmul(F.softmax(sparse_attn, dim=-1), V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.out_proj(attn_output)


class Mamba2Layer(nn.Module):
    """
    Enhanced Mamba2 Layer with:
    - Multi-gate redundancy
    - Distributed compression  
    - Adaptive memory
    - Stable compression
    - Multi-timescale processing
    """
    def __init__(self, d_model, n_gates=3, n_timescales=3, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.n_gates = n_gates
        self.n_timescales = n_timescales
        
        # Multi-gate ensemble for redundancy
        self.gates = nn.ModuleList([
            GateModule(d_model) for _ in range(n_gates)
        ])
        self.gate_weights = nn.Parameter(torch.ones(n_gates) / n_gates)
        
        # Multi-timescale SSM for distributed processing
        decay_rates = [0.7, 0.9, 0.98]  # Fast, medium, slow
        self.ssm_fast = SSMBlock(d_model, decay=decay_rates[0], d_state=d_state)
        self.ssm_medium = SSMBlock(d_model, decay=decay_rates[1], d_state=d_state)
        self.ssm_slow = SSMBlock(d_model, decay=decay_rates[2], d_state=d_state)
        
        # Adaptive memory with sparse attention fallback
        self.sparse_attn = SparseAttention(d_model, sparsity=0.95)
        self.memory_gate = LearnedGate(d_model)
        
        # Compression controller for stable compression
        self.compression_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Output normalization and projection
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x, layer_idx=0):
        batch_size, seq_len, d_model = x.shape
        
        # Multi-gate processing
        gate_outputs = []
        for i, gate in enumerate(self.gates):
            gate_out = gate(x) * self.gate_weights[i]
            gate_outputs.append(gate_out)
        
        # Combine gate outputs
        gated_x = sum(gate_outputs)
        
        # Multi-timescale SSM processing
        ssm_fast_out = self.ssm_fast(gated_x, layer_idx)
        ssm_medium_out = self.ssm_medium(gated_x, layer_idx)
        ssm_slow_out = self.ssm_slow(gated_x, layer_idx)
        
        # Weighted combination of timescales
        timescale_weights = torch.softmax(torch.randn(3, device=x.device), dim=0)
        ssm_combined = (timescale_weights[0] * ssm_fast_out + 
                       timescale_weights[1] * ssm_medium_out + 
                       timescale_weights[2] * ssm_slow_out)
        
        # Adaptive memory processing
        memory_output = self.sparse_attn(ssm_combined)
        memory_gated = memory_output * self.memory_gate(ssm_combined)
        
        # Compression control
        compression_factor = self.compression_predictor(memory_gated.mean(dim=1, keepdim=True))
        compressed_output = memory_gated * compression_factor
        
        # Final processing
        normalized = self.layer_norm(compressed_output)
        output = self.output_proj(normalized)
        
        return output


class EnhancedSPDAnalyzer:
    """Enhanced SPD Analyzer with robust gradient computation"""
    
    def __init__(self, mamba_analyzer, device: Optional[torch.device] = None):
        self.owner = mamba_analyzer
        self.model = mamba_analyzer.model
        self.tokenizer = mamba_analyzer.tokenizer
        self.experiment_logger = mamba_analyzer.experiment_logger
        self.config = getattr(mamba_analyzer, "config", None)
        
        # Device setup
        if device is not None:
            self.device = device
        elif self.config is not None and hasattr(self.config, 'device'):
            self.device = self.config.device
        else:
            self.device = next(self.model.parameters()).device

        # Parameter info
        self.param_info = self._collect_param_info()
        self._total_params = sum(p['numel'] for p in self.param_info)
        
        # Memory settings
        self.max_memory_gb = 31.0
        
        # Enable memory optimization
        self._enable_memory_optimization()

    def _collect_param_info(self) -> List[Dict[str, Any]]:
        """Collect parameter information with enhanced metadata"""
        info = []
        offset = 0
        
        for name, p in self.model.named_parameters():
            numel = p.numel()
            info.append({
                "name": name,
                "param": p,
                "shape": tuple(p.shape),
                "numel": numel,
                "idx_range": (offset, offset + numel),
                "requires_grad": p.requires_grad,
                "dtype": p.dtype,
                "device": p.device
            })
            offset += numel
        
        logger.info(f"Collected {len(info)} parameter tensors, total params={offset}")
        return info

    def _enable_memory_optimization(self):
        """Enhanced memory optimization that preserves gradients"""
        # Disable gradient checkpointing during analysis
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            original_checkpointing = getattr(self.model, 'gradient_checkpointing', False)
            if hasattr(self.model, 'gradient_checkpointing_disable'):
                self.model.gradient_checkpointing_disable()
            self._original_checkpointing = original_checkpointing
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Enhanced memory optimization enabled")

    def _force_gradient_enable(self):
        """Force enable gradients for all parameters with verification"""
        logger.info("Forcing gradient enablement for all parameters...")
        
        enabled_count = 0
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                param.requires_grad_(True)
                enabled_count += 1
        
        logger.info(f"Enabled gradients for {enabled_count} parameters")
        
        # Verify gradient enablement
        grad_params = sum(1 for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters with gradients enabled: {grad_params}/{len(list(self.model.parameters()))}")

    def _robust_gradient_computation(self, inputs: Dict[str, torch.Tensor], target_token_idx: int = -1):
        """Robust gradient computation with multiple fallback strategies"""
        
        # Strategy 1: Standard gradient computation
        try:
            return self._standard_gradient_computation(inputs, target_token_idx)
        except Exception as e:
            logger.warning(f"Standard gradient computation failed: {e}")
        
        # Strategy 2: Layer-wise gradient computation
        try:
            return self._layer_wise_gradient_computation(inputs, target_token_idx)
        except Exception as e:
            logger.warning(f"Layer-wise gradient computation failed: {e}")
        
        # Strategy 3: Manual gradient computation
        try:
            return self._manual_gradient_computation(inputs, target_token_idx)
        except Exception as e:
            logger.error(f"All gradient computation strategies failed: {e}")
            raise

    def _standard_gradient_computation(self, inputs: Dict[str, torch.Tensor], target_token_idx: int = -1):
        """Standard gradient computation with enhanced monitoring"""
        
        # Force enable gradients
        self._force_gradient_enable()
        
        # Clear existing gradients
        self.model.zero_grad()
        
        # Move inputs to device
        device_inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass with gradient tracking
        with torch.enable_grad():
            outputs = self.model(**device_inputs)
            logits = outputs.logits
            
            # Multiple loss targets to capture different computation paths
            if target_token_idx < 0:
                target_token_idx = logits.shape[1] + target_token_idx
            
            # Use multiple loss components
            last_token_logits = logits[:, target_token_idx, :]
            
            # Component 1: Max logit
            loss1 = -last_token_logits.max(dim=-1)[0].mean()
            
            # Component 2: Top-k logits
            topk_logits = last_token_logits.topk(5, dim=-1)[0]
            loss2 = -topk_logits.mean()
            
            # Component 3: Specific token (if provided)
            if hasattr(self, 'target_token_id'):
                target_logits = last_token_logits[:, self.target_token_id]
                loss3 = -target_logits.mean()
            else:
                loss3 = torch.tensor(0.0, device=self.device)
            
            # Combined loss
            combined_loss = loss1 + 0.5 * loss2 + 0.1 * loss3
        
        # Backward pass
        combined_loss.backward()
        
        # Verify gradients
        grad_stats = self._analyze_gradient_stats()
        logger.info(f"Gradient stats: {grad_stats}")
        
        return combined_loss.item()

    def _layer_wise_gradient_computation(self, inputs: Dict[str, torch.Tensor], target_token_idx: int = -1):
        """Layer-wise gradient computation for large models"""
        
        self._force_gradient_enable()
        self.model.zero_grad()
        
        device_inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Use forward hooks to capture intermediate outputs
        gradients = {}
        
        def backward_hook(module, grad_input, grad_output):
            if hasattr(module, 'gradient_norm'):
                module.gradient_norm = grad_output[0].norm().item()
        
        # Register hooks
        hooks = []
        for name, module in self.model.named_modules():
            if any(isinstance(module, layer_type) for layer_type in [nn.Linear, nn.Conv1d, nn.LayerNorm]):
                hook = module.register_full_backward_hook(backward_hook)
                hooks.append(hook)
        
        try:
            with torch.enable_grad():
                outputs = self.model(**device_inputs)
                logits = outputs.logits
                
                # Simple loss
                loss = -logits[:, target_token_idx, :].max(dim=-1)[0].mean()
                loss.backward()
                
            return loss.item()
            
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

    def _manual_gradient_computation(self, inputs: Dict[str, torch.Tensor], target_token_idx: int = -1):
        """Manual gradient computation using autograd.grad"""
        
        self._force_gradient_enable()
        self.model.zero_grad()
        
        device_inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.enable_grad():
            outputs = self.model(**device_inputs)
            logits = outputs.logits
            target = logits[:, target_token_idx, :].max(dim=-1)[0].mean()
        
        # Manual gradient computation
        params = [p for p in self.model.parameters() if p.requires_grad]
        gradients = torch.autograd.grad(target, params, allow_unused=True)
        
        # Assign gradients manually
        for param, grad in zip(params, gradients):
            if grad is not None:
                param.grad = grad
        
        return target.item()

    def _analyze_gradient_stats(self) -> Dict[str, float]:
        """Analyze gradient statistics"""
        total_params = 0
        zero_grads = 0
        non_zero_grads = 0
        grad_norms = []
        
        for name, param in self.model.named_parameters():
            total_params += 1
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                if grad_norm > 1e-8:
                    non_zero_grads += 1
                else:
                    zero_grads += 1
            else:
                zero_grads += 1
        
        stats = {
            'total_parameters': total_params,
            'parameters_with_gradients': non_zero_grads,
            'parameters_without_gradients': zero_grads,
            'gradient_coverage': non_zero_grads / total_params if total_params > 0 else 0,
            'mean_gradient_norm': np.mean(grad_norms) if grad_norms else 0,
            'max_gradient_norm': np.max(grad_norms) if grad_norms else 0,
        }
        
        return stats

    def compute_robust_attributions(self, inputs: Dict[str, torch.Tensor], 
                                  layer_idx: Optional[int] = None,
                                  n_steps: int = 4) -> np.ndarray:
        """Compute robust attributions with gradient verification"""
        
        logger.info("Computing robust attributions...")
        
        # Pre-warm gradients
        # Use enhanced gradient warming for better coverage
        self._enhanced_gradient_warming(inputs, num_passes=5)
        
        # Compute baseline loss
        with torch.no_grad():
            device_inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**device_inputs)
            baseline_loss = -outputs.logits[:, -1, :].max().item()
        
        # Create parameter mask if layer specified
        param_mask = None
        if layer_idx is not None:
            param_mask = self._create_layer_mask(layer_idx)
        
        # Compute integrated gradients
        attributions = self._compute_integrated_gradients(
            inputs, n_steps=10, param_mask=param_mask  # Increased steps for better attribution quality
        )
        
        # Verify attributions
        attribution_stats = self._analyze_attribution_stats(attributions, param_mask)
        logger.info(f"Attribution stats: {attribution_stats}")
        
        return attributions

    def _gradient_pre_warming(self, inputs: Dict[str, torch.Tensor], num_passes: int = 3):
        """Enhanced gradient pre-warming"""
        
        logger.info("Running gradient pre-warming...")
        
        best_coverage = 0
        
        for pass_idx in range(num_passes):
            self.model.zero_grad()
            
            # Use different loss targets each pass
            loss = self._robust_gradient_computation(inputs, target_token_idx=-1)
            
            # Analyze gradient coverage
            stats = self._analyze_gradient_stats()
            coverage = stats['gradient_coverage']
            
            logger.info(f"Pre-warm pass {pass_idx + 1}: coverage = {coverage:.3f}")
            
            if coverage > best_coverage:
                best_coverage = coverage
            
            # Stop if we have good coverage
            if coverage > 0.8:
                break
        
        logger.info(f"Best gradient coverage: {best_coverage:.3f}")
        return best_coverage

    def _enhanced_gradient_warming(self, inputs: Dict[str, torch.Tensor], num_passes: int = 5):
        """Aggressive gradient warming to address high zero gradient ratio"""
        
        logger.info("Running aggressive gradient warming...")
        
        best_coverage = 0
        
        for pass_idx in range(num_passes):
            self.model.zero_grad()
            
            # Multiple loss targets for better gradient flow
            device_inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.enable_grad():
                outputs = self.model(**device_inputs)
                logits = outputs.logits
                
                # Combine multiple objectives
                loss1 = -logits[:, -1, :].max()  # Last token
                loss2 = -logits[:, :-1, :].mean()  # All tokens
                loss3 = logits.norm()  # L2 regularization
                
                # Add mix_weight regularization to reduce BIF instability
                mix_weight_reg = 0.0
                for name, param in self.model.named_parameters():
                    if 'mix_weight' in name:
                        # Regularize mix_weight to stay close to 0.5
                        mix_weight_reg += 0.01 * (param - 0.5).pow(2).sum()
                
                total_loss = loss1 + 0.3 * loss2 + 0.1 * loss3 + mix_weight_reg
                total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Analyze gradient coverage
            stats = self._analyze_gradient_stats()
            coverage = stats['gradient_coverage']
            
            logger.info(f"Aggressive warm pass {pass_idx + 1}: coverage = {coverage:.3f}")
            
            if coverage > best_coverage:
                best_coverage = coverage
            
            # Stop if we have good coverage
            if coverage > 0.5:  # Lower threshold for aggressive warming
                break
        
        logger.info(f"Best aggressive gradient coverage: {best_coverage:.3f}")
        return best_coverage

    def _create_layer_mask(self, layer_idx: int) -> np.ndarray:
        """Create parameter mask for specific layer"""
        mask = np.zeros(self._total_params, dtype=np.float32)
        
        layer_keywords = [
            f"layers.{layer_idx}.",
            f"layer_{layer_idx}.",
            f".{layer_idx}."
        ]
        
        for pinfo in self.param_info:
            if any(keyword in pinfo['name'] for keyword in layer_keywords):
                lo, hi = pinfo['idx_range']
                mask[lo:hi] = 1.0
        
        mask_sum = mask.sum()
        logger.info(f"Created layer mask for layer {layer_idx}: {mask_sum} parameters")
        
        return mask

    def _compute_integrated_gradients(self, inputs: Dict[str, torch.Tensor], 
                                    n_steps: int = 10,  # Increased from 4 for better attribution quality
                                    param_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute integrated gradients with improved attribution quality"""
        
        # Get baseline and current parameters
        baseline = self._create_baseline_parameters()
        current = self._flatten_params(as_numpy=True)
        
        # Initialize attribution accumulator
        total_attribution = np.zeros(self._total_params, dtype=np.float32)
        
        # Use non-linear path interpolation for better attribution quality
        alphas = np.power(np.linspace(0, 1, n_steps), 2)  # Quadratic spacing
        
        # Compute integrated gradients along the path
        for i, alpha in enumerate(alphas):
            # Interpolate parameters
            interpolated = baseline + alpha * (current - baseline)
            
            # Assign to model
            self._assign_flat_to_model(torch.from_numpy(interpolated))
            
            # Compute gradient at this point
            self.model.zero_grad()
            loss = self._robust_gradient_computation(inputs)
            
            # Collect gradients
            grad_vector = self._collect_gradient_vector()
            
            # Accumulate attribution with proper weighting
            if i < len(alphas) - 1:
                # Use trapezoidal rule for better integration
                weight = (alphas[i+1] - alphas[i-1]) / 2 if i > 0 else alphas[1] / 2
            else:
                weight = (alphas[i] - alphas[i-1]) / 2
            
            total_attribution += grad_vector * (current - baseline) * weight
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Restore original parameters
        self._assign_flat_to_model(torch.from_numpy(current))
        
        # Average attribution
        avg_attribution = total_attribution / n_steps
        
        # Apply mask if provided
        if param_mask is not None:
            avg_attribution = avg_attribution * param_mask
        
        return avg_attribution

    def _create_baseline_parameters(self) -> np.ndarray:
        """Create baseline parameters for integrated gradients"""
        current = self._flatten_params(as_numpy=True)
        
        # Use small random baseline to avoid zero gradients
        baseline = np.random.normal(0, 0.01, current.shape).astype(np.float32)
        
        return baseline

    def _collect_gradient_vector(self) -> np.ndarray:
        """Collect gradient vector from model parameters"""
        grad_parts = []
        
        for pinfo in self.param_info:
            param = pinfo['param']
            if param.grad is not None:
                grad_flat = param.grad.detach().cpu().numpy().flatten()
            else:
                grad_flat = np.zeros(pinfo['numel'], dtype=np.float32)
            
            grad_parts.append(grad_flat)
        
        return np.concatenate(grad_parts)

    def _analyze_attribution_stats(self, attributions: np.ndarray, 
                                 param_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Analyze attribution statistics"""
        
        if param_mask is not None:
            masked_attributions = attributions[param_mask > 0]
        else:
            masked_attributions = attributions
        
        stats = {
            'total_attribution_norm': float(np.linalg.norm(attributions)),
            'mean_attribution': float(np.mean(np.abs(masked_attributions))),
            'std_attribution': float(np.std(masked_attributions)),
            'max_attribution': float(np.max(np.abs(masked_attributions))),
            'min_attribution': float(np.min(np.abs(masked_attributions))),
            'zero_attributions': float(np.sum(np.abs(masked_attributions) < 1e-10)),
            'non_zero_attributions': float(np.sum(np.abs(masked_attributions) >= 1e-10)),
        }
        
        if len(masked_attributions) > 0:
            stats['zero_ratio'] = stats['zero_attributions'] / len(masked_attributions)
        else:
            stats['zero_ratio'] = 0.0
        
        return stats

    def _flatten_params(self, as_numpy: bool = False):
        """Flatten parameters with memory efficiency"""
        parts = []
        
        for pinfo in self.param_info:
            param_flat = pinfo['param'].detach().reshape(-1).cpu()
            parts.append(param_flat)
        
        flat = torch.cat(parts)
        
        if as_numpy:
            return flat.numpy()
        else:
            return flat

    def _assign_flat_to_model(self, flat: torch.Tensor):
        """Assign flattened parameters back to model"""
        assert flat.numel() == self._total_params
        
        ptr = 0
        for pinfo in self.param_info:
            numel = pinfo['numel']
            chunk = flat[ptr:ptr+numel].view(pinfo['shape'])
            
            # Move to correct device
            chunk = chunk.to(pinfo['param'].device)
            
            # Assign with gradient preservation
            with torch.no_grad():
                pinfo['param'].copy_(chunk)
            
            ptr += numel

    def run_enhanced_spd_analysis(self, 
                                layer_idx: int = 0,
                                reference_text: str = "The quick brown fox jumps over the lazy dog.",
                                n_attrib_steps: int = 4,
                                n_samples: int = 10,
                                n_clusters: int = 6) -> Dict[str, Any]:
        """Enhanced SPD analysis with robust gradient handling"""
        
        logger.info(f"Running enhanced SPD analysis for layer {layer_idx}")
        
        # Prepare inputs
        inputs = self.tokenizer(
            reference_text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=128
        )
        
        # Step 1: Compute robust attributions
        logger.info("Step 1: Computing robust attributions...")
        attributions = self.compute_robust_attributions(
            inputs, layer_idx=layer_idx, n_steps=n_attrib_steps
        )
        
        # Step 2: Cluster parameters
        logger.info("Step 2: Clustering parameters...")
        clusters = self._enhanced_clustering(attributions, layer_idx, n_clusters)
        
        # Step 3: Compute susceptibilities
        logger.info("Step 3: Computing susceptibilities...")
        cluster_susceptibilities = self._compute_cluster_susceptibilities(
            inputs, clusters, n_samples
        )
        
        # Step 4: Compute BIFs for important parameters
        logger.info("Step 4: Computing BIFs...")
        bif_results = self._compute_enhanced_bifs(inputs, attributions, clusters)
        
        # Compile results
        results = {
            'layer_idx': layer_idx,
            'attributions_stats': self._analyze_attribution_stats(attributions),
            'clusters': clusters,
            'cluster_susceptibilities': cluster_susceptibilities,
            'bif_results': bif_results,
            'analysis_method': 'enhanced_spd',
            'timestamp': str(np.datetime64('now'))
        }
        
        # Save results
        self._save_results(results, layer_idx)
        
        logger.info("Enhanced SPD analysis completed successfully")
        return results

    def _enhanced_clustering(self, attributions: np.ndarray, layer_idx: int, n_clusters: int) -> Dict[str, Any]:
        """Enhanced clustering with parameter type awareness"""
        
        # Get layer parameters
        layer_params = []
        for pinfo in self.param_info:
            if f"layers.{layer_idx}." in pinfo['name'] or f"layer_{layer_idx}." in pinfo['name']:
                layer_params.append(pinfo)
        
        logger.info(f"Clustering {len(layer_params)} layer parameters")
        
        # Extract features for clustering
        features = []
        param_names = []
        
        for pinfo in layer_params:
            lo, hi = pinfo['idx_range']
            param_attributions = attributions[lo:hi]
            
            # Skip if no attributions for this parameter
            if len(param_attributions) == 0:
                continue
            
            # Enhanced feature extraction with more discriminative features
            feature_vector = [
                np.mean(param_attributions),           # Mean attribution
                np.std(param_attributions),            # Std of attributions
                np.max(np.abs(param_attributions)),    # Max absolute attribution
                np.sum(np.abs(param_attributions)),    # Total attribution mass
                np.median(param_attributions),         # Median attribution
                np.sum(param_attributions > 0),        # Positive attributions count
                np.sum(param_attributions < 0),        # Negative attributions count
                pinfo['numel'],                        # Parameter count
                np.percentile(param_attributions, 90), # 90th percentile
                np.percentile(param_attributions, 10), # 10th percentile
                np.percentile(param_attributions, 75), # 75th percentile
                np.percentile(param_attributions, 25), # 25th percentile
            ]
            
            # Add statistical shape features
            try:
                import scipy.stats
                feature_vector.extend([
                    scipy.stats.skew(param_attributions),   # Skewness
                    scipy.stats.kurtosis(param_attributions), # Kurtosis
                ])
            except ImportError:
                # Fallback if scipy not available
                feature_vector.extend([0.0, 0.0])
            
            # Add parameter type feature
            param_type = self._classify_parameter_type(pinfo['name'])
            feature_vector.extend(param_type)
            
            features.append(feature_vector)
            param_names.append(pinfo['name'])
        
        features = np.array(features)
        
        # Normalize features with NaN handling
        features_mean = np.nanmean(features, axis=0)
        features_std = np.nanstd(features, axis=0)
        features = (features - features_mean) / (features_std + 1e-8)
        
        # Handle any remaining NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Cluster with increased count for better discrimination
        n_clusters = min(8, len(features))  # Try 8 clusters instead of 6
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
        else:
            cluster_labels = np.zeros(len(features), dtype=int)
        
        # Organize clusters
        clusters = {}
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_params = [param_names[i] for i in cluster_indices]
            
            clusters[str(cluster_id)] = {
                'parameter_names': cluster_params,
                'parameter_count': len(cluster_params),
                'mean_attribution': float(np.mean([
                    np.sum(np.abs(attributions[self._get_param_range(pname)[0]:self._get_param_range(pname)[1]]))
                    for pname in cluster_params
                ]))
            }
        
        return {'clusters': clusters, 'n_clusters': n_clusters}

    def _classify_parameter_type(self, param_name: str) -> List[float]:
        """Classify parameter type for better clustering"""
        
        # One-hot encoding for parameter types
        type_encoding = [0.0] * 6  # SSM, Linear, Norm, Bias, Conv, Other
        
        if any(keyword in param_name for keyword in ['A', 'B', 'D', 'delta', 'ssm']):
            type_encoding[0] = 1.0  # SSM
        elif 'proj' in param_name or 'linear' in param_name:
            type_encoding[1] = 1.0  # Linear
        elif 'norm' in param_name:
            type_encoding[2] = 1.0  # Norm
        elif 'bias' in param_name:
            type_encoding[3] = 1.0  # Bias
        elif 'conv' in param_name:
            type_encoding[4] = 1.0  # Conv
        else:
            type_encoding[5] = 1.0  # Other
        
        return type_encoding

    def _get_param_range(self, param_name: str) -> Tuple[int, int]:
        """Get parameter range by name"""
        for pinfo in self.param_info:
            if pinfo['name'] == param_name:
                return pinfo['idx_range']
        return (0, 0)

    def _compute_cluster_susceptibilities(self, inputs: Dict[str, torch.Tensor], 
                                        clusters: Dict[str, Any], 
                                        n_samples: int) -> Dict[str, Any]:
        """Compute susceptibilities for each cluster"""
        
        base_flat = self._flatten_params(as_numpy=True)
        susceptibilities = {}
        
        for cluster_id, cluster_info in clusters['clusters'].items():
            logger.info(f"Computing susceptibility for cluster {cluster_id}")
            
            # Create cluster mask
            cluster_mask = np.zeros(self._total_params, dtype=np.float32)
            for param_name in cluster_info['parameter_names']:
                lo, hi = self._get_param_range(param_name)
                cluster_mask[lo:hi] = 1.0
            
            # Compute susceptibility
            sus = self._estimate_susceptibility(
                inputs, cluster_mask, n_samples=min(n_samples, 5)
            )
            
            susceptibilities[cluster_id] = {
                'parameter_names': cluster_info['parameter_names'],
                'susceptibility': sus,
                'parameter_count': cluster_info['parameter_count']
            }
        
        return susceptibilities

    def _estimate_susceptibility(self, inputs: Dict[str, torch.Tensor], 
                               mask: np.ndarray, n_samples: int) -> Dict[str, Any]:
        """Estimate susceptibility with robust computation"""
        
        base_flat = self._flatten_params(as_numpy=True)
        effects = []
        norms = []
        
        for i in range(n_samples):
            # Generate perturbation
            perturbation = self._generate_perturbation(base_flat, mask)
            
            # Assign perturbed parameters
            self._assign_flat_to_model(torch.from_numpy(perturbation))
            
            # Compute effect
            with torch.no_grad():
                device_inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**device_inputs)
                logits = outputs.logits
                effect = -logits[:, -1, :].max().item()  # Negative loss = effect
                
            effects.append(effect)
            norms.append(np.linalg.norm(perturbation - base_flat))
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Restore original parameters
        self._assign_flat_to_model(torch.from_numpy(base_flat))
        
        effects = np.array(effects)
        
        return {
            'mean_effect': float(np.mean(effects)),
            'std_effect': float(np.std(effects)),
            'cov_effect': float(np.std(effects) / (np.mean(effects) + 1e-10)),
            'samples_effects': effects.tolist(),
            'perturbation_norms': norms
        }

    def _generate_perturbation(self, base: np.ndarray, mask: np.ndarray, sigma: float = 1e-3) -> np.ndarray:
        """Generate single perturbation"""
        noise = np.random.normal(0, sigma, base.shape).astype(np.float32)
        return base + noise * mask

    def _compute_enhanced_bifs(self, inputs: Dict[str, torch.Tensor], 
                             attributions: np.ndarray, 
                             clusters: Dict[str, Any]) -> Dict[str, Any]:
        """Compute enhanced BIFs with verification"""
        
        bif_results = {}
        
        for cluster_id, cluster_info in clusters['clusters'].items():
            cluster_bifs = {}
            
            # Get top parameters by attribution
            param_attributions = []
            for param_name in cluster_info['parameter_names']:
                lo, hi = self._get_param_range(param_name)
                total_attr = np.sum(np.abs(attributions[lo:hi]))
                param_attributions.append((param_name, total_attr))
            
            # Sort by attribution
            param_attributions.sort(key=lambda x: x[1], reverse=True)
            
            # Compute BIFs for top parameters
            top_params = param_attributions[:4]  # Top 4 parameters
            
            for param_name, attr_value in top_params:
                bif_val = self._compute_robust_bif(inputs, param_name)
                cluster_bifs[param_name] = {
                    'bif_value': bif_val,
                    'attribution': attr_value
                }
            
            bif_results[cluster_id] = cluster_bifs
        
        return bif_results

    def _compute_robust_bif(self, inputs: Dict[str, torch.Tensor], param_name: str) -> float:
        """Compute robust BIF for a parameter"""
        
        # Find parameter index
        param_idx = None
        for pinfo in self.param_info:
            if pinfo['name'] == param_name:
                param_idx = pinfo['idx_range'][0]
                break
        
        if param_idx is None:
            return 0.0
        
        base_flat = self._flatten_params(as_numpy=True)
        
        # Use adaptive epsilon based on parameter scale
        epsilon = 1e-3 * np.abs(base_flat[param_idx]) if np.abs(base_flat[param_idx]) > 0 else 1e-3
        
        # Compute finite difference
        pert_plus = base_flat.copy()
        pert_plus[param_idx] += epsilon
        
        pert_minus = base_flat.copy()
        pert_minus[param_idx] -= epsilon
        
        # Evaluate at three points
        self._assign_flat_to_model(torch.from_numpy(base_flat))
        with torch.no_grad():
            base_outputs = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
            base_loss = -base_outputs.logits[:, -1, :].max().item()
        
        self._assign_flat_to_model(torch.from_numpy(pert_plus))
        with torch.no_grad():
            plus_outputs = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
            plus_loss = -plus_outputs.logits[:, -1, :].max().item()
        
        self._assign_flat_to_model(torch.from_numpy(pert_minus))
        with torch.no_grad():
            minus_outputs = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
            minus_loss = -minus_outputs.logits[:, -1, :].max().item()
        
        # Restore original
        self._assign_flat_to_model(torch.from_numpy(base_flat))
        
        # Central difference for better accuracy
        bif = (plus_loss - minus_loss) / (2 * epsilon)
        
        logger.debug(f"BIF for {param_name}: {bif:.6f} (ε={epsilon:.2e})")
        
        return float(bif)

    def _save_results(self, results: Dict[str, Any], layer_idx: int):
        """Save results with enhanced metadata"""
        
        # Add system info
        results['system_info'] = {
            'device': str(self.device),
            'total_parameters': self._total_params,
            'timestamp': str(np.datetime64('now')),
            'python_version': sys.version,
            'torch_version': torch.__version__
        }
        
        # Save via experiment logger
        filename = f"enhanced_spd_layer_{layer_idx}_{int(np.datetime64('now').astype(int))}.json"
        self.experiment_logger.save_results(results, filename)
        
        logger.info(f"Results saved to {filename}")


# Legacy SPDAnalyzer for backward compatibility
class SPDAnalyzer:
    """Legacy SPD Analyzer - delegates to EnhancedSPDAnalyzer"""
    
    def __init__(self, mamba_analyzer, device: Optional[torch.device] = None):
        self.enhanced_analyzer = EnhancedSPDAnalyzer(mamba_analyzer, device)
        self.owner = mamba_analyzer
        self.model = mamba_analyzer.model
        self.tokenizer = mamba_analyzer.tokenizer
        self.experiment_logger = mamba_analyzer.experiment_logger
        self.config = getattr(mamba_analyzer, "config", None)
        self.device = self.enhanced_analyzer.device
        self.param_info = self.enhanced_analyzer.param_info
        self._total_params = self.enhanced_analyzer._total_params

    def run_layer_focused_spd(self, layer_idx: int = 0, reference_text: Optional[str] = None):
        """Run layer-focused SPD analysis using enhanced analyzer"""
        if reference_text is None:
            reference_text = "The quick brown fox jumps over the lazy dog."
        
        return self.enhanced_analyzer.run_enhanced_spd_analysis(
            layer_idx=layer_idx,
            reference_text=reference_text,
            n_attrib_steps=4,
            n_samples=10,
            n_clusters=6
        )

    def run_mamba2_separate_spd(self, layer_idx: int = 0, reference_text: str = "The quick brown fox jumps over the lazy dog.", n_samples: int = 50, sigma: float = 1e-3):
        """Run Mamba2 separate SPD analysis using enhanced analyzer"""
        return self.enhanced_analyzer.run_enhanced_spd_analysis(
            layer_idx=layer_idx,
            reference_text=reference_text,
            n_attrib_steps=4,
            n_samples=n_samples,
            n_clusters=6
        )

    def run_enhanced_mamba2_spd(self, layer_idx: int = 0, reference_text: str = "The quick brown fox jumps over the lazy dog.", n_samples: int = 50, sigma: float = 1e-3):
        """Run enhanced Mamba2 SPD analysis using enhanced analyzer"""
        return self.enhanced_analyzer.run_enhanced_spd_analysis(
            layer_idx=layer_idx,
            reference_text=reference_text,
            n_attrib_steps=4,
            n_samples=n_samples,
            n_clusters=6
        )

    def run(self, layer_idx: int = 0, reference_text: str = "The quick brown fox jumps over the lazy dog.", n_attrib_steps: int = 4, n_samples: int = 10, n_clusters: int = 6, sigma: float = 1e-3):
        """Run SPD analysis using enhanced analyzer (backward compatibility)"""
        return self.enhanced_analyzer.run_enhanced_spd_analysis(
            layer_idx=layer_idx,
            reference_text=reference_text,
            n_attrib_steps=n_attrib_steps,
            n_samples=n_samples,
            n_clusters=n_clusters
        )

    def fix_zero_attributions(self):
        """Diagnose and fix zero attribution issues"""
        
        print("=== Zero Attribution Diagnosis ===")
        
        # Check gradient flow
        test_inputs = self.tokenizer("Test", return_tensors='pt').to(self.device)
        
        # Enable all gradients
        for p in self.model.parameters():
            p.requires_grad_(True)
        
        # Test forward/backward
        outputs = self.model(**test_inputs)
        loss = outputs.logits.mean()
        loss.backward()
        
        # Analyze gradients
        zero_grads = 0
        total_params = 0
        
        for name, p in self.model.named_parameters():
            total_params += 1
            if p.grad is None or p.grad.abs().sum() < 1e-10:
                zero_grads += 1
                if zero_grads < 10:  # Show first 10
                    print(f"🚨 Zero grad: {name}")
        
        print(f"Parameters with zero gradients: {zero_grads}/{total_params}")
        
        # Suggested fixes
        if zero_grads > total_params * 0.5:
            print("\n=== Suggested Fixes ===")
            print("1. Use EnhancedSPDAnalyzer for robust gradient computation")
            print("2. Check if model is in evaluation mode (should be in train mode)")
            print("3. Verify that gradient checkpointing is disabled")
            print("4. Try different loss targets (last token, specific token, etc.)")
            print("5. Increase perturbation size for BIF computation")
        
        return zero_grads, total_params


def replace_layer_with_enhanced_mamba2(model, layer_idx: int, d_model: int = 768):
    """
    Replace a model layer's forward pass with enhanced Mamba2
    
    This creates a new Mamba2Layer and wraps the original layer's forward
    to route through the enhanced architecture.
    """
    logger.info(f"Replacing layer {layer_idx} with enhanced Mamba2...")
    
    # Create enhanced Mamba2 layer
    mamba2_layer = Mamba2Layer(d_model=d_model, n_gates=3, n_timescales=3, d_state=16)
    mamba2_layer = mamba2_layer.to(next(model.parameters()).device)
    mamba2_layer = mamba2_layer.to(next(model.parameters()).dtype)
    
    # Store original layer
    original_layer = model.backbone.layers[layer_idx]
    
    # Create wrapper that routes through Mamba2
    class Mamba2WrappedLayer(nn.Module):
        def __init__(self, original, mamba2, layer_idx):
            super().__init__()
            self.original = original
            self.mamba2 = mamba2
            self.layer_idx = layer_idx
            self.norm = original.norm  # Keep original norm
            
        def forward(self, hidden_states, residual=None, inference_params=None):
            # Route through enhanced Mamba2 instead of original mixer
            normed = self.norm(hidden_states)
            
            # Enhanced Mamba2 forward (handles 2D or 3D input)
            if normed.dim() == 2:
                normed = normed.unsqueeze(0)
                squeeze_output = True
            else:
                squeeze_output = False
            
            mamba2_out = self.mamba2(normed, self.layer_idx)
            
            if squeeze_output:
                mamba2_out = mamba2_out.squeeze(0)
            
            # Add residual
            if residual is not None:
                output = residual + mamba2_out
            else:
                output = hidden_states + mamba2_out
            
            return output
    
    # Replace the layer
    wrapped = Mamba2WrappedLayer(original_layer, mamba2_layer, layer_idx)
    model.backbone.layers[layer_idx] = wrapped
    
    logger.info(f"✅ Layer {layer_idx} now uses enhanced Mamba2")
    return mamba2_layer


# Usage example
def run_enhanced_spd_analysis(mamba_analyzer, layer_idx=0):
    """Run enhanced SPD analysis"""
    
    analyzer = EnhancedSPDAnalyzer(mamba_analyzer)
    
    try:
        results = analyzer.run_enhanced_spd_analysis(
            layer_idx=layer_idx,
            reference_text="The quick brown fox jumps over the lazy dog.",
            n_attrib_steps=4,
            n_samples=10,
            n_clusters=6
        )
        
        # Print summary
        print("\n=== Enhanced SPD Analysis Summary ===")
        print(f"Layer: {results['layer_idx']}")
        print(f"Attribution norm: {results['attributions_stats']['total_attribution_norm']:.6f}")
        print(f"Non-zero attributions: {results['attributions_stats']['non_zero_attributions']}")
        print(f"Zero ratio: {results['attributions_stats']['zero_ratio']:.3f}")
        print(f"Number of clusters: {results['clusters']['n_clusters']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Enhanced SPD analysis failed: {e}")
        raise