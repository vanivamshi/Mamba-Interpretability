"""
Stochastic Parameter Decomposition (SPD) extension
Integrates with MambaMechanisticAnalyzer to provide APD/SPD-style
parameter-space decomposition, attribution, stochastic sampling,
susceptibility / BIF approximations, clustering into functional
parameter subspaces, and visualization.

Save as: spd_extension.py

Usage (high-level):
    from spd_extension import SPDAnalyzer
    spd = SPDAnalyzer(mamba_analyzer)
    results = spd.run(layer_idx=0, target='logit', n_samples=128, steps=20)

This file intentionally avoids heavy external dependencies; it uses
PyTorch and sklearn (for clustering) which are already present in the
main repo. For very large models, some routines (inverse-Hessian
approximation) are approximate and configurable.

Notes:
- The implementation aims to be practical and safe to run within the
  existing experimental framework; computational shortcuts are used
  (e.g. mini-batch attributions, diagonal Fisher approximations) but
  hooks are included for more advanced exact computations.
- Outputs are saved via the parent's ExperimentLogger to keep
  consistency with the rest of the framework.

"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
import json
import logging
import psutil
import gc

logger = logging.getLogger(__name__)


class SPDAnalyzer:
    """Stochastic Parameter Decomposition analyzer.

    Args:
        mamba_analyzer: instance of MambaMechanisticAnalyzer (owner)
        device: device string or torch.device; will default to owner's device
    """

    def __init__(self, mamba_analyzer, device: Optional[torch.device] = None):
        self.owner = mamba_analyzer
        self.model = mamba_analyzer.model
        self.tokenizer = mamba_analyzer.tokenizer
        self.experiment_logger = mamba_analyzer.experiment_logger
        self.config = getattr(mamba_analyzer, "config", None)
        
        # Validate that required components are available
        if self.model is None:
            raise ValueError("MambaMechanisticAnalyzer.model is None. Please ensure the model is properly loaded.")
        if self.tokenizer is None:
            raise ValueError("MambaMechanisticAnalyzer.tokenizer is None. Please ensure the tokenizer is properly loaded.")
        if self.experiment_logger is None:
            raise ValueError("MambaMechanisticAnalyzer.experiment_logger is None. Please ensure the experiment logger is properly initialized.")
        
        # Set device
        if device is not None:
            self.device = device
        elif self.config is not None and hasattr(self.config, 'device'):
            self.device = self.config.device
        else:
            self.device = next(self.model.parameters()).device

        # Cached parameter groups and shapes
        self.param_info = self._collect_param_info()
        
        # Memory management settings
        self.max_memory_gb = 31.0  # Maximum memory usage in GB

    def _collect_param_info(self) -> List[Dict[str, Any]]:
        """Collect info for each parameter tensor in the model.

        We return a list of dicts: {"name","param","shape","numel","idx_range"}
        idx_range indexes into a flattened global parameter vector for convenience.
        """
        info = []
        offset = 0
        for name, p in self.model.named_parameters():
            numel = p.numel()
            info.append({
                "name": name,
                "param": p,
                "shape": tuple(p.shape),
                "numel": numel,
                "idx_range": (offset, offset + numel)
            })
            offset += numel
        self._total_params = offset
        logger.info(f"Collected {len(info)} parameter tensors, total params={self._total_params}")
        return info

    def _check_memory_usage(self, operation_name: str, estimated_memory_gb: float, verbose: bool = True) -> bool:
        """Check if estimated memory usage is within safe limits."""
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        if verbose:
            logger.info(f"Memory check for {operation_name}:")
            logger.info(f"  Available memory: {available_memory_gb:.2f} GB")
            logger.info(f"  Estimated usage: {estimated_memory_gb:.2f} GB")
            logger.info(f"  Max allowed: {self.max_memory_gb:.2f} GB")
        
        if estimated_memory_gb > self.max_memory_gb:
            logger.warning(f"Estimated memory usage ({estimated_memory_gb:.2f} GB) exceeds limit ({self.max_memory_gb:.2f} GB)")
            return False
        
        if estimated_memory_gb > available_memory_gb * 0.8:  # Use max 80% of available memory
            logger.warning(f"Estimated memory usage ({estimated_memory_gb:.2f} GB) would use >80% of available memory")
            return False
            
        return True

    def _ensure_activations_for_layer(self, layer_idx: int, sample_texts: Optional[List[str]] = None):
        """
        Ensure activation_data has activations for layer_idx.
        If missing, collect activations on a small set of sample_texts (or defaults).
        Returns: activations tensor (or None on failure).
        """
        # Access activation data through the owner (MambaMechanisticAnalyzer)
        if hasattr(self.owner, 'activation_data') and layer_idx in self.owner.activation_data:
            return self.owner.activation_data[layer_idx]

        logger.warning(f"Activations for layer {layer_idx} not found. Attempting to collect a small sample now...")
        if sample_texts is None:
            sample_texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Artificial intelligence is transforming industries worldwide.",
                "Machine learning models require large amounts of training data."
            ]
        try:
            # Use the owner's collect_activations method
            collected = self.owner.collect_activations(sample_texts, layer_indices=[layer_idx])
            if layer_idx in collected:
                return collected[layer_idx]
            else:
                logger.error(f"Activation collection returned no activations for layer {layer_idx}.")
                return None
        except Exception as e:
            logger.error(f"Failed to collect activations for layer {layer_idx}: {e}")
            return None

    def _flatten_params(self, as_numpy: bool = False) -> torch.Tensor:
        """Return flattened parameter vector (torch tensor on device)."""
        parts = [p['param'].detach().reshape(-1) for p in self.param_info]
        flat = torch.cat(parts).to(self.device)
        return flat.cpu().numpy() if as_numpy else flat

    def _assign_flat_to_model(self, flat: torch.Tensor):
        """Assign values from a flattened parameter tensor back into model parameters (in-place)."""
        assert flat.numel() == self._total_params
        ptr = 0
        for pinfo in self.param_info:
            numel = pinfo['numel']
            chunk = flat[ptr:ptr+numel].view(pinfo['shape']).to(pinfo['param'].device)
            with torch.no_grad():
                pinfo['param'].copy_(chunk)
            ptr += numel

    def _target_loss(self, inputs: Dict[str, torch.Tensor], target_token_idx: Optional[int] = -1, target_logit_idx: Optional[int] = None):
        """Default target: final-token logit (optionally class idx). Returns scalar loss.

        - target_token_idx: which token to inspect (-1 last token default)
        - target_logit_idx: if None, use max logit; otherwise index in vocab/logit dims
        """
        # Ensure model is in training mode for gradient computation
        self.model.train()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Enable gradients for all parameters
        for p in self.model.parameters():
            p.requires_grad = True
            
        outputs = self.model(**inputs, output_hidden_states=False, return_dict=True)
        logits = outputs.logits  # [B, T, V]
        token_logits = logits[:, target_token_idx, :]
        if target_logit_idx is None:
            # choose argmax across vocab
            top_idx = token_logits.argmax(dim=-1)
            # Negative logit as "loss" (we maximize logit when computing attribution sign)
            sel_logits = token_logits[0, top_idx[0]]
        else:
            sel_logits = token_logits[0, target_logit_idx]
        # We return negative so gradients point towards increasing that logit when minimized
        return -sel_logits

    def compute_gradient_attributions(self, inputs: Dict[str, torch.Tensor], n_steps: int = 1, baseline_params: Optional[torch.Tensor] = None, target_token_idx: int = -1, target_logit_idx: Optional[int] = None, param_mask: Optional[torch.Tensor] = None) -> np.ndarray:
        """Compute (integrated) gradients of target loss w.r.t. flattened parameters.

        - If n_steps==1: single-step gradient
        - If n_steps>1: simple Riemann-sum integrated gradients along straight-line path from baseline to current params

        Returns: numpy array of gradients (shape: total_params)
        """
        # Use CPU for large parameter tensors to avoid CUDA OOM
        flat_params = self._flatten_params(as_numpy=True)  # Get as numpy first
        flat_params_tensor = torch.from_numpy(flat_params).float()
        
        if baseline_params is None:
            baseline = torch.zeros_like(flat_params_tensor)  # Keep on CPU
        else:
            baseline = baseline_params.cpu().float()

        total_grad = torch.zeros_like(flat_params_tensor)  # Keep on CPU
        alphas = torch.linspace(0.0, 1.0, steps=n_steps)  # Keep on CPU

        orig_flat = flat_params_tensor.clone()

        for alpha in alphas:
            cur_flat = baseline + alpha * (orig_flat - baseline)
            # write into model (move to GPU only when assigning)
            self._assign_flat_to_model(cur_flat.to(self.device))

            # compute gradient
            for p in self.model.parameters():
                p.requires_grad = True
            self.model.zero_grad()

            loss = self._target_loss(inputs, target_token_idx=target_token_idx, target_logit_idx=target_logit_idx)
            loss.backward()

            # Collect gradients and move to CPU immediately
            grads = []
            for p in self.model.parameters():
                if p.grad is not None:
                    grads.append(p.grad.reshape(-1).cpu())
                else:
                    grads.append(torch.zeros(p.numel()))
            grads_cpu = torch.cat(grads)
            total_grad += grads_cpu

            # free gradients
            for p in self.model.parameters():
                p.grad = None
            
            # Clear CUDA cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # restore original params
        self._assign_flat_to_model(orig_flat.to(self.device))

        # Average and scale by (orig - baseline)
        avg_grad = total_grad / float(n_steps)
        ig = (orig_flat - baseline) * avg_grad

        ig_np = ig.numpy()
        if param_mask is not None:
            mask_np = param_mask.cpu().numpy()
            ig_np = ig_np * mask_np
        return ig_np

    def sample_stochastic_perturbations(self, n_samples: int = 128, sigma: float = 1e-3, group_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Sample n_samples perturbations in flattened parameter space.

        If group_mask is provided (shape [total_params]), perturb only in masked dims.
        Returns array shape [n_samples, total_params].
        """
        base = self._flatten_params(as_numpy=True)
        rng = np.random.default_rng()
        if group_mask is None:
            noise = rng.normal(scale=sigma, size=(n_samples, base.size)).astype(np.float32)
        else:
            noise = rng.normal(scale=sigma, size=(n_samples, base.size)).astype(np.float32) * group_mask
        samples = base[None, :].astype(np.float32) + noise
        return samples

    def estimate_susceptibility_matrix(self, inputs: Dict[str, torch.Tensor], param_subspace_mask: np.ndarray, n_samples: int = 128, sigma: float = 1e-3, batch: int = 8, verbose_memory_check: bool = True) -> Dict[str, Any]:
        """Estimate how perturbations in a parameter subspace affect a scalar target (empirical susceptibility).

        Returns dictionary containing: mean_effect, cov_effect, samples_effects, perturbation_norms
        """
        # Check memory usage before proceeding
        estimated_memory_gb = (n_samples * self._total_params * 4) / (1024**3)  # 4 bytes per float32
        if not self._check_memory_usage("susceptibility estimation", estimated_memory_gb, verbose=verbose_memory_check):
            # Reduce sample count if memory usage is too high
            n_samples = min(n_samples, int(self.max_memory_gb * (1024**3) / (self._total_params * 4)))
            logger.warning(f"Reducing sample count to {n_samples} due to memory constraints")
        
        # Use smaller batch size for very large models
        if self._total_params > 50_000_000:  # 50M parameters
            batch = min(batch, 4)
            if verbose_memory_check:  # Only log once
                logger.info(f"Using smaller batch size ({batch}) for large model")
        
        effects = []
        norms = []
        base_flat = self._flatten_params(as_numpy=True)

        # Process samples in smaller batches to avoid memory issues
        for i in range(0, n_samples, batch):
            batch_samples = []
            for j in range(i, min(i + batch, n_samples)):
                # Generate single sample instead of storing all samples
                sample = self._generate_single_perturbation(base_flat, sigma, param_subspace_mask)
                batch_samples.append(sample)
            
            for s in batch_samples:
                flat_tensor = torch.from_numpy(s).to(self.device)
                self._assign_flat_to_model(flat_tensor)
                # Evaluate target logit
                loss = self._target_loss(inputs)
                # we use negative loss value as effect magnitude (higher logit -> lower loss)
                effects.append(-float(loss.item()))
                norms.append(np.linalg.norm((s - base_flat)))
            
            # Clear CUDA cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # restore original params
        self._assign_flat_to_model(torch.from_numpy(base_flat))

        effects = np.array(effects)
        norms = np.array(norms)

        return {
            'mean_effect': float(effects.mean()),
            'cov_effect': float(np.var(effects)),
            'samples_effects': effects,
            'perturbation_norms': norms
        }
    
    def _generate_single_perturbation(self, base: np.ndarray, sigma: float, group_mask: np.ndarray) -> np.ndarray:
        """Generate a single perturbation sample without storing all samples."""
        rng = np.random.default_rng()
        if group_mask is None:
            noise = rng.normal(scale=sigma, size=base.size).astype(np.float32)
        else:
            noise = rng.normal(scale=sigma, size=base.size).astype(np.float32) * group_mask
        return base.astype(np.float32) + noise

    def approximate_bif(self, inputs: Dict[str, torch.Tensor], param_idx: int, epsilon: float = 1e-3) -> float:
        """Approximate a Bayesian Influence Function-like quantity for a small perturbation on
        parameter coordinate `param_idx` by finite-difference on the target logit.

        This is a local, cheap approximation and should be used as a heuristic.
        """
        base_flat = self._flatten_params(as_numpy=True)
        pert = base_flat.copy()
        pert[param_idx] += epsilon
        # eval base
        self._assign_flat_to_model(torch.from_numpy(base_flat))
        base_loss = float(self._target_loss(inputs).item())
        self._assign_flat_to_model(torch.from_numpy(pert))
        pert_loss = float(self._target_loss(inputs).item())
        self._assign_flat_to_model(torch.from_numpy(base_flat))
        # BIF ~ d(loss)/d(param) approximated by finite difference
        return (pert_loss - base_loss) / epsilon

    def cluster_parameter_subspaces(self, attributions: np.ndarray, n_clusters: int = 12, method: str = 'agglomerative') -> Dict[str, Any]:
        """Cluster flattened-parameter attributions into groups.

        attributions: (total_params,) or (n_examples, total_params)
        Returns: mapping cluster_id -> list of (param_tensor_name, local_indices)
        """
        # If per-example matrix, average first
        if attributions.ndim == 2:
            attr = attributions.mean(axis=0)
        else:
            attr = attributions

        # Memory-efficient clustering approach for large parameter spaces
        # Instead of creating massive pseudo-sample arrays, we'll cluster parameter tensors directly
        # based on their attribution statistics
        
        logger.info(f"SPD: Clustering {len(self.param_info)} parameter tensors into {n_clusters} clusters")
        
        # Extract features for each parameter tensor
        tensor_features = []
        tensor_names = []
        
        for pinfo in self.param_info:
            lo, hi = pinfo['idx_range']
            segment = attr[lo:hi]
            
            # Compute statistical features for this parameter tensor
            features = np.array([
                np.mean(segment),           # Mean attribution
                np.std(segment),            # Standard deviation
                np.max(np.abs(segment)),    # Max absolute attribution
                np.sum(np.abs(segment)),    # Total attribution mass
                np.percentile(segment, 25), # 25th percentile
                np.percentile(segment, 75), # 75th percentile
                np.sum(segment > 0),        # Count of positive attributions
                np.sum(segment < 0),        # Count of negative attributions
            ])
            
            tensor_features.append(features)
            tensor_names.append(pinfo['name'])
        
        tensor_features = np.array(tensor_features)
        
        # Normalize features for clustering
        feature_means = np.mean(tensor_features, axis=0)
        feature_stds = np.std(tensor_features, axis=0)
        feature_stds[feature_stds == 0] = 1.0  # Avoid division by zero
        normalized_features = (tensor_features - feature_means) / feature_stds
        
        # Use KMeans for clustering (more memory efficient than agglomerative)
        logger.info("SPD: Using KMeans for parameter tensor clustering...")
        clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        logger.info("SPD: Starting clustering...")
        labels = clustering.fit_predict(normalized_features)
        logger.info("SPD: Clustering complete")

        # Map cluster labels back to parameter tensor names
        clusters = {i: [] for i in range(n_clusters)}
        for i, cluster_id in enumerate(labels):
            clusters[cluster_id].append(tensor_names[i])

        return {
            'clusters': clusters,
            'n_clusters': n_clusters
        }

    def summarize_and_save(self, results: Dict[str, Any], layer_idx: int = 0, tag: str = 'spd') -> None:
        """Save results through experiment logger in canonical filename."""
        fname = f"spd_results_layer_{layer_idx}_{tag}.json"
        self.experiment_logger.save_results(results, fname)
        logger.info(f"SPD results saved to {fname}")

    def run(self, layer_idx: int = 0, reference_text: Optional[str] = None, n_attrib_steps: int = 8, n_samples: int = 256, sigma: float = 1e-3, n_clusters: int = 12) -> Dict[str, Any]:
        """High-level orchestration for SPD analysis for a given layer.

        Steps performed:
          1. prepare a reference input
          2. compute integrated-gradient style attributions over parameters
          3. cluster attributed parameters into candidate subspaces
          4. estimate susceptibilities by stochastic sampling per cluster
          5. compute approximate BIFs for top coordinates in each cluster
          6. save and return results
        """
        # Adjust parameters for large models to prevent memory issues
        if self._total_params > 100_000_000:  # 100M+ parameters
            logger.info("Large model detected, using conservative memory settings")
            n_samples = min(n_samples, 64)  # Reduce sample count
            n_clusters = min(n_clusters, 8)  # Reduce cluster count
            n_attrib_steps = min(n_attrib_steps, 2)  # Reduce attribution steps further
        
        # Prepare reference input
        if reference_text is None:
            reference_text = "The quick brown fox jumps over the lazy dog."
        inputs = self.tokenizer(reference_text, return_tensors='pt', truncation=True, max_length=128)

        # 1) compute attributions
        logger.info("SPD: computing gradient-based attributions...")
        attributions = self.compute_gradient_attributions(inputs, n_steps=n_attrib_steps)

        # 2) cluster
        logger.info("SPD: clustering parameter attributions to subspaces...")
        clusters = self.cluster_parameter_subspaces(attributions, n_clusters=n_clusters)

        # 3) for each cluster, compute susceptibility
        cluster_sus = {}
        base_flat = self._flatten_params(as_numpy=True)
        cluster_count = 0
        for cid, names in clusters['clusters'].items():
            # create mask for params in cluster
            mask = np.zeros(base_flat.shape, dtype=np.float32)
            for pinfo in self.param_info:
                if pinfo['name'] in names:
                    lo, hi = pinfo['idx_range']
                    mask[lo:hi] = 1.0
            if mask.sum() == 0:
                continue
            
            # Only show verbose memory checks for the first cluster
            verbose_check = (cluster_count == 0)
            logger.info(f"Processing cluster {cid} ({len(names)} parameter tensors)...")
            
            sus = self.estimate_susceptibility_matrix(inputs, mask, n_samples=min(n_samples, 128), sigma=sigma, verbose_memory_check=verbose_check)
            # compute top BIFs within cluster (top-k coords by |attr|)
            lo_idxs = np.where(mask > 0)[0]
            cluster_attr = np.abs(attributions[lo_idxs])
            topk = min(8, lo_idxs.size)
            top_coords = lo_idxs[np.argsort(-cluster_attr)[:topk]]
            bif_vals = {int(intc): float(self.approximate_bif(inputs, int(intc), epsilon=sigma)) for intc in top_coords}
            cluster_sus[cid] = {
                'names': names,
                'susceptibility': sus,
                'top_bifs': bif_vals
            }
            cluster_count += 1

        results = {
            'layer_idx': layer_idx,
            'attributions_norm': float(np.linalg.norm(attributions)),
            'clusters': clusters,
            'cluster_susceptibilities': cluster_sus
        }

        self.summarize_and_save(results, layer_idx=layer_idx)
        return results


# End of spd_extension.py
