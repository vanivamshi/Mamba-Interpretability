"""
Attribution-based Parameter Decomposition (APD) extension
---------------------------------------------------------
Deterministic parameter-space interpretability module for MambaMechanisticAnalyzer.

Implements:
  - Parameter attribution (gradient × weight or integrated gradients)
  - Parameter subspace clustering
  - Causal weight ablation and effect measurement
  - Export of attribution heatmaps and top contributing tensors

Usage:
    from apd_extension import APDAnalyzer
    apd = APDAnalyzer(mamba_analyzer)
    results = apd.run(layer_idx=0, reference_text="The quick brown fox...")
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
from sklearn.cluster import KMeans
import logging
import json

logger = logging.getLogger(__name__)

class APDAnalyzer:
    """Attribution-based Parameter Decomposition (APD)"""

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
            
        self.param_info = self._collect_param_info()

    # --------------------------
    # Parameter utilities
    # --------------------------
    def _collect_param_info(self):
        info = []
        offset = 0
        for name, p in self.model.named_parameters():
            info.append({
                "name": name,
                "param": p,
                "shape": tuple(p.shape),
                "numel": p.numel(),
                "idx_range": (offset, offset + p.numel())
            })
            offset += p.numel()
        self._total_params = offset
        return info

    def _flatten_params(self, as_numpy=False):
        # Ensure all parameters are on the same device before concatenation
        flat_params = []
        for p in self.param_info:
            param = p["param"].detach().reshape(-1).to(self.device)
            flat_params.append(param)
        flat = torch.cat(flat_params)
        return flat.cpu().numpy() if as_numpy else flat

    def _assign_flat(self, flat: torch.Tensor):
        assert flat.numel() == self._total_params
        ptr = 0
        for pinfo in self.param_info:
            numel = pinfo["numel"]
            # Ensure the chunk is on the same device as the target parameter
            target_device = pinfo["param"].device
            chunk = flat[ptr:ptr+numel].view(pinfo["shape"]).to(target_device)
            with torch.no_grad():
                pinfo["param"].copy_(chunk)
            ptr += numel

    # --------------------------
    # Core attribution logic
    # --------------------------
    def _target_loss(self, inputs, token_idx=-1, logit_idx=None):
        """Return scalar loss (negative logit of target)"""
        # Ensure model is in training mode for gradient computation
        self.model.train()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Enable gradients for all parameters
        for p in self.model.parameters():
            p.requires_grad = True
            
        outputs = self.model(**inputs, output_hidden_states=False, return_dict=True)
        logits = outputs.logits
        token_logits = logits[:, token_idx, :]
        if logit_idx is None:
            top_idx = token_logits.argmax(dim=-1)
            selected = token_logits[0, top_idx[0]]
        else:
            selected = token_logits[0, logit_idx]
        return -selected  # negative for gradient ascent direction

    def compute_gradxparam(self, inputs):
        """Compute gradient × parameter attributions."""
        flat_params = self._flatten_params(as_numpy=False)
        for p in self.model.parameters():
            p.requires_grad = True
        self.model.zero_grad()

        loss = self._target_loss(inputs)
        loss.backward()

        grads = torch.cat([
            p.grad.reshape(-1).to(self.device) if p.grad is not None
            else torch.zeros(p.numel(), device=self.device)
            for p in self.model.parameters()
        ])

        with torch.no_grad():
            gxp = grads * flat_params
        return gxp.cpu().numpy()

    def compute_integrated_gradients(self, inputs, steps=16):
        """Integrated gradients w.r.t. parameters."""
        flat_params = self._flatten_params(as_numpy=False)
        baseline = torch.zeros_like(flat_params)
        alphas = torch.linspace(0, 1, steps=steps, device=self.device)
        total_grad = torch.zeros_like(flat_params)
        orig_flat = flat_params.clone()

        for alpha in alphas:
            interp_flat = baseline + alpha * (orig_flat - baseline)
            self._assign_flat(interp_flat)
            self.model.zero_grad()
            loss = self._target_loss(inputs)
            loss.backward()
            grads = torch.cat([
                p.grad.reshape(-1).to(self.device) if p.grad is not None
                else torch.zeros(p.numel(), device=self.device)
                for p in self.model.parameters()
            ])
            total_grad += grads
            for p in self.model.parameters():
                p.grad = None

        self._assign_flat(orig_flat)
        avg_grad = total_grad / float(steps)
        ig = (orig_flat - baseline) * avg_grad
        return ig.cpu().numpy()

    # --------------------------
    # Analysis
    # --------------------------
    def cluster_attributions(self, attributions: np.ndarray, n_clusters=10):
        """Cluster parameter-level attributions into groups."""
        attr = np.abs(attributions)
        attr = attr / (attr.max() + 1e-8)
        sample_idx = np.linspace(0, len(attr)-1, num=min(2000, len(attr)), dtype=int)
        data = attr[sample_idx].reshape(-1, 1)
        clustering = KMeans(n_clusters=n_clusters, n_init=5)
        labels = clustering.fit_predict(data)

        clusters = {i: [] for i in range(n_clusters)}
        for idx, pinfo in enumerate(self.param_info):
            lo, hi = pinfo["idx_range"]
            weight = attr[lo:hi].mean()
            cluster_id = labels[idx % len(labels)]
            clusters[cluster_id].append({"name": pinfo["name"], "mean_attr": float(weight)})
        return clusters

    def ablation_test(self, inputs, attributions, top_frac=0.05):
        """Ablate top fraction of parameters by attribution magnitude and measure effect."""
        base_flat = self._flatten_params(as_numpy=True)
        n = len(base_flat)
        k = int(top_frac * n)
        top_idx = np.argsort(-np.abs(attributions))[:k]

        perturbed = base_flat.copy()
        perturbed[top_idx] = 0.0  # zero out influential weights

        self._assign_flat(torch.from_numpy(perturbed))
        new_loss = float(self._target_loss(inputs).item())
        self._assign_flat(torch.from_numpy(base_flat))  # restore

        return {"top_frac": top_frac, "ablation_effect": new_loss}

    # --------------------------
    # Orchestration
    # --------------------------
    def run(self, layer_idx=0, reference_text=None, method="gradxparam", n_clusters=10):
        """Run the full APD pipeline."""
        if reference_text is None:
            reference_text = "The quick brown fox jumps over the lazy dog."
        inputs = self.tokenizer(reference_text, return_tensors="pt", truncation=True, max_length=128)

        logger.info(f"Running APD ({method}) on layer {layer_idx}...")

        if method == "gradxparam":
            attributions = self.compute_gradxparam(inputs)
        else:
            attributions = self.compute_integrated_gradients(inputs)

        clusters = self.cluster_attributions(attributions, n_clusters=n_clusters)
        ablation = self.ablation_test(inputs, attributions, top_frac=0.05)

        results = {
            "layer_idx": layer_idx,
            "method": method,
            "attr_norm": float(np.linalg.norm(attributions)),
            "clusters": clusters,
            "ablation": ablation,
        }

        fname = f"apd_results_layer_{layer_idx}_{method}.json"
        self.experiment_logger.save_results(results, fname)
        logger.info(f"APD results saved to {fname}")
        return results
