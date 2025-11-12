"""
Sparse Autoencoder (SAE) Implementation for Mamba Mechanistic Interpretability

This module implements Sparse Autoencoders to find interpretable latent features
in Mamba model activations, following the methodology outlined in the research framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
import logging
import json

logger = logging.getLogger(__name__)

class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for discovering interpretable features in Mamba activations.
    
    Architecture:
    - Encoder: h -> z (sparse latent representation)
    - Decoder: z -> ĥ (reconstruction)
    - Loss: MSE(h, ĥ) + λ₁||z||₁ (sparsity penalty)
    """
    
    def __init__(self, input_dim: int, latent_dim: int, sparsity_weight: float = 1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.sparsity_weight = sparsity_weight
        
        # Encoder: input -> latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU()
        )
        
        # Decoder: latent -> reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, input_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input activations [batch_size, input_dim]
            
        Returns:
            z: Sparse latent representation [batch_size, latent_dim]
            x_recon: Reconstructed input [batch_size, input_dim]
        """
        # Encode to sparse representation
        z = self.encoder(x)
        
        # Decode to reconstruction
        x_recon = self.decoder(z)
        
        return z, x_recon
    
    def compute_loss(self, x: torch.Tensor, z: torch.Tensor, x_recon: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute SAE loss with sparsity penalty.
        
        Args:
            x: Original input
            z: Sparse latent representation
            x_recon: Reconstructed input
            
        Returns:
            Dictionary containing loss components
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x)
        
        # Sparsity loss (L1 penalty)
        sparsity_loss = torch.mean(torch.abs(z))
        
        # Total loss
        total_loss = recon_loss + self.sparsity_weight * sparsity_loss
        
        # Compute sparsity metrics
        sparsity_rate = torch.mean((torch.abs(z) < 1e-6).float())
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'sparsity_loss': sparsity_loss,
            'sparsity_rate': sparsity_rate
        }

class SAETrainer:
    """Trainer for Sparse Autoencoders."""
    
    def __init__(self, model: SparseAutoencoder, device: str = "cuda"):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.sparsity_rates = []
        
    def train(self, 
              train_data: torch.Tensor,
              val_data: Optional[torch.Tensor] = None,
              num_epochs: int = 100,
              batch_size: int = 256,
              learning_rate: float = 1e-3,
              verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the SAE model.
        
        Args:
            train_data: Training activations [num_samples, input_dim]
            val_data: Validation activations [num_val_samples, input_dim]
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            verbose: Whether to print training progress
            
        Returns:
            Training history dictionary
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(train_data)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if val_data is not None:
            val_dataset = torch.utils.data.TensorDataset(val_data)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        for epoch in range(num_epochs):
            # Training
            epoch_train_losses = []
            epoch_sparsity_rates = []
            
            for batch_data, in train_loader:
                batch_data = batch_data.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                z, x_recon = self.model(batch_data)
                loss_dict = self.model.compute_loss(batch_data, z, x_recon)
                
                # Backward pass
                loss_dict['total_loss'].backward()
                optimizer.step()
                
                epoch_train_losses.append(loss_dict['total_loss'].item())
                epoch_sparsity_rates.append(loss_dict['sparsity_rate'].item())
            
            # Validation
            epoch_val_losses = []
            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    for batch_data, in val_loader:
                        batch_data = batch_data.to(self.device)
                        z, x_recon = self.model(batch_data)
                        loss_dict = self.model.compute_loss(batch_data, z, x_recon)
                        epoch_val_losses.append(loss_dict['total_loss'].item())
                self.model.train()
            
            # Record metrics
            avg_train_loss = np.mean(epoch_train_losses)
            avg_sparsity_rate = np.mean(epoch_sparsity_rates)
            avg_val_loss = np.mean(epoch_val_losses) if epoch_val_losses else None
            
            self.train_losses.append(avg_train_loss)
            self.sparsity_rates.append(avg_sparsity_rate)
            if avg_val_loss is not None:
                self.val_losses.append(avg_val_loss)
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                val_str = f", Val Loss: {avg_val_loss:.6f}" if avg_val_loss is not None else ""
                logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                          f"Train Loss: {avg_train_loss:.6f}{val_str}, "
                          f"Sparsity Rate: {avg_sparsity_rate:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'sparsity_rates': self.sparsity_rates
        }

class SAEAnalyzer:
    """Analyzer for SAE results and interpretability."""
    
    def __init__(self, sae_model: SparseAutoencoder):
        self.sae_model = sae_model
        self.feature_correlations = {}
        self.top_features = {}
    
    def analyze_feature_correlations(self, 
                                  activations: torch.Tensor,
                                  task_labels: Optional[torch.Tensor] = None,
                                  task_name: str = "task") -> Dict[str, Any]:
        """
        Analyze correlations between SAE features and task variables.
        
        Args:
            activations: Input activations [num_samples, input_dim]
            task_labels: Task labels/variables [num_samples, num_variables]
            task_name: Name of the task for labeling
            
        Returns:
            Dictionary containing correlation analysis results
        """
        self.sae_model.eval()
        
        with torch.no_grad():
            # Get sparse features
            z, _ = self.sae_model(activations)
            z_np = z.cpu().numpy()
        
        correlations = {}
        
        if task_labels is not None:
            task_labels_np = task_labels.cpu().numpy() if isinstance(task_labels, torch.Tensor) else task_labels
            
            # Ensure task_labels_np is 2D
            if task_labels_np.ndim == 1:
                task_labels_np = task_labels_np.reshape(-1, 1)
            
            # Compute correlations for each task variable
            for var_idx in range(task_labels_np.shape[1]):
                var_name = f"{task_name}_var_{var_idx}"
                var_correlations = []
                
                for feat_idx in range(z_np.shape[1]):
                    # Compute correlation between feature and task variable
                    correlation = np.corrcoef(z_np[:, feat_idx], task_labels_np[:, var_idx])[0, 1]
                    var_correlations.append(abs(correlation))  # Use absolute correlation
                
                correlations[var_name] = var_correlations
                
                # Find top features for this variable
                top_feature_indices = np.argsort(var_correlations)[-10:][::-1]
                self.top_features[var_name] = [
                    (idx, var_correlations[idx]) for idx in top_feature_indices
                ]
        
        self.feature_correlations[task_name] = correlations
        
        return {
            'correlations': correlations,
            'top_features': self.top_features,
            'mean_correlation': {var: np.mean(corrs) for var, corrs in correlations.items()},
            'max_correlation': {var: np.max(corrs) for var, corrs in correlations.items()}
        }
    
    def visualize_features(self, 
                          activations: torch.Tensor,
                          task_labels: Optional[torch.Tensor] = None,
                          save_path: Optional[str] = None) -> None:
        """
        Visualize SAE features and their correlations.
        
        Args:
            activations: Input activations
            task_labels: Task labels for correlation analysis
            save_path: Path to save visualization
        """
        self.sae_model.eval()
        
        with torch.no_grad():
            z, x_recon = self.sae_model(activations)
            z_np = z.cpu().numpy()
            x_np = activations.cpu().numpy()
            x_recon_np = x_recon.cpu().numpy()
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('SAE Feature Analysis', fontsize=16)
        
        # Plot 1: Feature activations heatmap
        im1 = axes[0, 0].imshow(z_np[:100, :].T, cmap='viridis', aspect='auto')
        axes[0, 0].set_title('SAE Feature Activations (First 100 Samples)')
        axes[0, 0].set_xlabel('Sample Index')
        axes[0, 0].set_ylabel('Feature Index')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Plot 2: Sparsity histogram
        feature_norms = np.linalg.norm(z_np, axis=0)
        axes[0, 1].hist(feature_norms, bins=50, alpha=0.7)
        axes[0, 1].set_title('Feature Activation Norms')
        axes[0, 1].set_xlabel('L2 Norm')
        axes[0, 1].set_ylabel('Count')
        
        # Plot 3: Reconstruction quality
        reconstruction_error = np.mean((x_np - x_recon_np) ** 2, axis=1)
        axes[0, 2].hist(reconstruction_error, bins=50, alpha=0.7)
        axes[0, 2].set_title('Reconstruction Error Distribution')
        axes[0, 2].set_xlabel('MSE')
        axes[0, 2].set_ylabel('Count')
        
        # Plot 4: Feature sparsity over time
        sparsity_per_sample = np.mean(np.abs(z_np) < 1e-6, axis=1)
        axes[1, 0].plot(sparsity_per_sample)
        axes[1, 0].set_title('Sparsity Rate Over Samples')
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].set_ylabel('Sparsity Rate')
        
        # Plot 5: Top features by activation
        top_features_by_activation = np.argsort(np.mean(np.abs(z_np), axis=0))[-20:][::-1]
        top_activations = np.mean(np.abs(z_np), axis=0)[top_features_by_activation]
        axes[1, 1].bar(range(len(top_features_by_activation)), top_activations)
        axes[1, 1].set_title('Top 20 Features by Mean Activation')
        axes[1, 1].set_xlabel('Feature Rank')
        axes[1, 1].set_ylabel('Mean Absolute Activation')
        
        # Plot 6: Correlation with task (if available)
        if task_labels is not None:
            task_labels_np = task_labels.cpu().numpy() if isinstance(task_labels, torch.Tensor) else task_labels
            if task_labels_np.ndim == 1:
                correlations = [np.corrcoef(z_np[:, i], task_labels_np)[0, 1] for i in range(z_np.shape[1])]
                correlations = np.abs(correlations)  # Use absolute correlation
                
                axes[1, 2].hist(correlations, bins=50, alpha=0.7)
                axes[1, 2].set_title('Feature-Task Correlations')
                axes[1, 2].set_xlabel('Absolute Correlation')
                axes[1, 2].set_ylabel('Count')
            else:
                axes[1, 2].text(0.5, 0.5, 'Multi-dimensional\nTask Labels', 
                               ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('Task Correlation Analysis')
        else:
            axes[1, 2].text(0.5, 0.5, 'No Task Labels\nProvided', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Task Correlation Analysis')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SAE visualization saved to {save_path}")
        
        plt.show()
    
    def get_interpretable_features(self, 
                                 correlation_threshold: float = 0.6,
                                 min_activation: float = 0.1) -> Dict[str, List[int]]:
        """
        Get features that are both interpretable and active.
        
        Args:
            correlation_threshold: Minimum correlation with task variables
            min_activation: Minimum mean activation level
            
        Returns:
            Dictionary mapping task variables to interpretable feature indices
        """
        interpretable_features = {}
        
        for task_name, correlations in self.feature_correlations.items():
            interpretable_features[task_name] = []
            
            for var_name, corrs in correlations.items():
                # Find features that meet both criteria
                strong_correlations = np.array(corrs) > correlation_threshold
                
                # Get mean activations for these features
                if hasattr(self, 'mean_activations'):
                    high_activations = self.mean_activations > min_activation
                    interpretable_indices = np.where(strong_correlations & high_activations)[0]
                else:
                    interpretable_indices = np.where(strong_correlations)[0]
                
                interpretable_features[task_name].extend(interpretable_indices.tolist())
        
        return interpretable_features

class SparseProbingEncoder:
    """
    Sparse Probing Encoder using Lasso/ElasticNet for causal dimension discovery.
    
    This implements the hypothesis probes step of the experimental framework.
    """
    
    def __init__(self, alpha: float = 1e-3, l1_ratio: float = 0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Fit the sparse probing encoder.
        
        Args:
            X: Activation data [num_samples, num_features]
            y: Target labels [num_samples]
            
        Returns:
            Dictionary containing fitting results
        """
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X_scaled, y)
        
        # Get results
        coefficients = self.model.coef_
        nonzero_indices = np.where(np.abs(coefficients) > 1e-6)[0]
        
        # Compute metrics
        y_pred = self.model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        
        self.is_fitted = True
        
        return {
            'coefficients': coefficients,
            'nonzero_indices': nonzero_indices,
            'num_nonzero': len(nonzero_indices),
            'r2_score': r2,
            'sparsity_rate': 1.0 - (len(nonzero_indices) / len(coefficients))
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance (absolute coefficients)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        return np.abs(self.model.coef_)

class CircuitTester:
    """Tests causal role of circuits via activation patching"""
    
    def __init__(self, model, layer_idx: int, circuit_indices: List[int]):
        self.model = model
        self.layer_idx = layer_idx
        self.circuit_indices = circuit_indices
        self.hooks = []
        self.stored_activations = {}
    
    def _call_model(self, inputs: torch.Tensor):
        """Robust model calling that handles different input formats"""
        logger.info(f"🔍 _call_model: Input type: {type(inputs)}, shape: {inputs.shape if hasattr(inputs, 'shape') else 'no shape'}")
        
        if not isinstance(inputs, torch.Tensor):
            logger.error(f"_call_model: Expected torch.Tensor, got {type(inputs)}")
            logger.error(f"_call_model: Input value: {inputs}")
            raise TypeError(f"Expected torch.Tensor, got {type(inputs)}")
        
        # Check if model has a specific forward signature
        import inspect
        try:
            sig = inspect.signature(self.model.forward)
            logger.info(f"_call_model: Model forward signature: {sig}")
        except:
            logger.info("_call_model: Could not inspect model forward signature")
        
        # Try different calling conventions
        try:
            # Method 1: Direct positional call
            logger.info("_call_model: Trying positional call...")
            result = self.model(inputs)
            logger.info("_call_model: Positional call succeeded!")
            return result
        except Exception as e1:
            logger.error(f"_call_model: Positional call failed: {e1}")
            logger.error(f"_call_model: Error type: {type(e1)}")
            try:
                # Method 2: Keyword argument call
                logger.info("_call_model: Trying keyword call...")
                result = self.model(input_ids=inputs)
                logger.info("_call_model: Keyword call succeeded!")
                return result
            except Exception as e2:
                logger.error(f"_call_model: Keyword call failed: {e2}")
                logger.error(f"_call_model: Error type: {type(e2)}")
                try:
                    # Method 3: Call with output_hidden_states
                    logger.info("_call_model: Trying call with output_hidden_states...")
                    result = self.model(input_ids=inputs, output_hidden_states=True)
                    logger.info("_call_model: Call with output_hidden_states succeeded!")
                    return result
                except Exception as e3:
                    logger.error(f"_call_model: All model call methods failed:")
                    logger.error(f"  Positional: {e1}")
                    logger.error(f"  Keyword: {e2}")
                    logger.error(f"  With hidden states: {e3}")
                    
                    # Method 4: Try calling the model's forward method directly
                    try:
                        logger.info("_call_model: Trying direct forward call...")
                        result = self.model.forward(inputs)
                        logger.info("_call_model: Direct forward call succeeded!")
                        return result
                    except Exception as e4:
                        logger.error(f"_call_model: Direct forward call failed: {e4}")
                        raise e3
    
    def test_necessity(self, inputs: torch.Tensor, modes: List[str] = ['zero']) -> Dict:
        """
        Test if circuit is necessary by ablating it
        COMPLETE FIX: Use positional arguments
        """
        logger.info(f"🔍 CircuitTester.test_necessity called with {len(self.circuit_indices)} dimensions in layer {self.layer_idx}")
        logger.info(f"🔍 Input type: {type(inputs)}, shape: {inputs.shape if hasattr(inputs, 'shape') else 'no shape'}")
        
        results = {}
        
        # ✅ CRITICAL: Ensure inputs is tensor
        if isinstance(inputs, dict):
            inputs = inputs.get('input_ids', inputs)
        
        if not isinstance(inputs, torch.Tensor):
            raise TypeError(f"Expected tensor, got {type(inputs)}")
        
        logger.info(f"   Input tensor type: {type(inputs)}, shape: {inputs.shape}")
        
        # ✅ CRITICAL: Double-check that inputs is actually a tensor
        if not isinstance(inputs, torch.Tensor):
            logger.error(f"Inputs is not a tensor! Type: {type(inputs)}, Value: {inputs}")
            raise TypeError(f"Expected torch.Tensor, got {type(inputs)}")
        
        # Get baseline output WITHOUT hooks
        with torch.no_grad():
            logger.info(f"   Calling model with tensor: {inputs.shape}")
            baseline_output = self._call_model(inputs)
            baseline_logits = baseline_output.logits if hasattr(baseline_output, 'logits') else baseline_output[0]
        
        for mode in modes:
            logger.info(f"Running necessity test: ablating {len(self.circuit_indices)} dimensions in layer {self.layer_idx}")
            
            try:
                # Register ablation hook
                hook_fn = self._create_ablation_hook(mode)
                handle = self._register_hook_at_layer(self.layer_idx, hook_fn)
                self.hooks.append(handle)
                
                # Forward pass with ablation
                with torch.no_grad():
                    ablated_output = self._call_model(inputs)
                    ablated_logits = ablated_output.logits if hasattr(ablated_output, 'logits') else ablated_output[0]
                
                # Measure effect
                logit_diff = torch.abs(baseline_logits - ablated_logits).mean().item()
                
                results[mode] = {
                    'logit_difference': logit_diff,
                    'success': True
                }
                
                # Clean up hook immediately
                handle.remove()
                
            except Exception as e:
                logger.error(f"Necessity test failed for mode {mode}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                results[mode] = {
                    'logit_difference': 0.0,
                    'success': False,
                    'error': str(e)
                }
                try:
                    handle.remove()
                except:
                    pass
        
        return results
    
    def test_sufficiency(self, target_inputs: torch.Tensor, reference_inputs: torch.Tensor) -> Dict:
        """
        Test if circuit is sufficient by patching it
        COMPLETE FIX: Use positional arguments
        """
        logger.info(f"Running sufficiency test: patching {len(self.circuit_indices)} dimensions from reference to target")
        
        # ✅ CRITICAL: Ensure tensors
        if isinstance(target_inputs, dict):
            target_inputs = target_inputs.get('input_ids', target_inputs)
        if isinstance(reference_inputs, dict):
            reference_inputs = reference_inputs.get('input_ids', reference_inputs)
        
        try:
            # Step 1: Collect reference activations
            self.stored_activations = {}
            
            def save_hook(module, input, output):
                if isinstance(output, tuple):
                    self.stored_activations['hidden'] = output[0].detach().clone()
                else:
                    self.stored_activations['hidden'] = output.detach().clone()
                return output
            
            handle = self._register_hook_at_layer(self.layer_idx, save_hook)
            
            with torch.no_grad():
                _ = self._call_model(reference_inputs)
            
            handle.remove()
            
            # Step 2: Get baseline
            with torch.no_grad():
                baseline_output = self._call_model(target_inputs)
                baseline_logits = baseline_output.logits if hasattr(baseline_output, 'logits') else baseline_output[0]
            
            # Step 3: Patch
            def patch_hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0].clone()
                    rest = output[1:]
                else:
                    hidden = output.clone()
                    rest = None
                
                if 'hidden' in self.stored_activations:
                    ref_hidden = self.stored_activations['hidden']
                    min_seq = min(hidden.shape[1], ref_hidden.shape[1])
                    
                    for idx in self.circuit_indices:
                        if idx < hidden.shape[-1] and idx < ref_hidden.shape[-1]:
                            hidden[:, :min_seq, idx] = ref_hidden[:, :min_seq, idx]
                
                if rest is not None:
                    return (hidden,) + rest
                return hidden
            
            handle = self._register_hook_at_layer(self.layer_idx, patch_hook)
            
            with torch.no_grad():
                patched_output = self._call_model(target_inputs)
                patched_logits = patched_output.logits if hasattr(patched_output, 'logits') else patched_output[0]
            
            handle.remove()
            
            logit_diff = torch.abs(baseline_logits - patched_logits).mean().item()
            
            return {
                'logit_difference': logit_diff,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Sufficiency test failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'logit_difference': 0.0,
                'success': False,
                'error': str(e)
            }
    
    def run_control_tests(self, inputs: torch.Tensor, num_controls: int = 10) -> Dict:
        """
        Test random circuits as controls
        COMPLETE FIX: Use positional arguments
        """
        logger.info(f"Running {num_controls} random control tests")
        
        # ✅ CRITICAL: Ensure tensor
        if isinstance(inputs, dict):
            inputs = inputs.get('input_ids', inputs)
        
        try:
            # Get hidden size
            with torch.no_grad():
                test_output = self._call_model(inputs[:1])
                
                if hasattr(test_output, 'hidden_states') and test_output.hidden_states:
                    hidden_size = test_output.hidden_states[self.layer_idx].shape[-1]
                else:
                    hidden_size = 768  # Fallback
            
            control_effects = []
            
            for i in range(num_controls):
                num_dims = len(self.circuit_indices)
                random_indices = np.random.choice(
                    hidden_size, 
                    size=min(num_dims, hidden_size),
                    replace=False
                ).tolist()
                
                temp_tester = CircuitTester(self.model, self.layer_idx, random_indices)
                necessity = temp_tester.test_necessity(inputs, modes=['zero'])
                
                if necessity.get('zero', {}).get('success'):
                    control_effects.append(necessity['zero']['logit_difference'])
            
            if control_effects:
                return {
                    'mean_effect': float(np.mean(control_effects)),
                    'std_effect': float(np.std(control_effects)),
                    'num_successful': len(control_effects)
                }
            else:
                return {
                    'mean_effect': 0.0,
                    'std_effect': 0.0,
                    'num_successful': 0,
                    'error': 'All control tests failed'
                }
            
        except Exception as e:
            logger.error(f"Control tests failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'mean_effect': 0.0,
                'std_effect': 0.0,
                'num_successful': 0,
                'error': str(e)
            }
    
    def _create_ablation_hook(self, mode: str):
        """Create hook function for ablation"""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0].clone()
            else:
                hidden = output.clone()
            
            # Apply ablation
            indices = torch.tensor(self.circuit_indices, dtype=torch.long, device=hidden.device)
            
            if mode == 'zero':
                hidden[..., indices] = 0
            elif mode == 'noise':
                noise = torch.randn_like(hidden[..., indices])
                hidden[..., indices] = noise
            elif mode == 'mean':
                mean_val = hidden[..., indices].mean(dim=-1, keepdim=True)
                hidden[..., indices] = mean_val
            
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        
        return hook_fn
    
    def _register_hook_at_layer(self, layer_idx: int, hook_fn):
        """Register hook at specific layer"""
        layer = self.model.backbone.layers[layer_idx]
        return layer.register_forward_hook(hook_fn)
    
    def is_circuit_significant(self, necessity_results: Dict, control_results: Dict) -> bool:
        """Check if circuit effect is significantly larger than controls"""
        if not necessity_results.get('zero', {}).get('success'):
            return False
        
        circuit_effect = necessity_results['zero']['logit_difference']
        control_mean = control_results.get('mean_effect', 0.0)
        control_std = control_results.get('std_effect', 0.0)
        
        # Circuit is significant if effect > mean + 2*std
        threshold = control_mean + 2 * control_std
        return circuit_effect > threshold

def run_activation_patching_analysis(
    model,
    inputs: torch.Tensor,  # ← Should be tensor, not dict
    candidate_circuits: List[List[int]],
    layer_idx: int,
    reference_inputs: torch.Tensor = None  # ← Should be tensor
) -> Dict[str, Any]:
    """
    FIXED: Proper tensor handling
    """
    logger.info(f"Running activation patching analysis on {len(candidate_circuits)} circuits")
    
    # ✅ CRITICAL FIX: Ensure inputs are tensors
    if isinstance(inputs, dict):
        if 'input_ids' in inputs:
            inputs = inputs['input_ids']
        else:
            raise ValueError(f"Expected tensor or dict with 'input_ids', got dict with keys: {inputs.keys()}")
    
    if reference_inputs is None:
        reference_inputs = inputs
    elif isinstance(reference_inputs, dict):
        reference_inputs = reference_inputs.get('input_ids', reference_inputs)
    
    # ✅ ADDITIONAL CHECK: Ensure we have actual tensors
    if not isinstance(inputs, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(inputs)}")
    if not isinstance(reference_inputs, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(reference_inputs)}")
    
    logger.info(f"   Input tensor shape: {inputs.shape}")
    logger.info(f"   Reference tensor shape: {reference_inputs.shape}")
    
    results = {}
    
    for i, circuit_indices in enumerate(candidate_circuits):
        logger.info(f"Testing circuit {i+1}/{len(candidate_circuits)}: {len(circuit_indices)} dimensions")
        
        circuit_tester = CircuitTester(model, layer_idx, circuit_indices)
        
        try:
            # Test necessity
            necessity_results = circuit_tester.test_necessity(
                inputs=inputs,  # ← Tensor
                modes=['zero', 'noise', 'mean']
            )
            
            # Test sufficiency
            sufficiency_results = circuit_tester.test_sufficiency(
                target_inputs=inputs,  # ← Tensor
                reference_inputs=reference_inputs  # ← Tensor
            )
            
            # Random controls
            control_results = circuit_tester.run_control_tests(
                inputs=inputs,  # ← Tensor
                num_controls=10
            )
            
            results[f"circuit_{i}"] = {
                'circuit_indices': circuit_indices,
                'necessity': necessity_results,
                'sufficiency': sufficiency_results,
                'controls': control_results,
                'is_significant': circuit_tester.is_circuit_significant(
                    necessity_results, control_results
                )
            }
            
        except Exception as e:
            logger.error(f"Circuit {i} testing failed: {e}")
            results[f"circuit_{i}"] = {
                'circuit_indices': circuit_indices,
                'error': str(e)
            }
    
    return results

class CircuitCausalityAnalyzer:
    """Main analyzer class for circuit causality testing"""
    
    def __init__(self, model, tokenizer, config: Dict):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.circuit_candidates = []
        self.patching_results = {}
        
        # Create experiment logger
        self.experiment_logger = ExperimentLogger(config.get('output_dir', './results'))
    
    def set_circuit_candidates(self, candidates: List[Any]):
        """Set the circuit candidates for testing"""
        self.circuit_candidates = candidates
        logger.info(f"Set {len(candidates)} circuit candidates for testing")
    
    def test_circuit_causality(self, layer_idx: int = 0) -> Dict[str, Any]:
        """
        Step 6: Activation patching for necessity and sufficiency testing.
        FIXED: Pass tensor inputs, not dicts
        """
        logger.info(f"Step 6: Testing circuit causality with activation patching for layer {layer_idx}...")
        
        if not self.circuit_candidates:
            logger.error("No candidate circuits available. Run circuit selection first.")
            return {}
        
        # Prepare test inputs
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming industries worldwide.",
            "Machine learning models require large amounts of training data."
        ]
        
        # Tokenize all texts with consistent length
        all_inputs = []
        max_length = 0
        
        for text in test_texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            max_length = max(max_length, inputs["input_ids"].shape[1])
            all_inputs.append(inputs)
        
        # Pad all sequences to max length
        padded_inputs = []
        for inputs in all_inputs:
            input_ids = inputs["input_ids"]
            if input_ids.shape[1] < max_length:
                pad_length = max_length - input_ids.shape[1]
                padding = torch.full((1, pad_length), self.tokenizer.pad_token_id, 
                                   dtype=input_ids.dtype, device=input_ids.device)
                input_ids = torch.cat([input_ids, padding], dim=1)
            padded_inputs.append(input_ids)
        
        # Concatenate into single batch
        test_input = torch.cat(padded_inputs, dim=0).to(self.config.device)  # [batch, seq_len]
        
        logger.info(f"   Test input type: {type(test_input)}, shape: {test_input.shape}")
        
        # ✅ FIX: Extract simple indices from circuit dicts
        circuit_indices_only = []
        for circuit in self.circuit_candidates:
            if isinstance(circuit, dict):
                indices = circuit.get('indices', [])
            else:
                indices = circuit
            circuit_indices_only.append(indices)
        
        logger.info(f"   Testing {len(circuit_indices_only)} circuits")
        
        # ✅ CRITICAL FIX: Ensure test_input is a tensor, not dict
        if isinstance(test_input, dict):
            logger.error(f"   ERROR: test_input is a dict! Keys: {test_input.keys()}")
            test_input = test_input.get('input_ids', test_input)
        
        logger.info(f"   Final test_input type: {type(test_input)}, shape: {test_input.shape if hasattr(test_input, 'shape') else 'no shape'}")
        
        # ✅ DEBUG: Test model call directly
        logger.info("   Testing model call directly...")
        try:
            with torch.no_grad():
                test_output = self.model(test_input)
            logger.info(f"   Direct model call succeeded! Output type: {type(test_output)}")
        except Exception as e:
            logger.error(f"   Direct model call failed: {e}")
            logger.error(f"   Error type: {type(e)}")
        
        # ✅ FIX: Pass ONLY input_ids tensor, not dict
        patching_results = run_activation_patching_analysis(
            model=self.model,
            inputs=test_input,  # ← Just the tensor, not a dict!
            candidate_circuits=circuit_indices_only,
            layer_idx=layer_idx,
            reference_inputs=test_input  # ← Also just tensor
        )
        
        # Merge results with circuit metadata
        enhanced_results = {}
        for i, (circuit_meta, patch_result) in enumerate(zip(self.circuit_candidates, patching_results.values())):
            circuit_key = f"circuit_{i}"
            enhanced_results[circuit_key] = {
                'circuit_info': circuit_meta if isinstance(circuit_meta, dict) else {'indices': circuit_meta},
                'patching_results': patch_result
            }
        
        self.patching_results[layer_idx] = enhanced_results
        self.experiment_logger.save_results(enhanced_results, f"patching_results_layer_{layer_idx}.json")
        
        logger.info("✅ Circuit causality testing complete!")
        return enhanced_results
    
    def get_significant_circuits(self, layer_idx: int = 0) -> List[Dict]:
        """Get circuits that showed significant causal effects"""
        if layer_idx not in self.patching_results:
            logger.warning(f"No patching results available for layer {layer_idx}")
            return []
        
        significant_circuits = []
        results = self.patching_results[layer_idx]
        
        for circuit_key, circuit_data in results.items():
            patching_results = circuit_data.get('patching_results', {})
            if patching_results.get('is_significant', False):
                significant_circuits.append({
                    'circuit_key': circuit_key,
                    'circuit_info': circuit_data.get('circuit_info', {}),
                    'necessity': patching_results.get('necessity', {}),
                    'sufficiency': patching_results.get('sufficiency', {}),
                    'controls': patching_results.get('controls', {})
                })
        
        logger.info(f"Found {len(significant_circuits)} significant circuits in layer {layer_idx}")
        return significant_circuits
    
    def summarize_results(self, layer_idx: int = 0) -> Dict[str, Any]:
        """Summarize circuit causality testing results"""
        if layer_idx not in self.patching_results:
            return {'error': f'No results available for layer {layer_idx}'}
        
        results = self.patching_results[layer_idx]
        total_circuits = len(results)
        significant_circuits = sum(1 for r in results.values() 
                                 if r.get('patching_results', {}).get('is_significant', False))
        
        # Collect necessity effects
        necessity_effects = []
        sufficiency_effects = []
        
        for circuit_data in results.values():
            patching_results = circuit_data.get('patching_results', {})
            necessity = patching_results.get('necessity', {})
            sufficiency = patching_results.get('sufficiency', {})
            
            if necessity.get('zero', {}).get('success'):
                necessity_effects.append(necessity['zero']['logit_difference'])
            
            if sufficiency.get('success'):
                sufficiency_effects.append(sufficiency['logit_difference'])
        
        summary = {
            'layer_idx': layer_idx,
            'total_circuits_tested': total_circuits,
            'significant_circuits': significant_circuits,
            'significance_rate': significant_circuits / total_circuits if total_circuits > 0 else 0,
            'mean_necessity_effect': float(np.mean(necessity_effects)) if necessity_effects else 0.0,
            'mean_sufficiency_effect': float(np.mean(sufficiency_effects)) if sufficiency_effects else 0.0,
            'max_necessity_effect': float(np.max(necessity_effects)) if necessity_effects else 0.0,
            'max_sufficiency_effect': float(np.max(sufficiency_effects)) if sufficiency_effects else 0.0
        }
        
        return summary

class ExperimentLogger:
    """Simple experiment logger for saving results"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def save_results(self, results: Dict, filename: str):
        """Save results to JSON file"""
        filepath = f"{self.output_dir}/{filename}"
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {filepath}")

def run_sae_analysis(activations: torch.Tensor, task_labels: np.ndarray, config: Dict) -> Dict:
    """
    FIXED: Ensure all required fields are returned
    """
    import torch.nn as nn
    import torch.optim as optim
    
    # Normalize inputs
    acts_np = activations.cpu().numpy()
    acts_mean = acts_np.mean(axis=0, keepdims=True)
    acts_std = acts_np.std(axis=0, keepdims=True) + 1e-8
    acts_normalized = (acts_np - acts_mean) / acts_std
    acts_tensor = torch.from_numpy(acts_normalized).float()
    
    # Define SAE
    input_dim = acts_normalized.shape[1]
    latent_dim = int(input_dim * config['latent_dim_ratio'])
    
    class SimpleAutoencoder(nn.Module):
        def __init__(self, input_dim, latent_dim):
            super().__init__()
            self.encoder = nn.Linear(input_dim, latent_dim)
            self.decoder = nn.Linear(latent_dim, input_dim)
            nn.init.xavier_uniform_(self.encoder.weight)
            nn.init.xavier_uniform_(self.decoder.weight)
            
        def forward(self, x):
            latent = torch.relu(self.encoder(x))
            reconstructed = self.decoder(latent)
            return reconstructed, latent
    
    # Train SAE
    sae = SimpleAutoencoder(input_dim, latent_dim)
    optimizer = optim.Adam(sae.parameters(), lr=config.get('learning_rate', 1e-3))
    
    for epoch in range(config.get('num_epochs', 50)):
        optimizer.zero_grad()
        reconstructed, latent = sae(acts_tensor)
        
        # Loss
        recon_loss = nn.MSELoss()(reconstructed, acts_tensor)
        sparsity_loss = config['sparsity_weight'] * torch.abs(latent).mean()
        loss = recon_loss + sparsity_loss
        
        loss.backward()
        optimizer.step()
    
    # Extract latent features
    with torch.no_grad():
        _, latent_features = sae(acts_tensor)
        latent_np = latent_features.numpy()
    
    # ✅ FIX: Compute correlations properly
    correlations = []
    for i in range(latent_np.shape[1]):
        corr = np.corrcoef(latent_np[:, i], task_labels)[0, 1]
        correlations.append(corr if not np.isnan(corr) else 0.0)
    
    # ✅ FIX: Extract top correlated dimensions
    correlations_array = np.array(correlations)
    sorted_indices = np.argsort(np.abs(correlations_array))[::-1]  # Sort by absolute value
    
    top_k = min(50, len(sorted_indices))  # Top 50 or all if fewer
    top_indices = sorted_indices[:top_k].tolist()
    top_corrs = [correlations[i] for i in top_indices]
    
    # ✅ FIX: Return ALL required fields including model_state
    results = {
        'correlations': correlations,
        'top_correlated_dims': top_indices,        # ← CRITICAL: Add this
        'top_correlations': top_corrs,             # ← CRITICAL: Add this
        'max_correlation': float(max(np.abs(correlations_array))),
        'mean_correlation': float(np.mean(correlations_array)),
        'sparsity': float((np.abs(latent_np) < 1e-3).mean()),
        'reconstruction_mse': float(recon_loss.item()),
        'num_latents': latent_dim,
        'num_active': int((np.abs(latent_np) > 1e-3).any(axis=0).sum()),
        'model_state': {  # ← CRITICAL: Add model_state for hypothesis probes
            'encoder_weight': sae.encoder.weight.detach().cpu().numpy().tolist(),
            'decoder_weight': sae.decoder.weight.detach().cpu().numpy().tolist(),
            'encoder_bias': sae.encoder.bias.detach().cpu().numpy().tolist(),
            'decoder_bias': sae.decoder.bias.detach().cpu().numpy().tolist(),
        },
        'latent_codes': latent_np.tolist()  # ← Also add latent codes for direct access
    }
    
    return results

def example_circuit_causality_usage():
    """
    Example usage of the circuit causality testing functionality.
    This demonstrates how to use the CircuitCausalityAnalyzer.
    """
    logger.info("Circuit Causality Testing Example")
    
    # This is a mock example - in practice you would use real model and tokenizer
    class MockModel:
        def __init__(self):
            self.backbone = MockBackbone()
        
        def __call__(self, input_ids, output_hidden_states=False):
            batch_size, seq_len = input_ids.shape
            hidden_size = 768
            
            # Mock hidden states
            hidden_states = []
            for i in range(12):  # 12 layers
                hidden_states.append(torch.randn(batch_size, seq_len, hidden_size))
            
            # Mock logits
            logits = torch.randn(batch_size, seq_len, 50257)  # vocab size
            
            class MockOutput:
                def __init__(self, logits, hidden_states):
                    self.logits = logits
                    self.hidden_states = hidden_states
            
            return MockOutput(logits, hidden_states)
    
    class MockBackbone:
        def __init__(self):
            self.layers = [MockLayer() for _ in range(12)]
    
    class MockLayer:
        def register_forward_hook(self, hook_fn):
            return MockHook()
    
    class MockHook:
        def remove(self):
            pass
    
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
        
        def __call__(self, text, return_tensors="pt", truncation=True, max_length=128):
            # Mock tokenization
            tokens = [1, 2, 3, 4, 5]  # Mock token IDs
            return {"input_ids": torch.tensor([tokens])}
    
    # Create mock components
    model = MockModel()
    tokenizer = MockTokenizer()
    config = {
        'device': 'cpu',
        'output_dir': './circuit_results'
    }
    
    # Initialize analyzer
    analyzer = CircuitCausalityAnalyzer(model, tokenizer, config)
    
    # Set some mock circuit candidates
    circuit_candidates = [
        {'indices': [10, 25, 42, 67], 'description': 'Circuit A'},
        {'indices': [15, 30, 55, 80], 'description': 'Circuit B'},
        {'indices': [5, 20, 35, 50], 'description': 'Circuit C'}
    ]
    
    analyzer.set_circuit_candidates(circuit_candidates)
    
    # Test circuit causality
    logger.info("Running circuit causality tests...")
    results = analyzer.test_circuit_causality(layer_idx=0)
    
    # Get significant circuits
    significant = analyzer.get_significant_circuits(layer_idx=0)
    
    # Summarize results
    summary = analyzer.summarize_results(layer_idx=0)
    
    logger.info("Circuit causality testing complete!")
    logger.info(f"Summary: {summary}")
    
    return {
        'results': results,
        'significant_circuits': significant,
        'summary': summary
    }

if __name__ == "__main__":
    # Example usage
    logger.info("SAE implementation complete!")
    
    # Create dummy data for testing
    num_samples = 1000
    input_dim = 768
    
    activations = torch.randn(num_samples, input_dim)
    task_labels = torch.randint(0, 5, (num_samples,))
    
    # Run SAE analysis
    config = {
        'latent_dim_ratio': 0.5,
        'sparsity_weight': 1e-3,
        'learning_rate': 1e-3,
        'num_epochs': 50
    }
    
    results = run_sae_analysis(activations, task_labels, config)
    
    print("SAE analysis complete!")
    print(f"Found {len(results['top_correlated_dims'])} top correlated dimensions")
    print(f"Max correlation: {results['max_correlation']:.3f}")
    print(f"Sparsity rate: {results['sparsity']:.3f}")
    
    # Example circuit causality testing
    print("\n" + "="*50)
    print("Circuit Causality Testing Example")
    print("="*50)
    
    circuit_example = example_circuit_causality_usage()
    print(f"Circuit testing summary: {circuit_example['summary']}")
