#!/usr/bin/env python3
"""
Diagnose the SAE analysis pipeline to find why Ridge coefficients are zero.
"""

import torch
import numpy as np
import json
from pathlib import Path

def diagnose_sae_pipeline():
    """Diagnose the SAE analysis pipeline step by step."""
    print("🔍 Diagnosing SAE Analysis Pipeline")
    print("=" * 50)
    
    # Load the most recent SAE results
    experiment_dirs = sorted([d for d in Path("experiment_logs").iterdir() if d.name.startswith("experiment_")])
    if not experiment_dirs:
        print("❌ No experiment directories found")
        return
    
    # Use the directory with SAE results (experiment_20251011_201444)
    latest_dir = Path("experiment_logs/experiment_20251011_201444")
    sae_file = latest_dir / "sae_results_layer_0.json"
    
    if not sae_file.exists():
        print(f"❌ SAE results file not found: {sae_file}")
        return
    
    print(f"📁 Loading SAE results from: {latest_dir}")
    
    with open(sae_file, 'r') as f:
        sae_data = json.load(f)
    
    # Check 1: SAE training history
    print(f"\n📊 SAE Training Analysis:")
    train_losses = sae_data["training_history"]["train_losses"]
    val_losses = sae_data["training_history"]["val_losses"]
    sparsity_rates = sae_data["training_history"]["sparsity_rates"]
    
    print(f"  Final train loss: {train_losses[-1]:.6f}")
    print(f"  Final val loss: {val_losses[-1]:.6f}")
    print(f"  Final sparsity: {sparsity_rates[-1]:.4f}")
    print(f"  Loss reduction: {train_losses[0]/train_losses[-1]:.2f}x")
    
    # Check 2: Model state
    print(f"\n🔧 SAE Model State:")
    encoder_weight = np.array(sae_data["model_state"]["encoder_weight"])
    print(f"  Encoder weight shape: {encoder_weight.shape}")
    print(f"  Encoder weight stats: mean={encoder_weight.mean():.6f}, std={encoder_weight.std():.6f}")
    print(f"  Encoder weight range: [{encoder_weight.min():.6f}, {encoder_weight.max():.6f}]")
    
    # Check 3: Correlation results
    print(f"\n📈 Correlation Analysis:")
    correlations = sae_data["correlation_results"]["correlations"]["main_task_var_0"]
    correlations = np.array(correlations)
    
    # Handle NaN values
    valid_mask = ~np.isnan(correlations)
    correlations_clean = correlations[valid_mask]
    
    print(f"  Total correlations: {len(correlations)}")
    print(f"  Valid correlations: {len(correlations_clean)}")
    print(f"  NaN correlations: {(~valid_mask).sum()}")
    print(f"  Max correlation: {correlations_clean.max():.2e}")
    print(f"  Mean correlation: {correlations_clean.mean():.2e}")
    print(f"  Std correlation: {correlations_clean.std():.2e}")
    
    # Check 4: Task labels
    print(f"\n🎯 Task Label Analysis:")
    print("  Generating task labels to check...")
    
    # Generate the same task labels as the analysis
    n_samples = 100  # Assuming 100 samples
    rng = np.random.default_rng(42)
    x = np.linspace(0, 8 * np.pi, n_samples)
    signal = np.sin(x) + 0.5 * np.sin(0.5 * x)
    noise = rng.normal(0, 0.1, n_samples)
    labels = signal + noise
    labels = (labels - labels.min()) / (labels.max() - labels.min())
    
    print(f"  Task labels shape: {labels.shape}")
    print(f"  Task labels stats: mean={labels.mean():.4f}, std={labels.std():.4f}")
    print(f"  Task labels range: [{labels.min():.4f}, {labels.max():.4f}]")
    
    # Check 5: Simulate SAE encoding
    print(f"\n🧠 SAE Encoding Simulation:")
    
    # Create dummy activations (similar to what Mamba would produce)
    dummy_activations = np.random.randn(100, 768) * 0.5  # Reasonable activation scale
    
    # Encode through SAE
    latent = dummy_activations.dot(encoder_weight.T)
    
    print(f"  Input activations shape: {dummy_activations.shape}")
    print(f"  Input activations stats: mean={dummy_activations.mean():.4f}, std={dummy_activations.std():.4f}")
    print(f"  Latent codes shape: {latent.shape}")
    print(f"  Latent codes stats: mean={latent.mean():.4f}, std={latent.std():.4f}")
    
    # Check 6: Ridge regression on simulated data
    print(f"\n🔍 Ridge Regression Test:")
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeCV
    
    # Remove zero-variance dimensions
    stds = latent.std(axis=0)
    nonzero_dims = stds > 1e-8
    latent_clean = latent[:, nonzero_dims]
    
    print(f"  Removed {latent.shape[1] - latent_clean.shape[1]} zero-variance dims")
    
    # Standardize and fit Ridge
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(latent_clean)
    
    ridge = RidgeCV(alphas=np.logspace(-6, 6, 13), cv=5)
    ridge.fit(features_scaled, labels)
    
    print(f"  Ridge coefficients stats: mean={ridge.coef_.mean():.4f}, std={ridge.coef_.std():.4f}")
    print(f"  Ridge coefficients range: [{ridge.coef_.min():.4f}, {ridge.coef_.max():.4f}]")
    print(f"  Non-zero coefficients: {(ridge.coef_ != 0).sum()}")
    
    # Check 7: Correlation analysis
    print(f"\n📊 Correlation Test:")
    correlations_test = np.array([
        np.corrcoef(features_scaled[:, i], labels)[0, 1] 
        for i in range(features_scaled.shape[1])
    ])
    
    print(f"  Test correlations stats: mean={correlations_test.mean():.4f}, std={correlations_test.std():.4f}")
    print(f"  Test correlations range: [{correlations_test.min():.4f}, {correlations_test.max():.4f}]")
    print(f"  Max test correlation: {correlations_test.max():.4f}")
    
    # Summary
    print(f"\n📋 Diagnosis Summary:")
    if correlations_clean.max() < 1e-10:
        print("  ❌ SAE correlations are essentially zero")
        print("  🔧 Possible causes:")
        print("    1. SAE not learning meaningful features")
        print("    2. Task labels not properly generated")
        print("    3. Activation collection issues")
        print("    4. SAE training problems")
    else:
        print("  ✅ SAE correlations look reasonable")
    
    if ridge.coef_.std() < 1e-10:
        print("  ❌ Ridge coefficients are essentially zero")
        print("  🔧 This suggests the SAE features don't correlate with task labels")
    else:
        print("  ✅ Ridge coefficients look reasonable")

if __name__ == "__main__":
    diagnose_sae_pipeline()
