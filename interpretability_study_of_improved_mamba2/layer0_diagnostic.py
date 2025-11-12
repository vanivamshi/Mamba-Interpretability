#!/usr/bin/env python3
"""
Diagnostic routine to check for real signal in layer 0 features
for comparison with layer 6.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path

def load_layer0_data():
    """Load the SAE results and configuration for layer 0."""
    # Use the layer 0 experiment directory
    experiment_path = Path('experiment_logs/experiment_20251011_121824')
    
    print(f"Loading data from: {experiment_path}")
    
    # Load SAE results
    sae_file = experiment_path / f"sae_results_layer_0.json"
    if not sae_file.exists():
        raise FileNotFoundError(f"SAE results file not found: {sae_file}")
    
    with open(sae_file, 'r') as f:
        sae_data = json.load(f)
    
    # Load config
    config_file = experiment_path / "config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return sae_data, config, experiment_path

def run_layer0_diagnostic():
    """Run the complete diagnostic routine for layer 0."""
    print("🔍 Layer 0 Feature Signal Diagnostic")
    print("=" * 50)
    
    # === 1️⃣ Load SAE correlation data ===
    try:
        sae_output, config, experiment_path = load_layer0_data()
        
        # Extract correlation data
        correlation_data = sae_output["correlation_results"]["correlations"]
        first_task = list(correlation_data.keys())[0]
        correlations = np.array(correlation_data[first_task])
        
        # Handle NaN values
        valid_mask = ~np.isnan(correlations)
        correlations_clean = correlations[valid_mask]
        feature_indices = np.where(valid_mask)[0]
        
        print(f"✅ Loaded correlation data: {len(correlations)} total features")
        print(f"✅ Valid correlations: {len(correlations_clean)} features")
        print(f"✅ Available task variables: {list(correlation_data.keys())}")
        print(f"✅ NaN values: {(~valid_mask).sum()}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Use correlations as our "features" for analysis
    features = correlations_clean.reshape(-1, 1)  # Reshape to 2D for sklearn
    task_labels = np.ones(len(features))  # Dummy labels for correlation analysis
    
    print(f"✅ Prepared data: features {features.shape}, labels {task_labels.shape}")
    
    # === 2️⃣ Standardize features ===
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    print(f"✅ Features standardized")
    
    # === 3️⃣ RidgeCV probe (never zeroes weights) ===
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 7))
    ridge.fit(features_scaled, task_labels)
    
    print(f"\n📊 Ridge Regression Results:")
    print(f"Best alpha: {ridge.alpha_:.4g}")
    print(f"Ridge R^2: {ridge.score(features_scaled, task_labels):.4f}")
    
    # === 4️⃣ Analyze correlation statistics ===
    abs_corrs = np.abs(correlations_clean)
    
    print(f"\n📈 Correlation Statistics:")
    print(f"Max |corr| = {abs_corrs.max():.4f}")
    print(f"Mean |corr| = {abs_corrs.mean():.4f}")
    print(f"Std |corr| = {abs_corrs.std():.4f}")
    print(f"Median |corr| = {np.median(abs_corrs):.4f}")
    
    # === 5️⃣ Plot correlation distribution ===
    plt.figure(figsize=(10, 6))
    plt.hist(abs_corrs, bins=50, color="blue", edgecolor="black", alpha=0.7)
    plt.title(f"Layer 0 Feature–Task Correlation Distribution")
    plt.xlabel("|Correlation|")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines for statistics
    plt.axvline(abs_corrs.mean(), color='red', linestyle='--', 
                label=f'Mean: {abs_corrs.mean():.3f}')
    plt.axvline(np.median(abs_corrs), color='green', linestyle='--', 
                label=f'Median: {np.median(abs_corrs):.3f}')
    plt.legend()
    
    # Save plot
    plot_path = experiment_path / "layer0_correlation_histogram.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Correlation histogram saved to: {plot_path}")
    plt.show()
    
    # === 6️⃣ Inspect top correlated features ===
    top_k = 10
    top_indices = np.argsort(-abs_corrs)[:top_k]
    
    print(f"\n🔝 Top {top_k} Correlated Features:")
    print("-" * 60)
    print(f"{'Feature':<8} {'Correlation':<12} {'Abs Corr':<10} {'Original Index':<15}")
    print("-" * 60)
    
    for i, idx in enumerate(top_indices):
        orig_idx = feature_indices[idx]
        corr_val = correlations_clean[idx]
        print(f"{i:<8} {corr_val:+11.3f} {abs_corrs[idx]:+9.3f} {orig_idx:<15}")
    
    # === 7️⃣ Additional diagnostics ===
    print(f"\n🔬 Additional Diagnostics:")
    print(f"Features with |corr| > 0.1: {(abs_corrs > 0.1).sum()}")
    print(f"Features with |corr| > 0.05: {(abs_corrs > 0.05).sum()}")
    print(f"Features with |corr| > 0.01: {(abs_corrs > 0.01).sum()}")
    print(f"Features with |corr| > 1e-17: {(abs_corrs > 1e-17).sum()}")
    print(f"Features with |corr| > 1e-18: {(abs_corrs > 1e-18).sum()}")
    
    # Check for zero correlations
    zero_corrs = (abs_corrs == 0).sum()
    print(f"Features with zero correlation: {zero_corrs}")
    
    # Additional analysis for very small correlations
    print(f"\n🔍 Detailed Analysis of Small Correlations:")
    print(f"Correlations in scientific notation:")
    print(f"Max correlation: {abs_corrs.max():.2e}")
    print(f"Mean correlation: {abs_corrs.mean():.2e}")
    print(f"Std correlation: {abs_corrs.std():.2e}")
    
    # Check if correlations are essentially zero (within machine precision)
    machine_epsilon = np.finfo(float).eps
    essentially_zero = (abs_corrs < machine_epsilon).sum()
    print(f"Features with correlations < machine epsilon ({machine_epsilon:.2e}): {essentially_zero}")
    
    # Summary interpretation
    print(f"\n📋 Summary Interpretation:")
    if abs_corrs.max() < 1e-15:
        print("⚠️  WARNING: All correlations are essentially zero (< 1e-15)")
        print("   This suggests:")
        print("   - No meaningful signal between SAE features and task variables")
        print("   - Possible issues with feature extraction or task definition")
        print("   - SAE may not be learning task-relevant representations")
    elif abs_corrs.max() < 1e-10:
        print("⚠️  CAUTION: Correlations are extremely small (< 1e-10)")
        print("   This suggests very weak signal, possibly due to:")
        print("   - Insufficient training data")
        print("   - Poor task definition")
        print("   - SAE not converging to meaningful features")
    else:
        print("✅ Correlations show some signal, though still quite small")
    
    # Save results
    results = {
        "layer_idx": 0,
        "n_features": int(len(correlations)),
        "n_valid_features": int(len(correlations_clean)),
        "n_nan_features": int((~valid_mask).sum()),
        "ridge_alpha": float(ridge.alpha_),
        "ridge_r2": float(ridge.score(features_scaled, task_labels)),
        "max_correlation": float(abs_corrs.max()),
        "mean_correlation": float(abs_corrs.mean()),
        "std_correlation": float(abs_corrs.std()),
        "median_correlation": float(np.median(abs_corrs)),
        "top_features": {
            "indices": top_indices.tolist(),
            "original_indices": feature_indices[top_indices].tolist(),
            "correlations": correlations_clean[top_indices].tolist(),
            "abs_correlations": abs_corrs[top_indices].tolist()
        },
        "correlation_thresholds": {
            "gt_0_1": int((abs_corrs > 0.1).sum()),
            "gt_0_05": int((abs_corrs > 0.05).sum()),
            "gt_0_01": int((abs_corrs > 0.01).sum()),
            "gt_1e_17": int((abs_corrs > 1e-17).sum()),
            "gt_1e_18": int((abs_corrs > 1e-18).sum()),
            "zero_correlations": int(zero_corrs)
        }
    }
    
    results_path = experiment_path / "layer0_diagnostic_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Diagnostic results saved to: {results_path}")
    
    return results

if __name__ == "__main__":
    results = run_layer0_diagnostic()
