#!/usr/bin/env python3
"""
Analyze SAE training history to check for convergence issues.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

def analyze_sae_training():
    """Analyze SAE training history for both layers."""
    print("🔍 SAE Training History Analysis")
    print("=" * 50)
    
    # Analyze both layers
    layers = [0, 6]
    experiment_dirs = [
        'experiment_logs/experiment_20251011_121824',  # layer 0
        'experiment_logs/experiment_20251011_145818'   # layer 6
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, (layer, exp_dir) in enumerate(zip(layers, experiment_dirs)):
        print(f"\n📊 Analyzing Layer {layer}:")
        
        # Load SAE results
        sae_file = Path(exp_dir) / f"sae_results_layer_{layer}.json"
        with open(sae_file, 'r') as f:
            sae_data = json.load(f)
        
        # Extract training history
        train_losses = np.array(sae_data["training_history"]["train_losses"])
        val_losses = np.array(sae_data["training_history"]["val_losses"])
        sparsity_rates = np.array(sae_data["training_history"]["sparsity_rates"])
        epochs = np.arange(1, len(train_losses) + 1)
        
        print(f"  ✅ Training epochs: {len(train_losses)}")
        print(f"  ✅ Final train loss: {train_losses[-1]:.6f}")
        print(f"  ✅ Final val loss: {val_losses[-1]:.6f}")
        print(f"  ✅ Final sparsity: {sparsity_rates[-1]:.4f}")
        print(f"  ✅ Loss reduction: {train_losses[0]/train_losses[-1]:.2f}x")
        
        # Plot training curves
        axes[i, 0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[i, 0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
        axes[i, 0].set_title(f'Layer {layer}: Loss Curves')
        axes[i, 0].set_xlabel('Epoch')
        axes[i, 0].set_ylabel('Loss')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_yscale('log')
        
        # Plot sparsity
        axes[i, 1].plot(epochs, sparsity_rates, 'g-', linewidth=2)
        axes[i, 1].set_title(f'Layer {layer}: Sparsity Rate')
        axes[i, 1].set_xlabel('Epoch')
        axes[i, 1].set_ylabel('Sparsity Rate')
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_ylim(0, 1)
        
        # Check for convergence issues
        print(f"  🔍 Convergence Analysis:")
        
        # Check if losses are still decreasing
        last_10_train = train_losses[-10:]
        train_trend = np.polyfit(range(len(last_10_train)), last_10_train, 1)[0]
        print(f"    Train loss trend (last 10 epochs): {train_trend:.2e}")
        
        last_10_val = val_losses[-10:]
        val_trend = np.polyfit(range(len(last_10_val)), last_10_val, 1)[0]
        print(f"    Val loss trend (last 10 epochs): {val_trend:.2e}")
        
        # Check for overfitting
        final_gap = val_losses[-1] - train_losses[-1]
        print(f"    Final train-val gap: {final_gap:.6f}")
        
        if abs(train_trend) < 1e-6 and abs(val_trend) < 1e-6:
            print(f"    ✅ Converged (losses stable)")
        elif train_trend > 0 or val_trend > 0:
            print(f"    ⚠️  Possible overfitting (losses increasing)")
        else:
            print(f"    ⚠️  Still converging (losses decreasing)")
        
        # Check sparsity convergence
        last_10_sparsity = sparsity_rates[-10:]
        sparsity_trend = np.polyfit(range(len(last_10_sparsity)), last_10_sparsity, 1)[0]
        print(f"    Sparsity trend (last 10 epochs): {sparsity_trend:.2e}")
        
        if abs(sparsity_trend) < 1e-4:
            print(f"    ✅ Sparsity converged")
        else:
            print(f"    ⚠️  Sparsity still changing")
    
    plt.tight_layout()
    plt.savefig('sae_training_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Training analysis plot saved to: sae_training_analysis.png")
    plt.show()
    
    # Compare training quality
    print(f"\n📋 Training Quality Comparison:")
    print(f"{'Metric':<20} {'Layer 0':<15} {'Layer 6':<15} {'Better':<10}")
    print("-" * 60)
    
    # Load both datasets for comparison
    layer0_data = json.load(open('experiment_logs/experiment_20251011_121824/sae_results_layer_0.json'))
    layer6_data = json.load(open('experiment_logs/experiment_20251011_145818/sae_results_layer_6.json'))
    
    metrics = [
        ("Final Train Loss", layer0_data["training_history"]["train_losses"][-1], 
         layer6_data["training_history"]["train_losses"][-1]),
        ("Final Val Loss", layer0_data["training_history"]["val_losses"][-1], 
         layer6_data["training_history"]["val_losses"][-1]),
        ("Loss Reduction", 
         layer0_data["training_history"]["train_losses"][0] / layer0_data["training_history"]["train_losses"][-1],
         layer6_data["training_history"]["train_losses"][0] / layer6_data["training_history"]["train_losses"][-1]),
        ("Final Sparsity", layer0_data["training_history"]["sparsity_rates"][-1], 
         layer6_data["training_history"]["sparsity_rates"][-1])
    ]
    
    for metric_name, layer0_val, layer6_val in metrics:
        if "Loss" in metric_name:
            better = "Layer 0" if layer0_val < layer6_val else "Layer 6"
        else:
            better = "Layer 0" if layer0_val > layer6_val else "Layer 6"
        
        print(f"{metric_name:<20} {layer0_val:<15.6f} {layer6_val:<15.6f} {better:<10}")

if __name__ == "__main__":
    analyze_sae_training()
