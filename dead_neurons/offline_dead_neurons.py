#!/usr/bin/env python3
"""
Offline plotting script for dead neurons analysis.
Loads data from logs created by 2_analyze_dead_neurons.py and creates plots.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def load_dead_neurons_log(log_file_path):
    """Load dead neurons analysis log file."""
    with open(log_file_path, 'r') as f:
        data = json.load(f)
    return data


def plot_dead_neurons_from_log(log_data, save_path="offline_dead_neurons.png"):
    """Create dead neurons plot from logged data."""
    plt.figure(figsize=(12, 8))
    
    for model_label, model_data in log_data["models_data"].items():
        layers = sorted(model_data.keys(), key=int)
        dead_ratios = [model_data[layer] for layer in layers]
        plt.plot(layers, dead_ratios, marker='o', linewidth=2, markersize=8, label=model_label)
    
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Dead Neuron Ratio", fontsize=12)
    plt.title("Dead Neuron Distribution Across Layers", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Dead neurons plot saved: {save_path}")


def plot_dead_neurons_by_model(log_data, save_dir="offline_plots"):
    """Create individual plots for each model."""
    os.makedirs(save_dir, exist_ok=True)
    
    for model_label, model_data in log_data["models_data"].items():
        plt.figure(figsize=(10, 6))
        
        layers = sorted(model_data.keys(), key=int)
        dead_ratios = [model_data[layer] for layer in layers]
        
        plt.plot(layers, dead_ratios, marker='o', linewidth=2, markersize=8, color='red')
        plt.fill_between(layers, dead_ratios, alpha=0.3, color='red')
        
        plt.xlabel("Layer", fontsize=12)
        plt.ylabel("Dead Neuron Ratio", fontsize=12)
        plt.title(f"Dead Neuron Distribution - {model_label}", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Clean filename
        safe_model_name = model_label.replace(" ", "_").replace("-", "_")
        save_path = os.path.join(save_dir, f"dead_neurons_{safe_model_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Individual plot saved: {save_path}")


def print_analysis_summary(log_data):
    """Print summary statistics from the logged data."""
    print("\n" + "="*60)
    print("DEAD NEURONS ANALYSIS SUMMARY")
    print("="*60)
    print(f"Analysis timestamp: {log_data['timestamp']}")
    print(f"Device used: {log_data['device']}")
    print(f"Sample text length: {log_data['sample_text_length']} characters")
    print(f"Models analyzed: {len(log_data['models_data'])}")
    
    print("\nDead Neuron Statistics:")
    print("-" * 40)
    
    for model_label, model_data in log_data["models_data"].items():
        layers = sorted(model_data.keys(), key=int)
        dead_ratios = [model_data[layer] for layer in layers]
        
        avg_dead_ratio = np.mean(dead_ratios)
        max_dead_ratio = np.max(dead_ratios)
        min_dead_ratio = np.min(dead_ratios)
        
        print(f"{model_label}:")
        print(f"  Average dead ratio: {avg_dead_ratio:.3f} ({avg_dead_ratio*100:.1f}%)")
        print(f"  Max dead ratio: {max_dead_ratio:.3f} ({max_dead_ratio*100:.1f}%)")
        print(f"  Min dead ratio: {min_dead_ratio:.3f} ({min_dead_ratio*100:.1f}%)")
        print(f"  Layers analyzed: {len(layers)}")
        print()


def main():
    """Main function for offline dead neurons plotting."""
    logs_dir = "logs"
    
    if not os.path.exists(logs_dir):
        print(f"❌ Logs directory '{logs_dir}' not found.")
        print("Please run 2_analyze_dead_neurons.py first to generate log files.")
        return
    
    # Find dead neurons log files
    dead_neuron_logs = [f for f in os.listdir(logs_dir) if f.startswith("dead_neurons_analysis_")]
    
    if not dead_neuron_logs:
        print("❌ No dead neurons log files found.")
        print("Please run 2_analyze_dead_neurons.py first to generate log files.")
        return
    
    # Use the most recent log file
    latest_log = max(dead_neuron_logs)
    log_path = os.path.join(logs_dir, latest_log)
    print(f"📊 Loading dead neurons data from: {log_path}")
    
    try:
        log_data = load_dead_neurons_log(log_path)
        
        # Print summary
        print_analysis_summary(log_data)
        
        # Create plots
        plot_dead_neurons_from_log(log_data)
        plot_dead_neurons_by_model(log_data)
        
        print("\n🎉 Offline dead neurons analysis complete!")
        print("📁 Check the current directory and 'offline_plots/' for generated plots.")
        
    except Exception as e:
        print(f"❌ Error processing log file: {e}")
        print("Please check if the log file is corrupted or incomplete.")


if __name__ == "__main__":
    main() 