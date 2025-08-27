#!/usr/bin/env python3
"""
Offline plotting script for n-gram analysis.
Loads data from logs created by 4_ngram_analysis.py and creates plots.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def load_ngram_analysis_log(log_file_path):
    """Load n-gram analysis log file."""
    with open(log_file_path, 'r') as f:
        data = json.load(f)
    return data


def plot_figure_3a_from_log(log_data, save_path="offline_figure_3a.png"):
    """Create Figure 3a: Token-detecting neurons per layer."""
    plt.figure(figsize=(10, 6))
    
    for model_label, res in log_data["results"].items():
        num_layers = len(res["layer_triggers"])
        depths = [l/num_layers for l in range(num_layers)]
        per_layer_neurons = []
        
        for l in range(num_layers):
            layer_str = str(l)
            if layer_str in res["layer_triggers"]:
                count = sum(1 for n in res["layer_triggers"][layer_str] 
                           if "1" in res["layer_triggers"][layer_str][n] 
                           and len(res["layer_triggers"][layer_str][n]["1"]) > 0)
                per_layer_neurons.append(count)
            else:
                per_layer_neurons.append(0)
        
        plt.plot(depths, per_layer_neurons, marker="o", linewidth=2, markersize=6, label=model_label)
    
    plt.xlabel("Layer (relative depth)", fontsize=12)
    plt.ylabel("# Token-detecting neurons", fontsize=12)
    plt.title("Figure 3a – Token-detecting neurons", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Figure 3a saved: {save_path}")


def plot_figure_3b_from_log(log_data, save_path="offline_figure_3b.png"):
    """Create Figure 3b: Cumulative token coverage."""
    plt.figure(figsize=(10, 6))
    
    for model_label, res in log_data["results"].items():
        num_layers = len(res["layer_triggers"])
        depths = [l/num_layers for l in range(num_layers)]
        cumulative_tokens = []
        seen = set()
        
        for l in range(num_layers):
            layer_str = str(l)
            if layer_str in res["layer_triggers"]:
                tokens = set()
                for n in res["layer_triggers"][layer_str]:
                    if "1" in res["layer_triggers"][layer_str][n]:
                        for ngram in res["layer_triggers"][layer_str][n]["1"]:
                            if len(ngram) == 1:
                                tokens.add(ngram[0])
                seen |= tokens
                cumulative_tokens.append(len(seen))
            else:
                cumulative_tokens.append(len(seen))
        
        plt.plot(depths, cumulative_tokens, linestyle="--", marker="o", linewidth=2, markersize=6, label=f"{model_label} cumulative")
    
    plt.xlabel("Layer (relative depth)", fontsize=12)
    plt.ylabel("# Tokens covered", fontsize=12)
    plt.title("Figure 3b – Token coverage", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Figure 3b saved: {save_path}")


def plot_figure_4_from_log(log_data, save_path="offline_figure_4.png"):
    """Create Figure 4: New token coverage."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for model_label, res in log_data["results"].items():
        num_layers = len(res["layer_triggers"])
        depths = [l/num_layers for l in range(num_layers)]
        new_overall = []
        new_vs_prev = []
        seen = set()
        prev = set()
        
        for l in range(num_layers):
            layer_str = str(l)
            if layer_str in res["layer_triggers"]:
                tokens = set()
                for n in res["layer_triggers"][layer_str]:
                    if "1" in res["layer_triggers"][layer_str][n]:
                        for ngram in res["layer_triggers"][layer_str][n]["1"]:
                            if len(ngram) == 1:
                                tokens.add(ngram[0])
                new_overall.append(len(tokens - seen))
                new_vs_prev.append(len(tokens - prev))
                seen |= tokens
                prev = tokens
            else:
                new_overall.append(0)
                new_vs_prev.append(0)
        
        ax1.plot(depths, new_overall, marker="o", linewidth=2, markersize=6, label=f"{model_label} new overall")
        ax2.plot(depths, new_vs_prev, marker="s", linestyle="--", linewidth=2, markersize=6, label=f"{model_label} new vs prev")
    
    ax1.set_xlabel("Layer (relative depth)", fontsize=12)
    ax1.set_ylabel("# New Tokens", fontsize=12)
    ax1.set_title("New token coverage (overall)", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    ax2.set_xlabel("Layer (relative depth)", fontsize=12)
    ax2.set_ylabel("# New Tokens", fontsize=12)
    ax2.set_title("New token coverage (vs prev)", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Figure 4 saved: {save_path}")


def plot_comprehensive_analysis(log_data, save_path="offline_comprehensive_ngram.png"):
    """Create comprehensive n-gram analysis plot with all figures."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Figure 3a: Token-detecting neurons per layer
    for model_label, res in log_data["results"].items():
        num_layers = len(res["layer_triggers"])
        depths = [l/num_layers for l in range(num_layers)]
        per_layer_neurons = []
        
        for l in range(num_layers):
            layer_str = str(l)
            if layer_str in res["layer_triggers"]:
                count = sum(1 for n in res["layer_triggers"][layer_str] 
                           if "1" in res["layer_triggers"][layer_str][n] 
                           and len(res["layer_triggers"][layer_str][n]["1"]) > 0)
                per_layer_neurons.append(count)
            else:
                per_layer_neurons.append(0)
        
        axes[0, 0].plot(depths, per_layer_neurons, marker="o", linewidth=2, markersize=6, label=model_label)
    
    axes[0, 0].set_xlabel("Layer (relative depth)", fontsize=12)
    axes[0, 0].set_ylabel("# Token-detecting neurons", fontsize=12)
    axes[0, 0].set_title("Figure 3a – Token-detecting neurons", fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=10)
    
    # Figure 3b: Cumulative token coverage
    for model_label, res in log_data["results"].items():
        num_layers = len(res["layer_triggers"])
        depths = [l/num_layers for l in range(num_layers)]
        cumulative_tokens = []
        seen = set()
        
        for l in range(num_layers):
            layer_str = str(l)
            if layer_str in res["layer_triggers"]:
                tokens = set()
                for n in res["layer_triggers"][layer_str]:
                    if "1" in res["layer_triggers"][layer_str][n]:
                        for ngram in res["layer_triggers"][layer_str][n]["1"]:
                            if len(ngram) == 1:
                                tokens.add(ngram[0])
                seen |= tokens
                cumulative_tokens.append(len(seen))
            else:
                cumulative_tokens.append(len(seen))
        
        axes[0, 1].plot(depths, cumulative_tokens, linestyle="--", marker="o", linewidth=2, markersize=6, label=f"{model_label} cumulative")
    
    axes[0, 1].set_xlabel("Layer (relative depth)", fontsize=12)
    axes[0, 1].set_ylabel("# Tokens covered", fontsize=12)
    axes[0, 1].set_title("Figure 3b – Token coverage", fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=10)
    
    # Figure 4a: New token coverage (overall)
    for model_label, res in log_data["results"].items():
        num_layers = len(res["layer_triggers"])
        depths = [l/num_layers for l in range(num_layers)]
        new_overall = []
        seen = set()
        
        for l in range(num_layers):
            layer_str = str(l)
            if layer_str in res["layer_triggers"]:
                tokens = set()
                for n in res["layer_triggers"][layer_str]:
                    if "1" in res["layer_triggers"][layer_str][n]:
                        for ngram in res["layer_triggers"][layer_str][n]["1"]:
                            if len(ngram) == 1:
                                tokens.add(ngram[0])
                new_overall.append(len(tokens - seen))
                seen |= tokens
            else:
                new_overall.append(0)
        
        axes[1, 0].plot(depths, new_overall, marker="o", linewidth=2, markersize=6, label=f"{model_label} new overall")
    
    axes[1, 0].set_xlabel("Layer (relative depth)", fontsize=12)
    axes[1, 0].set_ylabel("# New Tokens", fontsize=12)
    axes[1, 0].set_title("Figure 4a – New token coverage (overall)", fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=10)
    
    # Figure 4b: New token coverage (vs prev)
    for model_label, res in log_data["results"].items():
        num_layers = len(res["layer_triggers"])
        depths = [l/num_layers for l in range(num_layers)]
        new_vs_prev = []
        prev = set()
        
        for l in range(num_layers):
            layer_str = str(l)
            if layer_str in res["layer_triggers"]:
                tokens = set()
                for n in res["layer_triggers"][layer_str]:
                    if "1" in res["layer_triggers"][layer_str][n]:
                        for ngram in res["layer_triggers"][layer_str][n]["1"]:
                            if len(ngram) == 1:
                                tokens.add(ngram[0])
                new_vs_prev.append(len(tokens - prev))
                prev = tokens
            else:
                new_vs_prev.append(0)
        
        axes[1, 1].plot(depths, new_vs_prev, marker="s", linestyle="--", linewidth=2, markersize=6, label=f"{model_label} new vs prev")
    
    axes[1, 1].set_xlabel("Layer (relative depth)", fontsize=12)
    axes[1, 1].set_ylabel("# New Tokens", fontsize=12)
    axes[1, 1].set_title("Figure 4b – New token coverage (vs prev)", fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Comprehensive analysis saved: {save_path}")


def print_analysis_summary(log_data):
    """Print summary statistics from the logged data."""
    print("\n" + "="*60)
    print("N-GRAM ANALYSIS SUMMARY")
    print("="*60)
    print(f"Analysis timestamp: {log_data['timestamp']}")
    print(f"Models analyzed: {len(log_data['results'])}")
    print(f"Models: {', '.join(log_data['models_analyzed'])}")
    
    print("\nAnalysis Statistics:")
    print("-" * 40)
    
    for model_label, res in log_data["results"].items():
        num_layers = len(res["layer_triggers"])
        total_neurons = 0
        total_dead_neurons = 0
        
        for layer_str in res["layer_triggers"]:
            layer_neurons = len(res["layer_triggers"][layer_str])
            total_neurons += layer_neurons
            
            if layer_str in res["dead_neurons"]:
                total_dead_neurons += len(res["dead_neurons"][layer_str])
        
        print(f"{model_label}:")
        print(f"  Layers analyzed: {num_layers}")
        print(f"  Total neurons: {total_neurons}")
        print(f"  Total dead neurons: {total_dead_neurons}")
        print(f"  Dead neuron ratio: {total_dead_neurons/total_neurons:.3f} ({total_dead_neurons/total_neurons*100:.1f}%)")
        print()


def main():
    """Main function for offline n-gram analysis plotting."""
    logs_dir = "logs"
    
    if not os.path.exists(logs_dir):
        print(f"❌ Logs directory '{logs_dir}' not found.")
        print("Please run 4_ngram_analysis.py first to generate log files.")
        return
    
    # Find n-gram analysis log files
    ngram_logs = [f for f in os.listdir(logs_dir) if f.startswith("ngram_analysis_")]
    
    if not ngram_logs:
        print("❌ No n-gram analysis log files found.")
        print("Please run 4_ngram_analysis.py first to generate log files.")
        return
    
    # Use the most recent log file
    latest_log = max(ngram_logs)
    log_path = os.path.join(logs_dir, latest_log)
    print(f"📊 Loading n-gram analysis data from: {log_path}")
    
    try:
        log_data = load_ngram_analysis_log(log_path)
        
        # Print summary
        print_analysis_summary(log_data)
        
        # Create individual plots
        plot_figure_3a_from_log(log_data)
        plot_figure_3b_from_log(log_data)
        plot_figure_4_from_log(log_data)
        
        # Create comprehensive plot
        plot_comprehensive_analysis(log_data)
        
        print("\n🎉 Offline n-gram analysis complete!")
        print("📁 Check the current directory for generated plots.")
        
    except Exception as e:
        print(f"❌ Error processing log file: {e}")
        print("Please check if the log file is corrupted or incomplete.")


if __name__ == "__main__":
    main() 