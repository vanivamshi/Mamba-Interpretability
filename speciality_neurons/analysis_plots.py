#!/usr/bin/env python3
"""
analysis_plots.py

Plots and tables for comparing Mamba vs Transformer models
using results from comprehensive neuron analyses.

Automatically saves all plots to disk instead of trying to display them.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # <- IMPORTANT for headless environments

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# To ensure pretty plots
sns.set(style="whitegrid", font_scale=1.2)


def ensure_plot_dir():
    os.makedirs("plots", exist_ok=True)


def save_plot(fig, filename):
    path = os.path.join("plots", filename)
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved plot: {path}")
    plt.close(fig)


def plot_specialty_neurons(results_mamba, results_transformer=None):
    mamba_data = []
    transformer_data = []

    classes = list(results_mamba["specialty_neurons"].keys())

    for cls in classes:
        neurons_mamba = results_mamba["specialty_neurons"][cls]
        for neuron_idx, score in neurons_mamba:
            mamba_data.append({
                "Class": cls,
                "Neuron": neuron_idx,
                "Score": score,
                "Model": "Mamba"
            })

        if results_transformer:
            neurons_tr = results_transformer["specialty_neurons"].get(cls, [])
            for neuron_idx, score in neurons_tr:
                transformer_data.append({
                    "Class": cls,
                    "Neuron": neuron_idx,
                    "Score": score,
                    "Model": "Transformer"
                })

    df_mamba = pd.DataFrame(mamba_data)

    if results_transformer:
        df_transformer = pd.DataFrame(transformer_data)
        df_all = pd.concat([df_mamba, df_transformer])
    else:
        df_all = df_mamba

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=df_all,
        x="Class",
        y="Score",
        hue="Model",
        errorbar=None,
        ax=ax
    )
    ax.set_title("Specialty Neuron Scores by Class")
    ax.set_ylabel("Specialization Score")
    ax.set_xlabel("Class")
    ax.legend(title="Model")

    save_plot(fig, "specialty_neurons.png")

    # Print table
    if results_transformer:
        table = df_all.groupby(["Class", "Model"])["Score"].mean().reset_index()
    else:
        table = df_mamba.groupby(["Class"])["Score"].mean().reset_index()

    print("\n=== Specialty Neurons Table ===")
    print(table)


def plot_delta_sensitive_neurons(results_mamba, results_transformer=None):
    neurons = [n for n, _ in results_mamba["delta_sensitivity"]["top_neurons"]]
    mamba_variances = [v for _, v in results_mamba["delta_sensitivity"]["top_neurons"]]

    data = pd.DataFrame({
        "Neuron": neurons,
        "Variance": mamba_variances,
        "Model": "Mamba"
    })

    if results_transformer:
        transformer_variances = []
        for n in neurons:
            variance = next((v for nn, v in results_transformer["delta_sensitivity"]["top_neurons"] if nn == n), 0)
            transformer_variances.append(variance)

        transformer_data = pd.DataFrame({
            "Neuron": neurons,
            "Variance": transformer_variances,
            "Model": "Transformer"
        })

        data = pd.concat([data, transformer_data])

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=data,
        x="Neuron",
        y="Variance",
        hue="Model",
        ax=ax
    )
    ax.set_title("Delta-sensitive Neuron Variance")
    ax.set_xlabel("Neuron Index")
    ax.set_ylabel("Variance")
    ax.legend(title="Model")

    save_plot(fig, "delta_sensitive_neurons.png")

    print("\n=== Delta-sensitive Neurons Table ===")
    print(data)


def plot_typo_robustness(results_mamba, results_transformer=None):
    mamba_scores = []
    for typo_type, stats in results_mamba["typo_robustness_summary"]["overall_consistency"].items():
        mamba_scores.append({
            "Typo Type": typo_type,
            "Mean Consistency": stats["mean"],
            "Std": stats["std"],
            "Model": "Mamba"
        })

    if results_transformer:
        transformer_scores = []
        for typo_type, stats in results_transformer["typo_robustness_summary"]["overall_consistency"].items():
            transformer_scores.append({
                "Typo Type": typo_type,
                "Mean Consistency": stats["mean"],
                "Std": stats["std"],
                "Model": "Transformer"
            })
        data = pd.DataFrame(mamba_scores + transformer_scores)
    else:
        data = pd.DataFrame(mamba_scores)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=data,
        x="Typo Type",
        y="Mean Consistency",
        hue="Model",
        errorbar=None,
        ax=ax
    )
    ax.set_ylim(0, 1)
    ax.set_title("Typographical Robustness (Mean Consistency)")

    save_plot(fig, "typo_robustness.png")

    print("\n=== Typographical Robustness Table ===")
    print(data)


def plot_template_robustness(results_mamba, results_transformer=None):
    rows = []
    for rel, rel_stats in results_mamba["template_robustness"].items():
        mean_corr = rel_stats.get("mean_fact_correlation", 0)
        std_corr = rel_stats.get("std_fact_correlation", 0)
        rows.append({
            "Relation": rel,
            "Mean Correlation": mean_corr,
            "Std": std_corr,
            "Model": "Mamba"
        })

    if results_transformer:
        for rel, rel_stats in results_transformer["template_robustness"].items():
            mean_corr = rel_stats.get("mean_fact_correlation", 0)
            std_corr = rel_stats.get("std_fact_correlation", 0)
            rows.append({
                "Relation": rel,
                "Mean Correlation": mean_corr,
                "Std": std_corr,
                "Model": "Transformer"
            })

    data = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=data,
        x="Relation",
        y="Mean Correlation",
        hue="Model",
        errorbar=None,
        ax=ax
    )
    ax.set_ylim(0, 1)
    ax.set_title("Template Robustness by Relation")

    save_plot(fig, "template_robustness.png")

    print("\n=== Template Robustness Table ===")
    print(data)


def plot_template_robustness_neurons_activated(results_mamba, results_transformer=None):
    """
    Plot number of neurons activated for template robustness by relation.
    Creates bar chart showing Mamba vs Transformer for P176, P463, P407.
    """
    # Define the relations we want to plot
    relations = ["P176", "P463", "P407"]
    
    # Create single figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Template Robustness: Neurons Activated by Relation", fontsize=14)
    
    # Prepare data for plotting
    mamba_data = []
    transformer_data = []
    
    for relation in relations:
        # Get template robustness data for this relation
        mamba_data.append({
            "Relation": relation,
            "Mean Neurons Activated": 0,  # Placeholder - replace with actual activation counts
            "Model": "Mamba"
        })
        
        if results_transformer:
            transformer_data.append({
                "Relation": relation,
                "Mean Neurons Activated": 0,  # Placeholder - replace with actual activation counts
                "Model": "Transformer"
            })
    
    # Combine data
    all_data = mamba_data + transformer_data
    data = pd.DataFrame(all_data)
    
    # Create bar plot
    sns.barplot(
        data=data,
        x="Relation",
        y="Mean Neurons Activated",
        hue="Model",
        errorbar=None,
        ax=ax
    )
    
    ax.set_xlabel("Relation Type", fontsize=12)
    ax.set_ylabel("Mean Number of Neurons Activated", fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', fontsize=10)
    
    plt.tight_layout()
    save_plot(fig, "template_robustness_neurons_activated.png")
    
    print("\n=== Template Robustness: Neurons Activated by Relation ===")
    for relation in relations:
        print(f"{relation}: Mamba=0, Transformer=0 (placeholder values - need actual activation data)")


def plot_typo_robustness_neurons_activated(results_mamba, results_transformer=None):
    """
    Plot number of neurons activated for typo robustness.
    Creates bar chart showing Mamba vs Transformer for realistic, random, contextual.
    """
    # Define the typo types
    typo_types = ["realistic", "random", "contextual"]
    
    # Create single figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Layer-wise Neurons Activated: Typo Robustness Analysis", fontsize=14)
    
    # Prepare data for plotting
    mamba_data = []
    transformer_data = []
    
    for typo_type in typo_types:
        # Get typo robustness data for this typo type
        mamba_data.append({
            "Typo Type": typo_type,
            "Mean Neurons Activated": 0,  # Placeholder - replace with actual activation counts
            "Model": "Mamba"
        })
        
        if results_transformer:
            transformer_data.append({
                "Typo Type": typo_type,
                "Mean Neurons Activated": 0,  # Placeholder - replace with actual activation counts
                "Model": "Transformer"
            })
    
    # Combine data
    all_data = mamba_data + transformer_data
    data = pd.DataFrame(all_data)
    
    # Create bar plot
    sns.barplot(
        data=data,
        x="Typo Type",
        y="Mean Neurons Activated",
        hue="Model",
        errorbar=None,
        ax=ax
    )
    
    ax.set_xlabel("Typo Type", fontsize=12)
    ax.set_ylabel("Mean Number of Neurons Activated", fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', fontsize=10)
    
    plt.tight_layout()
    save_plot(fig, "typo_robustness_neurons_activated.png")
    
    print("\n=== Typo Robustness: Neurons Activated Summary ===")
    for typo_type in typo_types:
        print(f"{typo_type}: Mamba=0, Transformer=0 (placeholder values - need actual activation data)")


def plot_specialty_neurons_activated(results_mamba, results_transformer=None):
    """
    Plot number of neurons activated for specialty neurons by class.
    Creates bar chart showing Mamba vs Transformer for factual, narrative, technical.
    """
    # Define the specialty classes
    specialty_classes = ["factual", "narrative", "technical"]
    
    # Create single figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Specialty Neurons: Neurons Activated by Class", fontsize=14)
    
    # Prepare data for plotting
    mamba_data = []
    transformer_data = []
    
    for specialty_class in specialty_classes:
        # Get specialty neurons for this class
        mamba_neurons = results_mamba.get("specialty_neurons", {}).get(specialty_class, [])
        if mamba_neurons:
            # Count neurons with activation > 0 (all non-zero activations)
            activated_neurons = sum(1 for _, score in mamba_neurons if abs(score) > 0)
            
            mamba_data.append({
                "Class": specialty_class.title(),
                "Mean Neurons Activated": activated_neurons,
                "Model": "Mamba"
            })
        
        if results_transformer:
            transformer_neurons = results_transformer.get("specialty_neurons", {}).get(specialty_class, [])
            if transformer_neurons:
                activated_neurons = sum(1 for _, score in transformer_neurons if abs(score) > 0)
                
                transformer_data.append({
                    "Class": specialty_class.title(),
                    "Mean Neurons Activated": activated_neurons,
                    "Model": "Transformer"
                })
    
    # Combine data
    all_data = mamba_data + transformer_data
    data = pd.DataFrame(all_data)
    
    # Create bar plot
    sns.barplot(
        data=data,
        x="Class",
        y="Mean Neurons Activated",
        hue="Model",
        errorbar=None,
        ax=ax
    )
    
    ax.set_xlabel("Specialty Class", fontsize=12)
    ax.set_ylabel("Mean Number of Neurons Activated", fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', fontsize=10)
    
    plt.tight_layout()
    save_plot(fig, "specialty_neurons_activated.png")
    
    print("\n=== Specialty Neurons: Neurons Activated by Class ===")
    for specialty_class in specialty_classes:
        mamba_neurons = results_mamba.get("specialty_neurons", {}).get(specialty_class, [])
        transformer_neurons = results_transformer.get("specialty_neurons", {}).get(specialty_class, []) if results_transformer else []
        
        # Calculate activated neurons with threshold 0
        mamba_activated = sum(1 for _, score in mamba_neurons if abs(score) > 0) if mamba_neurons else 0
        transformer_activated = sum(1 for _, score in transformer_neurons if abs(score) > 0) if transformer_neurons else 0
        
        print(f"{specialty_class}: Mamba={mamba_activated}, Transformer={transformer_activated}")


def run_all_plots(results_mamba, results_transformer=None):
    ensure_plot_dir()

    # Original plots (unchanged)
    plot_specialty_neurons(results_mamba, results_transformer)
    plot_delta_sensitive_neurons(results_mamba, results_transformer)
    plot_typo_robustness(results_mamba, results_transformer)
    plot_template_robustness(results_mamba, results_transformer)
    
    # New plots showing neurons activated
    plot_specialty_neurons_activated(results_mamba, results_transformer)
    plot_typo_robustness_neurons_activated(results_mamba, results_transformer)
    plot_template_robustness_neurons_activated(results_mamba, results_transformer)


if __name__ == "__main__":
    import json

    with open("mamba_results.json") as f:
        results_mamba = json.load(f)

    try:
        with open("transformer_results.json") as f:
            results_transformer = json.load(f)
    except FileNotFoundError:
        results_transformer = None

    run_all_plots(results_mamba, results_transformer)
