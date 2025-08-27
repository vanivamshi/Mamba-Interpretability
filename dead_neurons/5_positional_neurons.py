# 5_positional_neurons.py
# Run as - python3 5_positional_neurons.py
"""
Generates positional neuron scatter plots (Figure 8 style) for multiple models,
plotting all results in a single figure with subplots for each model and layer.
Includes ablation study and qualitative activation inspection.
Also logs all results to files for offline analysis and plotting.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import random # For selecting random neurons for qualitative inspection
import json
import datetime
from pathlib import Path

# Set plot style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Import necessary functions from other modules
try:
    from main import setup_model_and_tokenizer, load_analysis_texts
    from neuron_characterization import (
        find_positional_neurons,
        ablate_neurons_and_evaluate_perplexity, # New import
        collect_neuron_activations_for_visualization # New import
    )
    from visualization_module import (
        plot_multiple_positional_neurons_scatter,
        plot_positional_ablation_results_multi_model, # New import
        plot_neuron_activation_heatmap # New import
    )
    from utils import get_model_layers # Import get_model_layers to determine num_layers
    from delta_extraction import evaluate_perplexity # Import baseline perplexity evaluation
except ImportError as e:
    print(f"Import error: {e}. Ensure main.py, neuron_characterization.py, visualization_module.py, delta_extraction.py, and utils.py are in the same directory.")
    exit(1)

# Define the models to analyze with their full Hugging Face names and custom min_corr
models_to_analyze = {
    "Mamba-130M": {"name": "state-spaces/mamba-130m-hf", "min_corr": 0.12}, # Lower threshold for smaller model
    "Mamba-370M": {"name": "state-spaces/mamba-370m-hf", "min_corr": 0.12}, # Lower threshold for smaller model
    "Mamba-790M": {"name": "state-spaces/mamba-790m-hf", "min_corr": 0.12}, # Medium threshold for medium model
    "Mamba-1.4B": {"name": "state-spaces/mamba-1.4b-hf", "min_corr": 0.12}, # Medium threshold for larger model
    "Mamba-2.8B": {"name": "state-spaces/mamba-2.8b-hf", "min_corr": 0.12}, # Higher threshold for largest model
    "GPT-2": {"name": "gpt2", "min_corr": 0.3}, # Medium threshold for GPT-2
}

OUTPUT_DIR = "analysis_outputs"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")

# Create necessary directories
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def setup_logging():
    """Setup logging with timestamp and create log files."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Text log file for detailed analysis tracking
    log_file = os.path.join(LOGS_DIR, f"positional_neurons_analysis_{timestamp}.log")
    
    # JSON data file for offline plotting
    data_file = os.path.join(DATA_DIR, f"positional_neurons_data_{timestamp}.json")
    
    # Summary file for quick reference
    summary_file = os.path.join(LOGS_DIR, f"positional_neurons_summary_{timestamp}.txt")
    
    return log_file, data_file, summary_file, timestamp


def log_to_file(log_file, message):
    """Write message to log file with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")


def save_results_to_json(data_file, all_models_layers_correlations, all_models_layers_ablation_results, analysis_metadata):
    """Save all results to JSON file for offline plotting."""
    # Convert numpy arrays to lists for JSON serialization
    serializable_data = {}
    
    for plot_label, correlations in all_models_layers_correlations.items():
        if correlations:
            # Handle different correlation data formats
            if isinstance(correlations, dict):
                # Dictionary format (neuron_idx -> correlation_data)
                serializable_correlations = {}
                for neuron_idx, corr_data in correlations.items():
                    if isinstance(corr_data, dict):
                        serializable_correlations[str(neuron_idx)] = {
                            k: v.tolist() if isinstance(v, np.ndarray) else v 
                            for k, v in corr_data.items()
                        }
                    else:
                        serializable_correlations[str(neuron_idx)] = corr_data.tolist() if isinstance(corr_data, np.ndarray) else corr_data
            elif isinstance(correlations, list):
                # List format (avg_correlations from find_positional_neurons)
                # Convert to dictionary with neuron indices as keys
                serializable_correlations = {}
                for neuron_idx, corr_value in enumerate(correlations):
                    if isinstance(corr_value, np.ndarray):
                        serializable_correlations[str(neuron_idx)] = corr_value.tolist()
                    else:
                        serializable_correlations[str(neuron_idx)] = corr_value
            else:
                # Fallback: convert to list if it's a numpy array
                serializable_correlations = correlations.tolist() if isinstance(correlations, np.ndarray) else correlations
            
            serializable_data[plot_label] = serializable_correlations
    
    # Create complete data structure
    results_data = {
        "metadata": analysis_metadata,
        "timestamp": datetime.datetime.now().isoformat(),
        "models_analyzed": list(models_to_analyze.keys()),
        "results": serializable_data,
        "ablation_results": all_models_layers_ablation_results
    }
    
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    return results_data


def create_summary_report(summary_file, all_models_layers_correlations, all_models_layers_ablation_results, analysis_metadata):
    """Create a human-readable summary report."""
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("POSITIONAL NEURONS ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Analysis Date: {analysis_metadata['timestamp']}\n")
        f.write(f"Number of texts analyzed: {analysis_metadata['num_texts']}\n")
        f.write(f"Models analyzed: {', '.join(analysis_metadata['models_analyzed'])}\n\n")
        
        f.write("RESULTS SUMMARY:\n")
        f.write("-" * 20 + "\n")
        
        total_layers = 0
        total_neurons = 0
        
        for plot_label, correlations in all_models_layers_correlations.items():
            if correlations:
                num_neurons = len(correlations)
                total_neurons += num_neurons
                total_layers += 1
                f.write(f"{plot_label}: {num_neurons} positional neurons found\n")
        
        f.write(f"\nTotal layers with positional neurons: {total_layers}\n")
        f.write(f"Total positional neurons found: {total_neurons}\n")
        
        # Ablation results summary
        f.write(f"\nABLATION STUDY RESULTS:\n")
        f.write("-" * 25 + "\n")
        
        ablation_count = 0
        for plot_label, ablation_data in all_models_layers_ablation_results.items():
            if ablation_data["baseline_ppl"] is not None:
                ablation_count += 1
                baseline = ablation_data["baseline_ppl"]
                ablated = ablation_data["ablated_ppl"]
                num_neurons = ablation_data["num_ablated_neurons"]
                f.write(f"{plot_label}: Baseline={baseline:.3f}, Ablated={ablated:.3f}, Neurons={num_neurons}\n")
        
        f.write(f"\nTotal ablation studies completed: {ablation_count}\n")
        
        if total_layers == 0:
            f.write("\n⚠️ No positional neurons found in any layer.\n")


def main():
    print("\n🚀 Starting Positional Neuron Analysis, Ablation, and Qualitative Inspection for Multiple Models (All Layers)")
    
    # Setup logging
    log_file, data_file, summary_file, timestamp = setup_logging()
    
    # Log analysis start
    log_to_file(log_file, "Starting positional neuron analysis with ablation and qualitative inspection")
    log_to_file(log_file, f"Models to analyze: {list(models_to_analyze.keys())}")

    # Load a subset of texts for analysis (adjust as needed)
    # Using 200 texts for a reasonable runtime, you can increase this for more thorough analysis
    texts = load_analysis_texts(200)
    # Use a smaller subset for perplexity evaluation to speed up ablation
    ablation_eval_texts = load_analysis_texts(50) 
    # Select one specific text for detailed qualitative inspection
    qualitative_text_sample = texts[0] if texts else "" 
    
    log_to_file(log_file, f"Loaded {len(texts)} texts for analysis, {len(ablation_eval_texts)} for ablation evaluation")

    # Dictionaries to store results for plotting
    all_models_layers_correlations = {} # For scatter plots
    all_models_layers_ablation_results = {} # For ablation bar plots
    
    # Metadata for analysis
    analysis_metadata = {
        "timestamp": timestamp,
        "num_texts": len(texts),
        "num_ablation_texts": len(ablation_eval_texts),
        "models_analyzed": [],
        "analysis_parameters": {
            "text_subset_size": 200,
            "ablation_text_subset_size": 50,
            "correlation_thresholds": {model: info["min_corr"] for model, info in models_to_analyze.items()}
        }
    }

    for model_label, model_info in models_to_analyze.items():
        model_name = model_info["name"]
        min_corr_threshold = model_info["min_corr"]
        print(f"\n🔍 Analyzing model: {model_label} (min_corr={min_corr_threshold})")
        log_to_file(log_file, f"Starting analysis for model: {model_label} ({model_name}) with min_corr={min_corr_threshold}")
        
        analysis_metadata["models_analyzed"].append(model_label)

        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer(model_name)
        log_to_file(log_file, f"Model and tokenizer loaded for {model_label}")

        # Dynamically determine the number of layers for the current model
        layers_obj = get_model_layers(model)
        if layers_obj is None:
            num_layers = getattr(model.config, "num_hidden_layers", None) or \
                         getattr(model.config, "n_layer", None)
            if num_layers is None:
                error_msg = f"Could not determine number of layers for {model_label}. Skipping."
                print(error_msg)
                log_to_file(log_file, f"ERROR: {error_msg}")
                continue
        else:
            num_layers = len(layers_obj)
        
        log_to_file(log_file, f"Model {model_label} has {num_layers} layers")

        # OLD CODE (commented out): Run analysis for all layers
        # for layer_idx in range(num_layers):
        #     print(f"  Analyzing Layer {layer_idx}/{num_layers-1}")
        #     log_to_file(log_file, f"Analyzing layer {layer_idx}/{num_layers-1} for {model_label}")
        # 
        #     # --- 1. Positional Neuron Identification ---
        #     positional_neurons, correlations = find_positional_neurons(
        #         model, tokenizer, texts, layer_idx=layer_idx, min_corr=min_corr_threshold
        #     )
        # 
        #     plot_label = f"{model_label} - Layer {layer_idx} (Corr > {min_corr_threshold})"
        #     if correlations:
        #         all_models_layers_correlations[plot_label] = correlations
        #         num_neurons = len(correlations)
        #         success_msg = f"Collected positional neuron data for {plot_label}: {num_neurons} neurons"
        #         print(f"  ✅ {success_msg}.")
        #         log_to_file(log_file, f"SUCCESS: {success_msg}")
        #     else:
        #         warning_msg = f"No positional neuron correlations found for {plot_label}. Skipping data collection."
        #         print(f"  ⚠️ {warning_msg}")
        #         log_to_file(log_file, f"WARNING: {warning_msg}")
        # 
        #     # --- 2. Ablation Study ---
        #     if positional_neurons:
        #         print(f"  Running ablation study for {len(positional_neurons)} positional neurons...")
        #         log_to_file(log_file, f"Running ablation study for {len(positional_neurons)} neurons in {plot_label}")
        #         
        #         # Calculate baseline perplexity
        #         baseline_ppl = evaluate_perplexity(model, tokenizer, ablation_eval_texts, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        #         print(f"    Baseline Perplexity: {baseline_ppl:.3f}")
        #         log_to_file(log_file, f"Baseline perplexity for {plot_label}: {baseline_ppl:.3f}")
        # 
        #         # Ablate positional neurons and evaluate perplexity
        #         ablated_ppl = ablate_neurons_and_evaluate_perplexity(
        #             model, tokenizer, ablation_eval_texts, layer_idx, positional_neurons
        #         )
        #         print(f"    Ablated Perplexity: {ablated_ppl:.3f}")
        #         log_to_file(log_file, f"Ablated perplexity for {plot_label}: {ablated_ppl:.3f}")
        # 
        #         all_models_layers_ablation_results[plot_label] = {
        #             "baseline_ppl": baseline_ppl,
        #             "ablated_ppl": ablated_ppl,
        #             "num_ablated_neurons": len(positional_neurons)
        #         }
        #         success_msg = f"Ablation results collected for {plot_label}"
        #         print(f"  ✅ {success_msg}.")
        #         log_to_file(log_file, f"SUCCESS: {success_msg}")
        #     else:
        #         skip_msg = f"Skipping ablation study for {plot_label}: No positional neurons found."
        #         print(f"  {skip_msg}")
        #         log_to_file(log_file, f"SKIP: {skip_msg}")
        #         all_models_layers_ablation_results[plot_label] = {
        #             "baseline_ppl": None, "ablated_ppl": None, "num_ablated_neurons": 0
        #         } # Store None to indicate no ablation was done

        # NEW CODE: Run analysis only for layer 0
        layer_idx = 0  # Only analyze layer 0
        print(f"  Analyzing Layer {layer_idx} (Layer 0 only)")
        log_to_file(log_file, f"Analyzing layer {layer_idx} for {model_label} (Layer 0 only)")

        # --- 1. Positional Neuron Identification ---
        positional_neurons, correlations = find_positional_neurons(
            model, tokenizer, texts, layer_idx=layer_idx, min_corr=min_corr_threshold
        )

        plot_label = f"{model_label} (Corr > {min_corr_threshold})"
        if correlations:
            all_models_layers_correlations[plot_label] = correlations
            num_neurons = len(correlations)
            success_msg = f"Collected positional neuron data for {plot_label}: {num_neurons} neurons (Layer 0 only)"
            print(f"  ✅ {success_msg}.")
            log_to_file(log_file, f"SUCCESS: {success_msg}")
        else:
            warning_msg = f"No positional neuron correlations found for {plot_label}. Skipping data collection."
            print(f"  ⚠️ {warning_msg}")
            log_to_file(log_file, f"WARNING: {warning_msg}")

        # --- 2. Ablation Study ---
        if positional_neurons:
            print(f"  Running ablation study for {len(positional_neurons)} positional neurons...")
            log_to_file(log_file, f"Running ablation study for {len(positional_neurons)} neurons in {plot_label}")
            
            # Calculate baseline perplexity
            baseline_ppl = evaluate_perplexity(model, tokenizer, ablation_eval_texts, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            print(f"    Baseline Perplexity: {baseline_ppl:.3f}")
            log_to_file(log_file, f"Baseline perplexity for {plot_label}: {baseline_ppl:.3f}")

            # Ablate positional neurons and evaluate perplexity
            ablated_ppl = ablate_neurons_and_evaluate_perplexity(
                model, tokenizer, ablation_eval_texts, layer_idx, positional_neurons
            )
            print(f"    Ablated Perplexity: {ablated_ppl:.3f}")
            log_to_file(log_file, f"Ablated perplexity for {plot_label}: {ablated_ppl:.3f}")

            all_models_layers_ablation_results[plot_label] = {
                "baseline_ppl": baseline_ppl,
                "ablated_ppl": ablated_ppl,
                "num_ablated_neurons": len(positional_neurons)
            }
            success_msg = f"Ablation results collected for {plot_label} (Layer 0 only)"
            print(f"  ✅ {success_msg}.")
            log_to_file(log_file, f"SUCCESS: {success_msg}")
        else:
            skip_msg = f"Skipping ablation study for {plot_label}: No positional neurons found."
            print(f"  {skip_msg}")
            log_to_file(log_file, f"SKIP: {skip_msg}")
            all_models_layers_ablation_results[plot_label] = {
                "baseline_ppl": None, "ablated_ppl": None, "num_ablated_neurons": 0
            } # Store None to indicate no ablation was done

        # --- 3. Qualitative Inspection (Heatmap) ---
        # OLD CODE (commented out): Plot heatmaps for all layers
        # if positional_neurons and qualitative_text_sample:
        #     # Select a few random positional neurons for detailed visualization
        #     num_neurons_to_plot = min(5, len(positional_neurons)) # Plot up to 5 neurons
        #     neurons_for_heatmap = random.sample(positional_neurons, num_neurons_to_plot) if num_neurons_to_plot > 0 else []
        # 
        #     if neurons_for_heatmap:
        #         print(f"  Collecting activations for qualitative inspection of neurons {neurons_for_heatmap}...")
        #         log_to_file(log_file, f"Collecting activations for qualitative inspection of neurons {neurons_for_heatmap} in {plot_label}")
        #         
        #         activations, tokens = collect_neuron_activations_for_visualization(
        #             model, tokenizer, qualitative_text_sample, layer_idx, neurons_for_heatmap
        #         )
        #         if activations.size > 0:
        #             plot_neuron_activation_heatmap(
        #                 activations, tokens, neurons_for_heatmap,
        #                 model_label, layer_idx, 0, PLOTS_DIR # 0 is sample text index
        #             )
        #             success_msg = f"Heatmap generated for {plot_label}"
        #             print(f"  ✅ {success_msg}.")
        #             log_to_file(log_file, f"SUCCESS: {success_msg}")
        #         else:
        #             warning_msg = f"No activation data collected for heatmap for {plot_label}"
        #             print(f"  ⚠️ {warning_msg}.")
        #             log_to_file(log_file, f"WARNING: {warning_msg}")
        #     else:
        #         skip_msg = f"Skipping heatmap for {plot_label}: No positional neurons to plot or text sample missing"
        #         print(f"  {skip_msg}.")
        #         log_to_file(log_file, f"SKIP: {skip_msg}")
        # else:
        #     skip_msg = f"Skipping heatmap for {plot_label}: No positional neurons found or text sample missing"
        #     print(f"  {skip_msg}.")
        #     log_to_file(log_file, f"SKIP: {skip_msg}")

        # NEW CODE: Plot heatmaps for layer 0 (since we're only analyzing layer 0)
        print(f"  DEBUG: positional_neurons = {positional_neurons}")
        print(f"  DEBUG: qualitative_text_sample exists = {bool(qualitative_text_sample)}")
        print(f"  DEBUG: len(positional_neurons) = {len(positional_neurons) if positional_neurons else 0}")
        
        if positional_neurons and qualitative_text_sample:
            print(f"  DEBUG: Entering heatmap generation block")
            # Select a few random positional neurons for detailed visualization
            num_neurons_to_plot = min(5, len(positional_neurons)) # Plot up to 5 neurons
            neurons_for_heatmap = random.sample(positional_neurons, num_neurons_to_plot) if num_neurons_to_plot > 0 else []
            print(f"  DEBUG: neurons_for_heatmap = {neurons_for_heatmap}")

            if neurons_for_heatmap:
                print(f"  Collecting activations for qualitative inspection of neurons {neurons_for_heatmap} (Layer 0 only)...")
                log_to_file(log_file, f"Collecting activations for qualitative inspection of neurons {neurons_for_heatmap} in {plot_label} (Layer 0 only)")
                
                try:
                    activations, tokens = collect_neuron_activations_for_visualization(
                        model, tokenizer, qualitative_text_sample, layer_idx, neurons_for_heatmap
                    )
                    print(f"  DEBUG: activations.shape = {activations.shape if hasattr(activations, 'shape') else 'No shape'}")
                    print(f"  DEBUG: tokens = {tokens[:10] if tokens else 'No tokens'}")  # Show first 10 tokens
                    
                    if activations.size > 0:
                        print(f"  DEBUG: Calling plot_neuron_activation_heatmap")
                        plot_neuron_activation_heatmap(
                            activations, tokens, neurons_for_heatmap,
                            model_label, layer_idx, 0, PLOTS_DIR # 0 is sample text index
                        )
                        success_msg = f"Heatmap generated for {plot_label} (Layer 0 only)"
                        print(f"  ✅ {success_msg}.")
                        log_to_file(log_file, f"SUCCESS: {success_msg}")
                    else:
                        warning_msg = f"No activation data collected for heatmap for {plot_label}"
                        print(f"  ⚠️ {warning_msg}.")
                        log_to_file(log_file, f"WARNING: {warning_msg}")
                except Exception as e:
                    error_msg = f"Error in heatmap generation for {plot_label}: {str(e)}"
                    print(f"  ❌ {error_msg}")
                    log_to_file(log_file, f"ERROR: {error_msg}")
            else:
                skip_msg = f"Skipping heatmap for {plot_label}: No positional neurons to plot or text sample missing"
                print(f"  {skip_msg}.")
                log_to_file(log_file, f"SKIP: {skip_msg}")
        else:
            skip_msg = f"Skipping heatmap for {plot_label}: No positional neurons found or text sample missing"
            print(f"  {skip_msg}.")
            log_to_file(log_file, f"SKIP: {skip_msg}")

    # Log analysis completion
    log_to_file(log_file, f"Analysis completed. Found data for {len(all_models_layers_correlations)} model-layer combinations")
    
    # Save results to JSON for offline plotting
    results_data = save_results_to_json(data_file, all_models_layers_correlations, all_models_layers_ablation_results, analysis_metadata)
    log_to_file(log_file, f"Results saved to JSON file: {data_file}")
    
    # Create summary report
    create_summary_report(summary_file, all_models_layers_correlations, all_models_layers_ablation_results, analysis_metadata)
    log_to_file(log_file, f"Summary report created: {summary_file}")

    # --- Final Plotting ---
    if all_models_layers_correlations:
        plot_path_scatter = plot_multiple_positional_neurons_scatter(
            all_models_layers_correlations,
            "All Layers", # Identifier for the combined plot title
            PLOTS_DIR
        )
        success_msg = f"Combined positional neuron scatter plot for all layers saved to: {plot_path_scatter}"
        print(f"\n🎉 {success_msg}")
        log_to_file(log_file, f"PLOT CREATED: {success_msg}")
    else:
        error_msg = "No positional neuron data collected across any models or layers to plot scatter."
        print(f"\n❌ {error_msg}")
        log_to_file(log_file, f"ERROR: {error_msg}")

    if all_models_layers_ablation_results:
        plot_path_ablation = plot_positional_ablation_results_multi_model(
            all_models_layers_ablation_results,
            "All Layers", # Identifier for the combined plot title
            PLOTS_DIR
        )
        success_msg = f"Combined positional neuron ablation plot for all layers saved to: {plot_path_ablation}"
        print(f"\n🎉 {success_msg}")
        log_to_file(log_file, f"PLOT CREATED: {success_msg}")
    else:
        error_msg = "No positional neuron ablation data collected across any models or layers to plot."
        print(f"\n❌ {error_msg}")
        log_to_file(log_file, f"ERROR: {error_msg}")

    # Final summary
    log_to_file(log_file, "Positional neuron analysis complete!")
    print(f"\n✅ Positional neuron analysis complete!")
    print(f"📊 Results saved to: {data_file}")
    print(f"📝 Analysis log: {log_file}")
    print(f"📋 Summary report: {summary_file}")
    if all_models_layers_correlations:
        print(f"🖼️  Scatter plot saved to: {plot_path_scatter}")
    if all_models_layers_ablation_results:
        print(f"🖼️  Ablation plot saved to: {plot_path_ablation}")


if __name__ == "__main__":
    main()
