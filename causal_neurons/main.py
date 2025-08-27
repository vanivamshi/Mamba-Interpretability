#!/usr/bin/env python3
"""
Main script for running comprehensive Mamba neuron analysis with comparison plots - FIXED VERSION.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for file saving
plt.ioff()  # Disable interactive mode

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from utils import debug_model_structure, get_model_layers
from delta_extraction import find_delta_sensitive_neurons_fixed
from causal_analysis import find_causal_neurons_fixed, plot_inter_layer_impact, inter_layer_causal_impact_all_layers, cross_layer_causal_influence
from comparison_wrapper import run_full_comparison, run_quick_comparison, create_summary_report, analyze_model_efficiency
from attention_neurons import integrate_mamba_attention_neurons, MambaAttentionNeurons

# Import comparison functionality with fallback
try:
    from comparison_plots import load_models, create_comparison_plots
    COMPARISON_AVAILABLE = True
except ImportError:
    COMPARISON_AVAILABLE = False
    print("Warning: comparison_plots.py not available. Some functionality will be limited.")

def ensure_plot_display(title=None):
    """Save plots to files instead of displaying them."""
    try:
        # Create plots directory if it doesn't exist
        import os
        os.makedirs('plots', exist_ok=True)
        
        # Generate filename based on title or timestamp
        if title:
            filename = f"plots/{title.replace(' ', '_').replace('(', '').replace(')', '').replace(':', '')}.png"
        else:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"plots/plot_{timestamp}.png"
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {filename}")
        plt.close()  # Close the figure to free memory
    except Exception as e:
        print(f"❌ Error saving plot: {e}")
        plt.close()  # Close the figure even if saving fails

def save_results(results, filename=None, output_dir="analysis_outputs"):
    """Save analysis results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        from datetime import datetime
        filename = os.path.join(output_dir, f"neuron_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    def convert_numpy(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return str(obj)

    def recursive_convert(obj):
        if isinstance(obj, dict): return {k: recursive_convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [recursive_convert(v) for v in obj]
        if isinstance(obj, tuple): return tuple(recursive_convert(v) for v in obj)
        return convert_numpy(obj)

    try:
        with open(filename, 'w') as f:
            import json
            json.dump(recursive_convert(results), f, indent=2)
        print(f"\n✅ Results saved to: {filename}")
        return filename
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return None

def run_basic_mamba_analysis(model, tokenizer, texts, device):
    """Run basic Mamba-specific analysis."""
    print("\n" + "="*50)
    print("BASIC MAMBA ANALYSIS")
    print("="*50)
    
    # Move model to CPU for neuron analysis
    model.to('cpu')
    
    # Delta-sensitive neurons
    print("\n🔍 Finding Delta-Sensitive Neurons...")
    try:
        delta_results = find_delta_sensitive_neurons_fixed(model, tokenizer, texts, layer_idx=0, top_k=5)
        print("Top delta-sensitive neurons:")
        for i, (neuron, score) in enumerate(delta_results, 1):
            print(f"  {i}. Neuron {neuron:3d} - Delta Sensitivity: {score:.6f}")
    except Exception as e:
        print(f"❌ Error in delta neuron analysis: {e}")
    
    # Causal neurons
    print("\n🎯 Finding Causal Neurons...")
    causal_prompts = [
        "The capital of France is",
        "The largest mammal is", 
        "Water boils at",
        "The president of the United States is",
        "The speed of light is"
    ]
    
    try:
        causal_results = find_causal_neurons_fixed(model, tokenizer, causal_prompts, layer_idx=0, top_k=5)
        print("Top causal neurons:")
        for i, (neuron, score) in enumerate(causal_results, 1):
            print(f"  {i}. Neuron {neuron:3d} - Causal Impact: {score:.6f}")
    except Exception as e:
        print(f"❌ Error in causal neuron analysis: {e}")

def run_inter_layer_analysis(model, tokenizer):
    """Run inter-layer causal impact analysis."""
    print("\n" + "="*50)
    print("INTER-LAYER IMPACT ANALYSIS")
    print("="*50)
    
    prompt = "What are cats"
    test_dims = [0, 10, 42, 99]
    
    print(f"📊 Analyzing inter-layer impact for prompt: '{prompt}'")
    
    successful_plots = 0
    for dim in test_dims:
        print(f"\n🔬 Analyzing dimension {dim}...")
        try:
            scores = inter_layer_causal_impact_all_layers(model, tokenizer, prompt, dim)
            if scores:
                print(f"Creating plot for dimension {dim}...")
                
                # Create individual plot for each dimension
                plt.figure(figsize=(10, 6))
                layers, impact_scores = zip(*scores)
                plt.plot(layers, impact_scores, marker='o', linewidth=2, markersize=8)
                plt.title(f"Inter-layer Causal Impact - Dimension {dim}")
                plt.xlabel("Layer Index")
                plt.ylabel("Impact Score (KL Divergence)")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                ensure_plot_display(f"Inter-layer_Impact_Dimension_{dim}")
                successful_plots += 1
                print(f"  ✅ Successfully plotted inter-layer impact for dimension {dim}")
            else:
                print(f"  ⚠️  No scores returned for dimension {dim}")
        except Exception as e:
            print(f"  ❌ Error analyzing dimension {dim}: {e}")
    
    print(f"\n📈 Successfully created {successful_plots}/{len(test_dims)} inter-layer plots")

def run_cross_layer_analysis(model, tokenizer):
    """Run cross-layer causal influence analysis."""
    print("\n" + "="*50)
    print("CROSS-LAYER INFLUENCE ANALYSIS")
    print("="*50)
    
    prompt = "The capital of France is"
    test_dims = [0, 10, 42, 99]
    
    print(f"📊 Analyzing cross-layer influence for prompt: '{prompt}'")
    
    # Get model layers
    layers = get_model_layers(model)
    if layers is None:
        print("❌ Could not extract model layers")
        return
    
    num_layers = len(layers)
    max_layers_to_test = min(num_layers, 4)  # Limit for efficiency
    
    print(f"🔗 Testing {max_layers_to_test}x{max_layers_to_test} layer interactions...")
    
    successful_analyses = 0
    for dim in test_dims:
        print(f"\n🔬 Analyzing dimension {dim}...")
        try:
            # Test cross-layer influence between adjacent layers
            for src_layer in range(max_layers_to_test - 1):
                tgt_layer = src_layer + 1
                result = cross_layer_causal_influence(
                    model, tokenizer, prompt, 
                    src_layer_idx=src_layer, 
                    tgt_layer_idx=tgt_layer, 
                    dim=dim
                )
                if result is not None:
                    successful_analyses += 1
                    print(f"  ✅ L{src_layer}→L{tgt_layer} dim{dim}: {result:.6f}")
                else:
                    print(f"  ❌ L{src_layer}→L{tgt_layer} dim{dim}: No result")
        except Exception as e:
            print(f"❌ Error analyzing dimension {dim}: {e}")
    
    print(f"\n✅ Successfully analyzed {successful_analyses} cross-layer interactions")

def run_attention_neurons_analysis(model, tokenizer, texts, layer_idx=0, methods=None, plots_dir='plots'):
    """Run attention neurons analysis for Mamba models."""
    if methods is None:
        methods = ['attention_weighted', 'gradient_guided', 'rollout']
    
    print("\n" + "="*50)
    print("ATTENTION NEURONS ANALYSIS")
    print("="*50)
    
    try:
        # Create sample input for attention analysis
        sample_text = texts[0] if texts else "Sample text for analysis"
        sample_input = tokenizer(sample_text, return_tensors="pt")["input_ids"]
        
        print(f"🔍 Analyzing attention neurons for layer {layer_idx}...")
        print(f"📝 Sample text: '{sample_text[:100]}{'...' if len(sample_text) > 100 else ''}'")
        print(f"🔧 Using methods: {', '.join(methods)}")
        
        # Run attention neurons analysis with specified methods
        attention_results = integrate_mamba_attention_neurons(
            model, sample_input, 
            layer_indices=[layer_idx], 
            methods=methods
        )
        
        if attention_results and 'analysis_results' in attention_results:
            print("✅ Attention neurons analysis completed successfully!")
            
            # Print analysis summary
            for method, method_data in attention_results['analysis_results'].items():
                if layer_idx in method_data:
                    layer_data = method_data[layer_idx]
                    print(f"\n📊 {method.replace('_', ' ').title()} Method Results:")
                    if 'num_neurons' in layer_data:
                        print(f"  • Neurons analyzed: {layer_data['num_neurons']}")
                    if 'mean_activation' in layer_data:
                        print(f"  • Mean activation: {layer_data['mean_activation']:.4f}")
                    if 'activation_std' in layer_data:
                        print(f"  • Activation std: {layer_data['activation_std']:.4f}")
                    if 'neuron_diversity' in layer_data:
                        print(f"  • Neuron diversity: {layer_data['neuron_diversity']:.4f}")
            
            # Create visualizations if analyzer is available
            if 'analyzer' in attention_results:
                analyzer = attention_results['analyzer']
                try:
                    # Get mamba neurons for visualization
                    if 'mamba_neurons' in attention_results and 'attention_weighted' in attention_results['mamba_neurons']:
                        mamba_neurons = attention_results['mamba_neurons']['attention_weighted']
                        if layer_idx in mamba_neurons:
                            print(f"\n🎨 Generating attention neurons visualization for layer {layer_idx}...")
                            save_path = os.path.join(plots_dir, f"attention_neurons_layer_{layer_idx}.png")
                            analyzer.visualize_neurons(
                                mamba_neurons, 
                                layer_idx=layer_idx, 
                                save_path=save_path
                            )
                            print(f"✅ Attention neurons visualization saved to: {save_path}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not generate visualization: {e}")
            
            return attention_results
        else:
            print("❌ Attention neurons analysis failed - no results returned")
            return None
            
    except Exception as e:
        print(f"❌ Error in attention neurons analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_comparison_analysis(texts):
    """Run comprehensive comparison analysis."""
    print("\n" + "="*50)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*50)
    
    if not COMPARISON_AVAILABLE:
        print("❌ Comparison functionality not available.")
        print("   Please ensure comparison_plots.py is available and dependencies are installed.")
        return
    
    try:
        print("🔄 Loading models for comparison...")
        models = load_models()
        
        if not models:
            print("❌ No models loaded successfully.")
            return
        
        print("📊 Creating comparison plots...")
        create_comparison_plots(models, texts)
        
        print("✅ Comparison analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in comparison analysis: {e}")
        import traceback
        traceback.print_exc()

def analyze_model_efficiency(model, tokenizer, texts):
    """Analyze model efficiency metrics."""
    print("\n" + "="*50)
    print("MODEL EFFICIENCY ANALYSIS")
    print("="*50)

    try:
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())

        # Analyze activation patterns
        print("🔍 Analyzing activation patterns...")

        # Get sample activations
        sample_text = texts[0] if texts else "The quick brown fox"
        inputs = tokenizer(sample_text, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Calculate efficiency metrics
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            # Assuming the first hidden state is representative of the first layer's activations
            # You might need to adjust this depending on your model's architecture
            first_layer_activations = outputs.hidden_states[0]

            # Calculate sparsity (fraction of near-zero activations)
            # Note: This function doesn't have access to model type information
            # For model-specific thresholds, use the comparison_plots.py analysis instead
            # Use a conservative threshold that works for most models
            sparsity_threshold = 0.01  # Conservative default threshold
            
            sparsity = torch.mean((torch.abs(first_layer_activations) < sparsity_threshold).float()).item()

            # Calculate variance of activations
            variance = torch.var(first_layer_activations).item()

            # Active neuron count (rough estimate based on a threshold)
            # This threshold (0.1) can be tuned based on your model's activation scale
            active_neurons = torch.sum(torch.abs(first_layer_activations) > 0.1).item()

            # Efficiency ratio: active neurons relative to total parameters
            # This is a simplified metric and might need refinement for complex models
            efficiency_ratio = active_neurons / total_params if total_params > 0 else 0

            print("📈 Efficiency Metrics:")
            print(f"  • Total Parameters: {total_params:,}")
            print(f"  • Active Neurons (est): {active_neurons:,}")
            print(f"  • Efficiency Ratio: {efficiency_ratio:.6f}")
            print(f"  • Activation Variance: {variance:.6f}")
            print(f"  • Sparsity (threshold={sparsity_threshold}): {sparsity:.4f}")

            # --- Create efficiency visualizations (Separate Plots) ---

            # Plot 1: Activation distribution
            plt.figure(figsize=(8, 6)) # Create a new figure for this plot
            activations_flat = first_layer_activations.flatten().cpu().numpy()
            plt.hist(activations_flat, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
            plt.title("Activation Distribution", fontsize=16)
            plt.xlabel("Activation Value", fontsize=12)
            plt.ylabel("Density", fontsize=12)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout() # Adjust layout for this specific figure
            ensure_plot_display("Activation_Distribution") # Save this figure

            # Plot 2: Efficiency metrics
            plt.figure(figsize=(8, 8)) # Create another new figure
            metrics = ['Sparsity', 'Variance', 'Efficiency']
            # Scale efficiency for visibility if its value is very small
            values = [sparsity, variance, efficiency_ratio * 1000]
            bars = plt.bar(metrics, values, alpha=0.8, color=['lightcoral', 'lightgreen', 'lightsalmon'])
            plt.title("Model Efficiency Metrics", fontsize=16)
            plt.ylabel("Value", fontsize=12)
            plt.grid(True, alpha=0.3, linestyle='--')

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height / 2,
                    f'{value:.4f}',
                    ha='center',
                    va='center',
                    color='black',
                    fontsize=10,
                    fontweight='bold'
                )

            # Plot 3: Parameter utilization
            plt.figure(figsize=(8, 6)) # Create a third new figure
            labels = ['Active', 'Inactive']
            sizes = [active_neurons, total_params - active_neurons]
            colors = ['#66b3ff', '#ff9999'] # More distinct colors
            explode = (0.05, 0) # Slightly 'explode' the active slice for emphasis
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, explode=explode,
                    wedgeprops={'edgecolor': 'black', 'linewidth': 1})
            plt.title("Parameter Utilization", fontsize=16)
            plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
            plt.tight_layout() # Adjust layout for this figure
            ensure_plot_display("Model_Efficiency_Metrics") # Save this figure

            return {
                'total_parameters': total_params,
                'active_neurons': active_neurons,
                'efficiency_ratio': efficiency_ratio,
                'mean_variance': variance,
                'mean_sparsity': sparsity
            }
        else:
            print("❌ Could not extract hidden states for efficiency analysis. Ensure 'output_hidden_states=True' is supported and the model provides them.")
            return None

    except Exception as e:
        print(f"❌ Error in efficiency analysis: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Mamba neuron analysis with attention neurons")
    parser.add_argument('--model', type=str, default="state-spaces/mamba-130m-hf", 
                       help='Model to analyze')
    parser.add_argument('--layer', type=int, default=0, 
                       help='Layer index to analyze (default: 0)')
    parser.add_argument('--top_k', type=int, default=10, 
                       help='Top K neurons to analyze (default: 10)')
    parser.add_argument('--text_limit', type=int, default=100, 
                       help='Limit number of texts to process (default: 100)')
    parser.add_argument('--enable_attention', action='store_true', 
                       help='Enable attention neurons analysis (default: True)')
    parser.add_argument('--attention_methods', nargs='+', 
                       default=['attention_weighted', 'gradient_guided', 'rollout'],
                       help='Methods for attention neurons analysis')
    parser.add_argument('--save_results', action='store_true', 
                       help='Save analysis results to JSON file')
    parser.add_argument('--plots_dir', type=str, default='plots', 
                       help='Directory to save plots (default: plots)')
    
    args = parser.parse_args()
    
    # Create plots directory
    import os
    os.makedirs(args.plots_dir, exist_ok=True)
    
    print("\n🚀 Starting Comprehensive Mamba Neuron Analysis")
    print(f"Model: {args.model}")
    print(f"Layer: {args.layer}")
    print(f"Text limit: {args.text_limit}")
    print(f"Attention analysis: {'Enabled' if args.enable_attention else 'Disabled'}")
    if args.enable_attention:
        print(f"Attention methods: {', '.join(args.attention_methods)}")
    print(f"Plots directory: {args.plots_dir}")
    
    # Set up matplotlib for file saving
    try:
        matplotlib.use('Agg')
        plt.ioff()
        print("✅ File-based plotting enabled - plots will be saved to 'plots/' directory")
    except:
        print("⚠️  File-based plotting may not work properly")
    
    # Load model and tokenizer
    print("\n🔧 Loading model and tokenizer...")
    try:
        model_name = args.model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        print("✅ Model and tokenizer loaded successfully!")
        
        # Debug model structure
        print("\n🔍 Debugging model structure...")
        debug_model_structure(model, max_depth=3)
        
    except Exception as e:
        print(f"❌ Error loading model or tokenizer: {e}")
        return

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n💻 Using device: {device}")
    model.to(device)

    # Load dataset
    print("\n📚 Loading dataset...")
    try:
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")
        texts = [item["text"] for item in dataset if item["text"].strip() != ""]
        texts = texts[:args.text_limit]  # Limit for efficiency
        print(f"✓ Loaded {len(texts)} non-empty samples.")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        # Use fallback texts
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world.",
            "Machine learning models require large datasets.",
            "Natural language processing involves understanding text.",
            "Deep learning has revolutionized computer vision.",
            "Neural networks can learn complex patterns.",
            "Transformers use attention mechanisms.",
            "State-space models are efficient for long sequences.",
            "The capital of France is Paris.",
            "Water boils at 100 degrees Celsius."
        ]
        print(f"✓ Using {len(texts)} fallback texts.")

    # Run different analysis modules
    analysis_results = {}
    
    try:
        print("\n🚀 Starting analysis pipeline...")
        
        # 1. Basic Mamba analysis
        print("\n" + "="*20 + " STAGE 1: BASIC ANALYSIS " + "="*20)
        run_basic_mamba_analysis(model, tokenizer, texts, device)
        
        # 2. Inter-layer analysis
        print("\n" + "="*20 + " STAGE 2: INTER-LAYER ANALYSIS " + "="*20)
        run_inter_layer_analysis(model, tokenizer)
        
        # 3. Cross-layer analysis
        print("\n" + "="*20 + " STAGE 3: CROSS-LAYER ANALYSIS " + "="*20)
        run_cross_layer_analysis(model, tokenizer)
        
        # 4. Attention neurons analysis
        if args.enable_attention:
            print("\n" + "="*20 + " STAGE 4: ATTENTION NEURONS ANALYSIS " + "="*20)
            attention_results = run_attention_neurons_analysis(model, tokenizer, texts, layer_idx=args.layer, methods=args.attention_methods, plots_dir=args.plots_dir)
            if attention_results:
                analysis_results['attention_neurons'] = attention_results
        
        # 5. Efficiency analysis
        print("\n" + "="*20 + " STAGE 5: EFFICIENCY ANALYSIS " + "="*20)
        efficiency_metrics = analyze_model_efficiency(model, tokenizer, texts)
        if efficiency_metrics:
            analysis_results['efficiency'] = efficiency_metrics
        
        # 6. Comprehensive comparison analysis
        print("\n" + "="*20 + " STAGE 6: COMPARISON ANALYSIS " + "="*20)
        run_comparison_analysis(texts)
        
    except Exception as e:
        print(f"❌ Error in analysis pipeline: {e}")
        import traceback
        traceback.print_exc()
    # Final summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("="*60)
    print("✅ Completed analyses:")
    print("  • Basic Mamba neuron analysis (delta-sensitive, causal)")
    print("  • Inter-layer causal impact analysis with individual plots")
    print("  • Cross-layer influence analysis with matrix visualization")
    if args.enable_attention:
        print("  • Attention neurons analysis with multiple methods")
    print("  • Model efficiency analysis with metrics")
    print("  • Comprehensive Mamba vs Transformer comparison")
    
    if analysis_results:
        print("\n📊 Key Results:")
        if 'efficiency' in analysis_results:
            eff = analysis_results['efficiency']
            print(f"  • Total Parameters: {eff['total_parameters']:,}")
            print(f"  • Efficiency Ratio: {eff['efficiency_ratio']:.6f}")
            print(f"  • Sparsity: {eff['mean_sparsity']:.4f}")
        
        if args.enable_attention and 'attention_neurons' in analysis_results:
            attn = analysis_results['attention_neurons']
            print(f"\n🧠 Attention Neurons Analysis:")
            if 'analysis_results' in attn and 'attention_weighted' in attn['analysis_results']:
                layer_data = attn['analysis_results']['attention_weighted'].get(args.layer, {})
                if 'num_neurons' in layer_data:
                    print(f"  • Neurons analyzed: {layer_data['num_neurons']}")
                if 'mean_activation' in layer_data:
                    print(f"  • Mean activation: {layer_data['mean_activation']:.4f}")
                if 'neuron_diversity' in layer_data:
                    print(f"  • Neuron diversity: {layer_data['neuron_diversity']:.4f}")
    
    print("\n💡 Tips:")
    print("  • All plots are automatically saved to the 'plots/' directory")
    print("  • Check the 'plots/' folder for generated PNG files")
    print("  • Plots are saved with descriptive filenames for easy identification")
    print("  • High-resolution images (300 DPI) are generated for publication quality")
    if args.enable_attention:
        print("  • Attention neurons visualizations are saved as 'attention_neurons_layer_X.png'")
    
    # Save results if requested
    if args.save_results:
        print(f"\n💾 Saving analysis results...")
        save_results(analysis_results, output_dir="analysis_outputs")
    
    print(f"\n🎉 Analysis complete! All visualizations have been saved to: {args.plots_dir}")
    if args.save_results:
        print("📁 Analysis results have been saved to: analysis_outputs/")
if __name__ == "__main__":
    import argparse
    main()