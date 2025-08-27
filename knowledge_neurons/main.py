#!/usr/bin/env python3
"""
Main script for knowledge neuron analysis using integrated gradients
on GPT-2 and Mamba. Updated to avoid top-k thresholding and handle empty tokenization.
Now includes attention weight mechanism integration for enhanced analysis.
"""

import os
import json
import torch
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from knowledge_neurons import KnowledgeNeuronsFinder
from attention_neurons import MambaAttentionNeurons, integrate_mamba_attention_neurons

def get_improved_facts():
    return {
        'capital_france': {
            'texts': [
                "What is the capital city of France?",
                "France has its capital in which city?",
                "Paris is the city that serves as the capital of France.",
                "The city serving as the political center of France is called",
                "In Europe, the country France has its capital city in"
            ],
            'ground_truth': 'Paris'
        },
        'capital_germany': {
            'texts': [
                "What is the capital city of Germany?",
                "Germany has its capital in which city?",
                "Berlin is the city that serves as the capital of Germany.",
                "The city serving as the political center of Germany is called",
                "In Europe, the country Germany has its capital city in"
            ],
            'ground_truth': 'Berlin'
        },
        'capital_italy': {
            'texts': [
                "What is the capital city of Italy?",
                "Italy has its capital in which city?",
                "Rome is the city that serves as the capital of Italy.",
                "The city serving as the political center of Italy is called",
                "In Europe, the country Italy has its capital city in"
            ],
            'ground_truth': 'Rome'
        },
        # Original facts for inter-relation analysis
        'wwii_end_year': {
            'texts': [
                "In what year did World War II conclude?",
                "World War II ended in",
                "The year of the Axis surrender in WWII was"
            ],
            'ground_truth': '1945'
        },
        'water_chemical_symbol': {
            'texts': [
                "What is the chemical symbol for water?",
                "The chemical formula for water is",
                "Water's scientific notation is"
            ],
            'ground_truth': 'H2O'
        },
        'eiffel_tower_city': {
            'texts': [
                "In which city is the Eiffel Tower located?",
                "The Eiffel Tower stands in",
                "The famous tower in Paris is the"
            ],
            'ground_truth': 'Paris'
        },
        'romeo_juliet_author': {
            'texts': [
                "Who wrote the play 'Romeo and Juliet'?",
                "'Romeo and Juliet' was penned by",
                "The author of 'Romeo and Juliet' is"
            ],
            'ground_truth': 'Shakespeare'
        },
        'longest_river_africa': {
            'texts': [
                "What is the longest river in Africa?",
                "The Nile is the longest river in",
                "The longest African river is the"
            ],
            'ground_truth': 'Nile'
        },
        # --- New facts for Intra-Relation Analysis (from previous turn) ---

        # More authors
        'hamlet_author': {
            'texts': [
                "Who wrote the play 'Hamlet'?",
                "'Hamlet' was penned by",
                "The author of 'Hamlet' is"
            ],
            'ground_truth': 'Shakespeare'
        },
        'macbeth_author': {
            'texts': [
                "Who wrote the play 'Macbeth'?",
                "'Macbeth' was penned by",
                "The author of 'Macbeth' is"
            ],
            'ground_truth': 'Shakespeare'
        },

        # More rivers
        'longest_river_south_america': {
            'texts': [
                "What is the longest river in South America?",
                "The Amazon is the longest river in",
                "The longest South American river is the"
            ],
            'ground_truth': 'Amazon'
        },
        'longest_river_asia': {
            'texts': [
                "What is the longest river in Asia?",
                "The Yangtze is the longest river in",
                "The longest Asian river is the"
            ],
            'ground_truth': 'Yangtze'
        },
    }

def get_fact_relations():
    return {
        'capital_france': 'Geography_Capitals',
        'capital_germany': 'Geography_Capitals',
        'capital_italy': 'Geography_Capitals',
        'wwii_end_year': 'History_Events',
        'water_chemical_symbol': 'Chemistry_Formulas',
        'eiffel_tower_city': 'Geography_Landmarks',
        'romeo_juliet_author': 'Literature_Authors',
        'longest_river_africa': 'Geography_Rivers',
        'hamlet_author': 'Literature_Authors',
        'macbeth_author': 'Literature_Authors',
        'longest_river_south_america': 'Geography_Rivers',
        'longest_river_asia': 'Geography_Rivers',
    }

def compute_overlap_thresholded(neuron_results, relations, threshold=0.001):
    """
    Computes overlap statistics (Jaccard similarity and intersection size)
    for knowledge neurons based on a threshold.
    """
    all_facts = list(neuron_results.keys())
    overlaps = []

    for i in range(len(all_facts)):
        for j in range(i + 1, len(all_facts)):
            fact1_name = all_facts[i]
            fact2_name = all_facts[j]

            # Get significant neurons for each fact
            neurons1 = neuron_results.get(fact1_name)
            neurons2 = neuron_results.get(fact2_name)

            if not neurons1 or not neurons2:
                print(f"Skipping overlap for {fact1_name} and {fact2_name}: missing neuron results.")
                continue

            # Filter neurons based on the absolute threshold
            # Ensure neurons are in a list/array format for easy filtering
            if isinstance(neurons1, list):
                significant_neurons1 = {idx for idx, val in enumerate(neurons1) if abs(val) > threshold}
            else: # Assume numpy array or similar
                significant_neurons1 = {idx for idx, val in enumerate(neurons1.flatten()) if abs(val) > threshold}

            if isinstance(neurons2, list):
                significant_neurons2 = {idx for idx, val in enumerate(neurons2) if abs(val) > threshold}
            else: # Assume numpy array or similar
                significant_neurons2 = {idx for idx, val in enumerate(neurons2.flatten()) if abs(val) > threshold}


            intersection = len(significant_neurons1.intersection(significant_neurons2))
            union = len(significant_neurons1.union(significant_neurons2))

            jaccard_similarity = intersection / union if union > 0 else 0

            same_relation = (relations.get(fact1_name) == relations.get(fact2_name))

            overlaps.append({
                'fact1': fact1_name,
                'fact2': fact2_name,
                'jaccard_similarity': jaccard_similarity,
                'intersection_size': intersection,
                'same_relation': same_relation,
                'relation_type': relations.get(fact1_name, 'unknown') # Add relation type for debugging/analysis
            })

    # Separate overlaps by relation type for summary statistics
    intra_relation_overlaps = [o for o in overlaps if o['same_relation']]
    inter_relation_overlaps = [o for o in overlaps if not o['same_relation']]

    stats = {
        'intra_relation': {
            'num_pairs': len(intra_relation_overlaps),
            'avg_jaccard': np.mean([o['jaccard_similarity'] for o in intra_relation_overlaps]) if intra_relation_overlaps else 0,
            'avg_intersection_size': np.mean([o['intersection_size'] for o in intra_relation_overlaps]) if intra_relation_overlaps else 0
        },
        'inter_relation': {
            'num_pairs': len(inter_relation_overlaps),
            'avg_jaccard': np.mean([o['jaccard_similarity'] for o in inter_relation_overlaps]) if inter_relation_overlaps else 0,
            'avg_intersection_size': np.mean([o['intersection_size'] for o in inter_relation_overlaps]) if inter_relation_overlaps else 0
        }
    }

    return {
        'summary_stats': stats,
        'detailed_overlaps': overlaps
    }


def run_analysis_for_model(model_name, output_filename):
    print("="*70)
    print(f"Running Knowledge Neurons Analysis for: {model_name}")
    print("="*70)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        model.eval()
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    except Exception as e:
        print(f"❌ Error loading model {model_name}: {e}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Check if this is a Mamba model for attention analysis
    is_mamba = 'mamba' in model_name.lower()
    
    if is_mamba:
        print("🔍 Mamba model detected. Will run attention analysis alongside knowledge neurons.")
    else:
        print("ℹ️  Non-Mamba model detected. Running standard knowledge neuron analysis.")

    kn_finder = KnowledgeNeuronsFinder(model, tokenizer, model_type='gpt2' if 'gpt2' in model_name else 'mamba')
    facts_dict = get_improved_facts()
    relations = get_fact_relations()

    # --- DEBUGGING BLOCK START ---
    for fact_name, fact_data in facts_dict.items():
        texts = fact_data['texts']
        ground_truth = fact_data['ground_truth']
        try:
            ground_truth_tokens = tokenizer.encode(ground_truth, add_special_tokens=False)
            if not ground_truth_tokens:
                print(f"DEBUG: Could not tokenize ground truth '{ground_truth}' for fact '{fact_name}'")
                continue
            target_token_id = ground_truth_tokens[0]
            print(f"\nDEBUG: Fact: {fact_name}, Ground Truth: '{ground_truth}', Tokenized GT ID: {target_token_id} ('{tokenizer.decode(target_token_id)}')")

            for i, text in enumerate(texts):
                if i >= 1: break # Just check the first prompt to keep output manageable
                input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(device)
                with torch.no_grad():
                    outputs = model(input_ids)
                    logits = outputs.logits
                    next_token_logits = logits[0, -1, :] # Logits for the last token position

                    predicted_token_id = torch.argmax(next_token_logits).item()
                    predicted_token = tokenizer.decode(predicted_token_id)

                    # Check likelihood of the actual target token ID
                    if target_token_id < len(next_token_logits):
                        target_token_prob = torch.softmax(next_token_logits, dim=-1)[target_token_id].item()
                        print(f"  Prompt: '{text[:80]}...' -> Predicted: '{predicted_token}' (ID: {predicted_token_id}), Target '{ground_truth}' Prob: {target_token_prob:.4f}")
                    else:
                        print(f"  Prompt: '{text[:80]}...' -> Predicted: '{predicted_token}' (ID: {predicted_token_id}), Target token ID {target_token_id} out of vocabulary.")
        except Exception as e:
            print(f"DEBUG: Error during pre-check for fact {fact_name}: {e}")
    # --- DEBUGGING BLOCK END ---

    # Run knowledge neuron analysis for multiple layers
    layer_indices = [0, 6, 12, 18] if is_mamba else [0, 4, 8, 11]  # Different layers for different models
    neuron_results_by_layer = {}
    
    for layer_idx in layer_indices:
        print(f"\n🔍 Analyzing Layer {layer_idx}...")
        layer_results = {}
        
        for fact_name, fact_data in facts_dict.items():
            print(f"  Analyzing knowledge neurons for fact: {fact_name} at layer {layer_idx}")
            
            # Get knowledge neurons for specific layer
            attributions = kn_finder.find_knowledge_neurons_for_fact_fixed(
                fact_data, layer_idx=layer_idx, steps=15, top_k=None
            )
            layer_results[fact_name] = attributions

            if attributions:
                print(f"    Raw attributions (first 5 and max): {attributions[:5]}... Max: {max(attributions):.4f}")
            else:
                print(f"    No raw attributions returned for {fact_name}.")
        
        neuron_results_by_layer[layer_idx] = layer_results

    # Get the final layer results for the main analysis
    final_layer_idx = layer_indices[-1]
    neuron_results = neuron_results_by_layer[final_layer_idx]

    # Compute overlaps using thresholding
    overlap_stats = compute_overlap_thresholded(neuron_results, relations, threshold=0.001)

    print("\n==== KNOWLEDGE NEURON RESULTS ====")
    for fact, neurons in neuron_results.items():
        print(f"\n{fact}:")
        if isinstance(neurons, list) and neurons:
            significant = [i for i, v in enumerate(neurons) if abs(v) > 0.001]
            print(f"  Significant neurons above threshold: {len(significant)}")
        else:
            print("  No neurons found or attributions not in expected format.")

    print("\n==== KNOWLEDGE NEURON OVERLAP ANALYSIS ====")
    stats = overlap_stats['summary_stats']
    for relation_type, s in stats.items():
        print(f"\n{relation_type.replace('_', ' ').title()}:")
        print(f"  Pairs: {s['num_pairs']}")
        print(f"  Avg Jaccard Similarity: {s['avg_jaccard']:.4f}")
        print(f"  Avg Intersection Size: {s['avg_intersection_size']:.2f}")

    # NEW: Inter-layer and Intra-layer Analysis
    print("\n==== INTER-LAYER AND INTRA-LAYER ANALYSIS ====")
    inter_intra_stats = analyze_inter_intra_layer_behavior(neuron_results_by_layer, relations)
    
    print("\n" + "="*70)
    print("INTER-LAYER AND INTRA-LAYER KNOWLEDGE NEURON STATISTICS")
    print("="*70)
    print(f"{'Layer':<8} | {'Relation Type':<15} | {'# Pairs':>8} | {'Avg Jaccard':>12} | {'Avg ∩ Size':>12}")
    print("-"*70)
    
    for layer_idx in layer_indices:
        if layer_idx in inter_intra_stats:
            layer_stats = inter_intra_stats[layer_idx]
            print(f"{layer_idx:<8} | {'Intra Relation':<15} | {layer_stats['intra_relation']['num_pairs']:>8} | {layer_stats['intra_relation']['avg_jaccard']:>12.4f} | {layer_stats['intra_relation']['avg_intersection_size']:>12.2f}")
            print(f"{layer_idx:<8} | {'Inter Relation':<15} | {layer_stats['inter_relation']['num_pairs']:>8} | {layer_stats['inter_relation']['avg_jaccard']:>12.4f} | {layer_stats['inter_relation']['avg_intersection_size']:>12.2f}")
    
    print("-"*70)

    print("\n==== DETAILED OVERLAP ANALYSIS ====")
    for o in overlap_stats['detailed_overlaps']:
        tag = "Intra" if o.get("same_relation") else "Inter"
        print(f"  [{tag}] {o['fact1']} ↔ {o['fact2']}: "
              f"Jaccard={o['jaccard_similarity']:.3f}, "
              f"Intersection={o['intersection_size']}")

    # Run attention analysis for Mamba models (as separate analysis)
    attention_results = None
    if is_mamba:
        print("\n🔍 Running Attention Analysis for Mamba Model...")
        try:
            # Get sample input for attention analysis
            sample_fact = list(facts_dict.keys())[0]
            sample_text = facts_dict[sample_fact]['texts'][0]
            sample_input = tokenizer(sample_text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(device)
            
            # Run attention analysis using the reference project approach
            attention_results = integrate_mamba_attention_neurons(
                model, sample_input, layer_indices=[0, 6, 12, 18], methods=['attention_weighted', 'gradient_guided']
            )
            
            print("✅ Attention analysis completed successfully")
            
            # Print attention analysis summary
            if attention_results and 'analysis_results' in attention_results:
                print("\n==== ATTENTION ANALYSIS SUMMARY ====")
                for method, method_results in attention_results['analysis_results'].items():
                    print(f"\n{method.replace('_', ' ').title()} Method:")
                    for layer_idx, layer_data in method_results.items():
                        if layer_data:
                            print(f"  Layer {layer_idx}: {layer_data.get('num_neurons', 0)} neurons, "
                                  f"Mean activation: {layer_data.get('mean_activation', 0):.4f}, "
                                  f"Diversity: {layer_data.get('neuron_diversity', 0):.4f}")
            
        except Exception as e:
            print(f"❌ Attention analysis failed: {e}")
            attention_results = None

    # Save results
    results_to_save = {
        "model_name": model_name,
        "knowledge_neurons": neuron_results,
        "knowledge_neurons_by_layer": neuron_results_by_layer,
        "inter_intra_layer_stats": inter_intra_stats,
        "overlap_statistics": overlap_stats,
        "fact_relations": relations,
        "analysis_type": "knowledge_neurons_with_attention",
        "has_attention_analysis": attention_results is not None,
        "timestamp": datetime.now().isoformat()
    }
    
    if attention_results:
        results_to_save["attention_analysis"] = {
            "attention_data_keys": list(attention_results.get('attention_data', {}).keys()),
            "mamba_neurons_methods": list(attention_results.get('mamba_neurons', {}).keys()),
            "analysis_results_summary": attention_results.get('analysis_results', {})
        }

    with open(output_filename, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\n✅ Results saved to {output_filename}")

    return {
        "model_name": model_name,
        "knowledge_neuron_overlap": stats,
        "inter_intra_layer_stats": inter_intra_stats,
        "attention_analysis": attention_results is not None
    }

def analyze_inter_intra_layer_behavior(neuron_results_by_layer, relations):
    """
    Analyze knowledge neuron behavior across different layers.
    
    Args:
        neuron_results_by_layer: Dictionary of neuron results for each layer
        relations: Fact relations dictionary
    
    Returns:
        Dictionary containing inter-layer and intra-layer statistics
    """
    inter_intra_stats = {}
    
    for layer_idx, layer_results in neuron_results_by_layer.items():
        # Compute overlaps for this layer
        layer_overlaps = compute_overlap_thresholded(layer_results, relations, threshold=0.001)
        inter_intra_stats[layer_idx] = layer_overlaps['summary_stats']
    
    return inter_intra_stats

def run_comprehensive_analysis(model_name, output_filename):
    """
    Run comprehensive analysis combining knowledge neurons and attention mechanisms.
    This provides a complete picture of both approaches for model understanding.
    """
    print("="*70)
    print(f"Running Comprehensive Analysis for: {model_name}")
    print("="*70)
    
    # Run the integrated analysis (knowledge neurons + attention)
    result = run_analysis_for_model(model_name, output_filename)
    
    print(f"\n✅ Comprehensive analysis completed for {model_name}")
    return result

def compare_models():
    print("\n" + "="*70)
    print("COMPARATIVE KNOWLEDGE NEURONS ANALYSIS")
    print("="*70)

    results = []
    
    for label, model_name, output_file in [
        ("Mamba", "state-spaces/mamba-130m-hf", "mamba_knowledge_neurons.json"),
        ("GPT-2", "gpt2", "gpt2_knowledge_neurons.json")
    ]:
        print(f"\n🔍 Analyzing {label}...")
        
        # Run knowledge neuron analysis (with attention for Mamba)
        result = run_analysis_for_model(model_name, output_file)
        results.append((label, result))

    print("\n" + "="*70)
    print("COMPARISON TABLE: Knowledge Neuron Overlap Statistics")
    print("="*70)
    print(f"{'Model':<15} | {'Relation Type':<15} | {'# Pairs':>8} | {'Avg Jaccard':>12} | {'Avg ∩ Size':>12}")
    print("-"*70)

    for model_label, result in results:
        if result is None:
            print(f"{model_label:<15} | {'ERROR':<15} | {'-':>8} | {'-':>12} | {'-':>12}")
            continue
        stats = result['knowledge_neuron_overlap']
        print(f"{model_label:<15} | {'Intra Relation':<15} | {stats['intra_relation']['num_pairs']:>8} | {stats['intra_relation']['avg_jaccard']:>12.4f} | {stats['intra_relation']['avg_intersection_size']:>12.2f}")
        print(f"{model_label:<15} | {'Inter Relation':<15} | {stats['inter_relation']['num_pairs']:>8} | {stats['inter_relation']['avg_jaccard']:>12.4f} | {stats['inter_relation']['avg_intersection_size']:>12.2f}")
        
        # Show attention analysis status for Mamba models
        if 'mamba' in model_label.lower() and result.get('attention_analysis'):
            print(f"{model_label:<15} | {'Attention Analysis':<15} | {'✓':>8} | {'Completed':>12} | {'-':>12}")

    print("-"*70)
    
    # Print detailed comparison summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    if len(results) >= 2:
        mamba_result = results[0][1] if 'mamba' in results[0][0].lower() else results[1][1]
        gpt2_result = results[1][1] if 'gpt2' in results[1][0].lower() else results[0][1]
        
        if mamba_result and gpt2_result:
            mamba_stats = mamba_result['knowledge_neuron_overlap']
            gpt2_stats = gpt2_result['knowledge_neuron_overlap']
            
            print("Mamba vs GPT-2 Knowledge Neuron Overlap Comparison:")
            print(f"  Intra Relation - Mamba: {mamba_stats['intra_relation']['avg_jaccard']:.4f}, GPT-2: {gpt2_stats['intra_relation']['avg_jaccard']:.4f}")
            print(f"  Inter Relation - Mamba: {mamba_stats['inter_relation']['avg_jaccard']:.4f}, GPT-2: {gpt2_stats['inter_relation']['avg_jaccard']:.4f}")
            
            # Calculate improvement ratios
            intra_improvement = mamba_stats['intra_relation']['avg_jaccard'] / (gpt2_stats['intra_relation']['avg_jaccard'] + 1e-8)
            inter_improvement = mamba_stats['inter_relation']['avg_jaccard'] / (gpt2_stats['inter_relation']['avg_jaccard'] + 1e-8)
            
            print(f"  Mamba shows {intra_improvement:.1f}x better intra-relation overlap")
            print(f"  Mamba shows {inter_improvement:.1f}x better inter-relation overlap")
            
            # Attention analysis summary for Mamba
            if mamba_result.get('attention_analysis'):
                print(f"  Mamba attention analysis: Completed successfully")
            else:
                print(f"  Mamba attention analysis: Not available")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        analysis_type = sys.argv[1].lower()
        
        if analysis_type == "comprehensive":
            print("🚀 Running Comprehensive Analysis (Knowledge Neurons + Attention)")
            # Run comprehensive analysis for Mamba model
            run_comprehensive_analysis("state-spaces/mamba-130m-hf", "mamba_comprehensive_analysis.json")
        elif analysis_type == "attention":
            print("🔍 Running Knowledge Neurons + Attention Analysis")
            # Run integrated analysis for Mamba model
            run_analysis_for_model("state-spaces/mamba-130m-hf", "mamba_analysis.json")
        elif analysis_type == "knowledge":
            print("📊 Running Knowledge Neuron Analysis Only")
            # Run knowledge neuron analysis for both models
            compare_models()
        else:
            print(f"❌ Unknown analysis type: {analysis_type}")
            print("Available options: comprehensive, attention, knowledge")
            print("Defaulting to comprehensive analysis...")
            run_comprehensive_analysis("state-spaces/mamba-130m-hf", "mamba_comprehensive_analysis.json")
    else:
        print("🚀 Running Default Comprehensive Analysis")
        print("Use command line arguments to specify analysis type:")
        print("  python main.py comprehensive  # Full analysis (default)")
        print("  python main.py attention      # Knowledge neurons + attention")
        print("  python main.py knowledge      # Knowledge neurons only")
        print()
        # Run comprehensive analysis by default
        run_comprehensive_analysis("state-spaces/mamba-130m-hf", "mamba_comprehensive_analysis.json")