import torch
import numpy as np
import random

# Relational templates for different Wikidata properties
RELATION_TEMPLATES = {
    "P176": {  # manufacturer
        "relation_name": "manufacturer",
        "templates": [
            "[X] is produced by [Y]",
            "[X] is a product of [Y]", 
            "[Y] and its product [X]"
        ]
    },
    "P463": {  # member_of
        "relation_name": "member_of",
        "templates": [
            "[X] is a member of [Y]",
            "[X] belongs to the organization of [Y]",
            "[X] is affiliated with [Y]"
        ]
    },
    "P407": {  # language_of_work
        "relation_name": "language_of_work", 
        "templates": [
            "[X] was written in [Y]",
            "The language of [X] is [Y]",
            "[X] was a [Y]-language work"
        ]
    }
}

# Example facts for each relation type
EXAMPLE_FACTS = {
    "P176": [  # manufacturer
        ("iPhone", "Apple"),
        ("Galaxy S24", "Samsung"),
        ("ThinkPad", "Lenovo"),
        ("PlayStation", "Sony"),
        ("Surface Pro", "Microsoft")
    ],
    "P463": [  # member_of
        ("Germany", "European Union"),
        ("Brazil", "BRICS"),
        ("Japan", "G7"),
        ("Canada", "NATO"),
        ("Australia", "Commonwealth")
    ],
    "P407": [  # language_of_work
        ("Don Quixote", "Spanish"),
        ("War and Peace", "Russian"),
        ("Les Misérables", "French"),
        ("The Great Gatsby", "English"),
        ("Faust", "German")
    ]
}

def generate_template_variations(facts, templates):
    """
    Generate different template variations for a set of facts.
    
    Args:
        facts: List of (X, Y) tuples representing factual relationships
        templates: List of template strings with [X] and [Y] placeholders
    
    Returns:
        Dict mapping each fact to its template variations
    """
    variations = {}
    
    for x, y in facts:
        fact_variations = []
        for template in templates:
            # Replace placeholders with actual entities
            text = template.replace("[X]", x).replace("[Y]", y)
            fact_variations.append(text)
        variations[(x, y)] = fact_variations
    
    return variations

def test_template_robustness(model, tokenizer, specialty_neurons, relation_type="P176", layer_idx=0):
    """
    Test robustness of specialty neurons to different template expressions of the same facts.
    
    Args:
        model: The language model
        tokenizer: Model tokenizer
        specialty_neurons: Dict of {class: [(neuron_idx, score), ...]}
        relation_type: Which relation type to test (P176, P463, P407)
        layer_idx: Layer to analyze
    
    Returns:
        Dict with template robustness results
    """
    if relation_type not in RELATION_TEMPLATES:
        raise ValueError(f"Unknown relation type: {relation_type}")
    
    relation_info = RELATION_TEMPLATES[relation_type]
    facts = EXAMPLE_FACTS[relation_type]
    templates = relation_info["templates"]
    
    print(f"\nTesting template robustness for relation: {relation_info['relation_name']}")
    print(f"Templates: {templates}")
    print(f"Testing {len(facts)} facts")
    
    # Generate template variations
    template_variations = generate_template_variations(facts, templates)
    
    results = {
        'fact_results': {},
        'overall_stats': {},
        'template_consistency': {}
    }
    
    # Analyze each fact across different templates
    for fact, variations in template_variations.items():
        x, y = fact
        print(f"\nAnalyzing fact: {x} -> {y}")
        
        fact_activations = []
        template_labels = []
        
        # Get activations for each template variation
        for i, text in enumerate(variations):
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                
                if layer_idx < len(outputs.hidden_states):
                    hidden_states = outputs.hidden_states[layer_idx]
                    # Average over sequence length
                    activation = hidden_states.mean(dim=1).squeeze(0).cpu().numpy()
                    fact_activations.append(activation)
                    template_labels.append(f"Template_{i+1}")
                    
                    print(f"  Template {i+1}: '{text}' -> activation shape: {activation.shape}")
                else:
                    print(f"Layer {layer_idx} not available")
                    continue
                    
            except Exception as e:
                print(f"  Error processing template {i+1}: {e}")
                continue
        
        if len(fact_activations) < 2:
            print(f"  Insufficient activations for fact {fact}")
            continue
        
        # Calculate consistency metrics
        fact_activations = np.array(fact_activations)
        
        # Pairwise correlations between templates
        correlations = []
        activation_distances = []
        
        for i in range(len(fact_activations)):
            for j in range(i+1, len(fact_activations)):
                # Overall correlation
                corr = np.corrcoef(fact_activations[i], fact_activations[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
                
                # Euclidean distance
                dist = np.linalg.norm(fact_activations[i] - fact_activations[j])
                activation_distances.append(dist)
        
        # Focus on specialty neurons if available
        specialty_correlations = []
        specialty_distances = []
        
        # Try to find relevant specialty neurons (use first available class)
        top_neurons = []
        if specialty_neurons:
            first_class = list(specialty_neurons.keys())[0]
            top_neurons = [neuron_idx for neuron_idx, _ in specialty_neurons[first_class][:10]]
            
            if top_neurons:
                specialty_activations = fact_activations[:, top_neurons]
                
                for i in range(len(specialty_activations)):
                    for j in range(i+1, len(specialty_activations)):
                        corr = np.corrcoef(specialty_activations[i], specialty_activations[j])[0, 1]
                        if not np.isnan(corr):
                            specialty_correlations.append(corr)
                        
                        dist = np.linalg.norm(specialty_activations[i] - specialty_activations[j])
                        specialty_distances.append(dist)
        
        # Store results for this fact
        results['fact_results'][fact] = {
            'templates': variations,
            'num_templates': len(fact_activations),
            'overall_correlation_mean': np.mean(correlations) if correlations else 0,
            'overall_correlation_std': np.std(correlations) if correlations else 0,
            'overall_distance_mean': np.mean(activation_distances) if activation_distances else 0,
            'overall_distance_std': np.std(activation_distances) if activation_distances else 0,
            'specialty_correlation_mean': np.mean(specialty_correlations) if specialty_correlations else 0,
            'specialty_correlation_std': np.std(specialty_correlations) if specialty_correlations else 0,
            'specialty_distance_mean': np.mean(specialty_distances) if specialty_distances else 0,
            'specialty_distance_std': np.std(specialty_distances) if specialty_distances else 0,
            'top_neurons_used': len(top_neurons)
        }
    
    # Calculate overall statistics
    if results['fact_results']:
        all_overall_corrs = [r['overall_correlation_mean'] for r in results['fact_results'].values()]
        all_specialty_corrs = [r['specialty_correlation_mean'] for r in results['fact_results'].values() if r['specialty_correlation_mean'] > 0]
        
        results['overall_stats'] = {
            'mean_fact_correlation': np.mean(all_overall_corrs),
            'std_fact_correlation': np.std(all_overall_corrs),
            'mean_specialty_correlation': np.mean(all_specialty_corrs) if all_specialty_corrs else 0,
            'std_specialty_correlation': np.std(all_specialty_corrs) if all_specialty_corrs else 0,
            'num_facts_analyzed': len(results['fact_results']),
            'relation_type': relation_type,
            'relation_name': relation_info['relation_name']
        }
    
    return results

def analyze_template_robustness_results(results):
    """Analyze and display template robustness test results."""
    print("\n" + "="*60)
    print("TEMPLATE ROBUSTNESS ANALYSIS")
    print("="*60)
    
    if 'overall_stats' in results and results['overall_stats']:
        stats = results['overall_stats']
        print(f"\nRelation: {stats['relation_name']} ({stats['relation_type']})")
        print(f"Facts analyzed: {stats['num_facts_analyzed']}")
        print(f"Overall template consistency: {stats['mean_fact_correlation']:.4f} ± {stats['std_fact_correlation']:.4f}")
        
        if stats['mean_specialty_correlation'] > 0:
            print(f"Specialty neuron consistency: {stats['mean_specialty_correlation']:.4f} ± {stats['std_specialty_correlation']:.4f}")
        
        # Interpret robustness
        consistency = stats['mean_fact_correlation']
        if consistency > 0.8:
            robustness_level = "HIGH - Templates produce very similar representations"
        elif consistency > 0.6:
            robustness_level = "MODERATE - Templates produce somewhat similar representations"
        elif consistency > 0.4:
            robustness_level = "LOW - Templates produce quite different representations"
        else:
            robustness_level = "VERY LOW - Templates produce very different representations"
        
        print(f"Robustness Level: {robustness_level}")
    
    print("\n" + "-"*40 + " DETAILED RESULTS " + "-"*40)
    
    # Detailed results for each fact
    for fact, fact_results in results.get('fact_results', {}).items():
        x, y = fact
        print(f"\nFact: {x} -> {y}")
        print(f"Templates tested: {fact_results['num_templates']}")
        
        for i, template in enumerate(fact_results['templates']):
            print(f"  {i+1}. '{template}'")
        
        print(f"Overall consistency: {fact_results['overall_correlation_mean']:.4f} ± {fact_results['overall_correlation_std']:.4f}")
        print(f"Average activation distance: {fact_results['overall_distance_mean']:.4f}")
        
        if fact_results['specialty_correlation_mean'] > 0:
            print(f"Specialty neuron consistency: {fact_results['specialty_correlation_mean']:.4f} ± {fact_results['specialty_correlation_std']:.4f}")
            print(f"Specialty neurons analyzed: {fact_results['top_neurons_used']}")

def get_available_relations():
    """Get list of available relation types."""
    return list(RELATION_TEMPLATES.keys())

def get_relation_info(relation_type):
    """Get information about a specific relation type."""
    return RELATION_TEMPLATES.get(relation_type)

def run_comprehensive_template_analysis(model, tokenizer, specialty_neurons, layer_idx=0):
    """
    Run template robustness analysis for all available relation types.
    
    Args:
        model: The language model
        tokenizer: Model tokenizer  
        specialty_neurons: Results from specialty neuron analysis
        layer_idx: Which layer to analyze
    
    Returns:
        Dict with results for all relation types
    """
    all_results = {}
    
    for relation_type in RELATION_TEMPLATES.keys():
        print(f"\n{'='*20} ANALYZING {relation_type} {'='*20}")
        
        try:
            results = test_template_robustness(
                model, tokenizer, specialty_neurons, 
                relation_type=relation_type, layer_idx=layer_idx
            )
            all_results[relation_type] = results
            analyze_template_robustness_results(results)
            
        except Exception as e:
            print(f"Error analyzing {relation_type}: {e}")
            continue
    
    # Summary across all relations
    print(f"\n{'='*60}")
    print("CROSS-RELATION SUMMARY")
    print("="*60)
    
    for relation_type, results in all_results.items():
        if 'overall_stats' in results and results['overall_stats']:
            stats = results['overall_stats']
            consistency = stats['mean_fact_correlation']
            print(f"{stats['relation_name']:20} | Consistency: {consistency:.4f} | Facts: {stats['num_facts_analyzed']}")
    
    return all_results

# Example usage function
def run_template_robustness_test():
    """Example of how to run the template robustness test."""
    
    # This would be called after loading your model and finding specialty neurons
    #
    # # Load model and tokenizer
    # model_name = "state-spaces/mamba-130m-hf"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # 
    # # Find specialty neurons first (optional - can pass empty dict)
    # class_texts = {
    #     "factual": [
    #         "Apple produces the iPhone smartphone device.",
    #         "Germany is a member of the European Union.",
    #         "Don Quixote was written in Spanish language.",
    #     ]
    # }
    # specialty_neurons = find_specialty_neurons_fixed(model, tokenizer, class_texts)
    #
    # # Run template robustness analysis
    # template_results = run_comprehensive_template_analysis(
    #     model, tokenizer, specialty_neurons, layer_idx=0
    # )
    #
    # # Or test a single relation type
    # single_results = test_template_robustness(
    #     model, tokenizer, specialty_neurons, 
    #     relation_type="P176", layer_idx=0
    # )
    # analyze_template_robustness_results(single_results)
    
    print("Template robustness test setup complete!")
    print("This tests whether models produce consistent internal representations")
    print("when the same factual relationship is expressed using different templates.")
    print("\nRelation types available:")
    for rel_id, info in RELATION_TEMPLATES.items():
        print(f"  {rel_id}: {info['relation_name']}")
        for template in info['templates']:
            print(f"    - {template}")

if __name__ == "__main__":
    run_template_robustness_test()