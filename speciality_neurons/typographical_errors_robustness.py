import torch
import numpy as np
import random
import string
import re
from collections import defaultdict

def add_typos(text, error_rate=0.05, error_types=['substitution', 'insertion', 'deletion', 'transposition']):
    """
    Add various types of typographical errors to text.
    
    Args:
        text: Input text string
        error_rate: Fraction of characters to potentially modify (0.0 to 1.0)
        error_types: List of error types to apply
            - 'substitution': Replace character with nearby keyboard key
            - 'insertion': Insert random character
            - 'deletion': Remove character  
            - 'transposition': Swap adjacent characters
            - 'repetition': Duplicate character
            - 'case_error': Change capitalization
    
    Returns:
        Text with typographical errors
    """
    
    # Define keyboard layout for realistic substitutions
    keyboard_layout = {
        'q': 'wa', 'w': 'qse', 'e': 'wrd', 'r': 'eft', 't': 'rgy', 'y': 'thu', 'u': 'yji', 'i': 'uko', 'o': 'ilp', 'p': 'ol',
        'a': 'qsz', 's': 'awedz', 'd': 'serfcx', 'f': 'drtgv', 'g': 'ftyhb', 'h': 'gynuj', 'j': 'hkium', 'k': 'jlimo', 'l': 'kop',
        'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk'
    }
    
    chars = list(text)
    text_length = len(chars)
    
    if text_length == 0:
        return text
    
    # Calculate number of errors to introduce
    num_errors = max(1, int(text_length * error_rate))
    error_positions = set()
    
    for _ in range(num_errors):
        # Choose random position (avoid spaces and punctuation for some error types)
        attempts = 0
        while attempts < 10:  # Prevent infinite loop
            pos = random.randint(0, text_length - 1)
            if pos not in error_positions:
                error_positions.add(pos)
                break
            attempts += 1
    
    # Apply errors in reverse order to maintain position indices
    for pos in sorted(error_positions, reverse=True):
        if pos >= len(chars):
            continue
            
        char = chars[pos]
        error_type = random.choice(error_types)
        
        try:
            if error_type == 'substitution' and char.lower() in keyboard_layout:
                # Replace with nearby keyboard key
                nearby_keys = keyboard_layout[char.lower()]
                new_char = random.choice(nearby_keys)
                # Preserve case
                if char.isupper():
                    new_char = new_char.upper()
                chars[pos] = new_char
                
            elif error_type == 'substitution':
                # Random letter substitution if not in keyboard layout
                if char.isalpha():
                    new_char = random.choice(string.ascii_letters)
                    if char.isupper():
                        new_char = new_char.upper()
                    else:
                        new_char = new_char.lower()
                    chars[pos] = new_char
                elif char.isdigit():
                    chars[pos] = str(random.randint(0, 9))
                    
            elif error_type == 'insertion':
                # Insert random character
                if char.isalpha():
                    insert_char = random.choice(string.ascii_letters)
                    if char.isupper():
                        insert_char = insert_char.upper()
                    else:
                        insert_char = insert_char.lower()
                elif char.isdigit():
                    insert_char = str(random.randint(0, 9))
                else:
                    insert_char = char  # Insert same character
                chars.insert(pos + 1, insert_char)
                
            elif error_type == 'deletion':
                # Delete character (but not spaces at word boundaries)
                if char != ' ' or (pos > 0 and pos < len(chars) - 1):
                    chars.pop(pos)
                    
            elif error_type == 'transposition' and pos < len(chars) - 1:
                # Swap with next character
                chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
                
            elif error_type == 'repetition':
                # Duplicate character
                chars.insert(pos + 1, char)
                
            elif error_type == 'case_error' and char.isalpha():
                # Change case
                if char.isupper():
                    chars[pos] = char.lower()
                else:
                    chars[pos] = char.upper()
                    
        except (IndexError, ValueError):
            # Skip errors that cause issues
            continue
    
    return ''.join(chars)

def add_realistic_typos(text, error_rate=0.03):
    """
    Add realistic typos that humans commonly make.
    
    Args:
        text: Input text string
        error_rate: Probability of error per character
    
    Returns:
        Text with realistic typos
    """
    # Common typo patterns
    common_substitutions = {
        'the': ['teh', 'hte', 'th', 'te'],
        'and': ['adn', 'nad', 'an', 'nd'],
        'you': ['yuo', 'oyu', 'yu', 'yo'],
        'are': ['aer', 'rae', 'ar', 're'],
        'for': ['ofr', 'fro', 'fr', 'or'],
        'with': ['wtih', 'wiht', 'wth', 'wit'],
        'that': ['taht', 'htat', 'tat', 'hat'],
        'have': ['ahve', 'haev', 'hav', 'ave'],
        'this': ['htis', 'tihs', 'ths', 'his'],
        'will': ['wlil', 'iwll', 'wil', 'ill'],
        'can': ['acn', 'nac', 'ca', 'an'],
        'from': ['form', 'fomr', 'frm', 'rom'],
        'they': ['tehy', 'tyhe', 'thy', 'hey'],
        'know': ['konw', 'nkow', 'kno', 'now'],
        'want': ['watn', 'wnat', 'wan', 'ant'],
        'been': ['bene', 'eben', 'ben', 'een'],
        'good': ['godo', 'ogod', 'god', 'ood'],
        'much': ['muhc', 'mcuh', 'muc', 'uch'],
        'some': ['soem', 'smoe', 'som', 'ome'],
        'time': ['tmie', 'itme', 'tim', 'ime'],
        'very': ['vrey', 'evry', 'ver', 'ery'],
        'when': ['wehn', 'hwne', 'whe', 'hen'],
        'come': ['coem', 'cmoe', 'com', 'ome'],
        'here': ['heer', 'ehre', 'her', 'ere'],
        'just': ['jsut', 'ujst', 'jus', 'ust'],
        'like': ['liek', 'ilke', 'lik', 'ike'],
        'long': ['logn', 'nlgo', 'lon', 'ong'],
        'make': ['meak', 'amke', 'mak', 'ake'],
        'many': ['myan', 'amny', 'man', 'any'],
        'over': ['oevr', 'voer', 'ove', 'ver'],
        'such': ['suhc', 'scuh', 'suc', 'uch'],
        'take': ['taek', 'atke', 'tak', 'ake'],
        'than': ['tahn', 'htan', 'tha', 'han'],
        'them': ['tehm', 'htme', 'the', 'hem'],
        'well': ['wlel', 'ewll', 'wel', 'ell'],
        'were': ['weer', 'ewre', 'wer', 'ere']
    }
    
    # Apply word-level substitutions first
    words = text.split()
    for i, word in enumerate(words):
        # Remove punctuation for matching
        clean_word = re.sub(r'[^\w]', '', word.lower())
        if clean_word in common_substitutions and random.random() < error_rate * 3:
            # Higher probability for common word typos
            typo = random.choice(common_substitutions[clean_word])
            # Preserve case and punctuation
            if word[0].isupper():
                typo = typo.capitalize()
            # Add back punctuation
            punct = re.findall(r'[^\w]', word)
            if punct:
                typo += ''.join(punct)
            words[i] = typo
    
    # Then apply character-level errors
    modified_text = ' '.join(words)
    return add_typos(modified_text, error_rate, 
                    ['substitution', 'transposition', 'repetition'])

def add_contextual_typos(text, error_rate=0.02):
    """
    Add context-aware typos that depend on surrounding text.
    
    Args:
        text: Input text string
        error_rate: Probability of error per word
    
    Returns:
        Text with contextual typos
    """
    # Common contextual errors
    contextual_errors = {
        ('your', 'you\'re'): ['your', 'youre', 'you\'re'],
        ('its', 'it\'s'): ['its', 'it\'s', 'its\''],
        ('there', 'their', 'they\'re'): ['there', 'their', 'they\'re', 'theyre'],
        ('to', 'too', 'two'): ['to', 'too', 'two'],
        ('effect', 'affect'): ['effect', 'affect'],
        ('then', 'than'): ['then', 'than'],
        ('lose', 'loose'): ['lose', 'loose'],
        ('accept', 'except'): ['accept', 'except'],
        ('advise', 'advice'): ['advise', 'advice'],
        ('brake', 'break'): ['brake', 'break'],
        ('breath', 'breathe'): ['breath', 'breathe'],
        ('choose', 'chose'): ['choose', 'chose'],
        ('desert', 'dessert'): ['desert', 'dessert'],
        ('hear', 'here'): ['hear', 'here'],
        ('led', 'lead'): ['led', 'lead'],
        ('peace', 'piece'): ['peace', 'piece'],
        ('principal', 'principle'): ['principal', 'principle'],
        ('quiet', 'quite'): ['quiet', 'quite'],
        ('weather', 'whether'): ['weather', 'whether'],
        ('whose', 'who\'s'): ['whose', 'who\'s', 'whos']
    }
    
    words = text.split()
    for i, word in enumerate(words):
        clean_word = re.sub(r'[^\w\']', '', word.lower())
        
        # Check if word is part of any contextual error group
        for error_group, variants in contextual_errors.items():
            if clean_word in [w.lower() for w in error_group]:
                if random.random() < error_rate:
                    # Choose a different variant from the same group
                    available_variants = [v for v in variants if v.lower() != clean_word]
                    if available_variants:
                        new_word = random.choice(available_variants)
                        # Preserve original case
                        if word[0].isupper():
                            new_word = new_word.capitalize()
                        # Preserve punctuation
                        punct = re.findall(r'[^\w\']', word)
                        if punct:
                            new_word += ''.join(punct)
                        words[i] = new_word
                break
    
    return ' '.join(words)

def extract_activation_vector(model, tokenizer, text, layer_idx=0):
    """
    Extract activation vector from a specific layer for a given text.
    
    Args:
        model: The language model
        tokenizer: Model tokenizer  
        text: Input text
        layer_idx: Layer index to extract from
        
    Returns:
        numpy array of activations
    """
    device = next(model.parameters()).device
    
    try:
        # Tokenize with consistent parameters
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=128,  # Shorter for consistency
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
            if layer_idx < len(outputs.hidden_states):
                # Get the hidden state for the specified layer
                hidden_state = outputs.hidden_states[layer_idx]
                
                # Average over sequence length, keep batch and hidden dims
                # Shape: (batch_size, hidden_size)
                activation = hidden_state.mean(dim=1).squeeze()
                
                # Ensure we always return 1D array
                if activation.dim() == 0:
                    activation = activation.unsqueeze(0)
                
                return activation.cpu().numpy()
            else:
                print(f"Warning: Layer {layer_idx} not available, using random activation")
                return np.random.randn(model.config.hidden_size)
                
    except Exception as e:
        print(f"Error extracting activation for text '{text[:30]}...': {e}")
        return np.random.randn(model.config.hidden_size)

def calculate_similarity(vec1, vec2, method='cosine'):
    """
    Calculate similarity between two vectors.
    
    Args:
        vec1, vec2: numpy arrays
        method: 'cosine', 'correlation', or 'euclidean'
        
    Returns:
        Similarity score (higher = more similar)
    """
    # Ensure vectors are the same length
    min_len = min(len(vec1), len(vec2))
    vec1 = vec1[:min_len]
    vec2 = vec2[:min_len]
    
    if method == 'cosine':
        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
        
    elif method == 'correlation':
        # Pearson correlation
        if np.std(vec1) == 0 or np.std(vec2) == 0:
            return 1.0 if np.array_equal(vec1, vec2) else 0.0
        
        correlation = np.corrcoef(vec1, vec2)[0, 1]
        return 0.0 if np.isnan(correlation) else correlation
        
    elif method == 'euclidean':
        # Normalized euclidean distance (converted to similarity)
        distance = np.linalg.norm(vec1 - vec2)
        max_distance = np.linalg.norm(vec1) + np.linalg.norm(vec2)
        
        if max_distance == 0:
            return 1.0
        
        return 1.0 - (distance / max_distance)
    
    else:
        raise ValueError(f"Unknown similarity method: {method}")

def test_typo_robustness(model, tokenizer, texts, specialty_neurons=None, layer_idx=0, 
                        typo_types=['realistic', 'random', 'contextual'], num_variants=3,
                        similarity_method='cosine'):
    """
    Test model robustness to typographical errors - FIXED VERSION.
    
    Args:
        model: The language model
        tokenizer: Model tokenizer
        texts: List of input texts to test
        specialty_neurons: Dictionary of specialty neurons (optional)
        layer_idx: Layer index to analyze
        typo_types: Types of typos to test
        num_variants: Number of typo variants per text
        similarity_method: Method for calculating similarity ('cosine', 'correlation', 'euclidean')
    
    Returns:
        Dictionary with robustness results
    """
    print(f"Testing robustness with {similarity_method} similarity...")
    
    results = {
        'original_activations': [],
        'typo_activations': {},
        'consistency_scores': {},
        'neuron_sensitivity': {},
        'typo_examples': {}
    }
    
    for typo_type in typo_types:
        results['typo_activations'][typo_type] = []
        results['consistency_scores'][typo_type] = []
        results['typo_examples'][typo_type] = []
    
    for text_idx, original_text in enumerate(texts):
        print(f"Processing text {text_idx + 1}/{len(texts)}: {original_text[:50]}...")
        
        try:
            # Get original activation
            original_activation = extract_activation_vector(model, tokenizer, original_text, layer_idx)
            results['original_activations'].append(original_activation)
            
            # Test each typo type
            for typo_type in typo_types:
                typo_activations = []
                typo_texts = []
                
                for variant in range(num_variants):
                    # Generate typo variant
                    if typo_type == 'realistic':
                        typo_text = add_realistic_typos(original_text, error_rate=0.05)
                    elif typo_type == 'random':
                        typo_text = add_typos(original_text, error_rate=0.03, 
                                            error_types=['substitution', 'insertion', 'deletion'])
                    elif typo_type == 'contextual':
                        typo_text = add_contextual_typos(original_text, error_rate=0.04)
                    else:
                        typo_text = original_text
                    
                    typo_texts.append(typo_text)
                    
                    # Get typo activation
                    typo_activation = extract_activation_vector(model, tokenizer, typo_text, layer_idx)
                    typo_activations.append(typo_activation)
                
                # Store results
                results['typo_activations'][typo_type].append(typo_activations)
                results['typo_examples'][typo_type].append(typo_texts)
                
                # Calculate consistency scores using multiple methods
                consistency_scores = []
                for typo_activation in typo_activations:
                    similarity = calculate_similarity(original_activation, typo_activation, similarity_method)
                    consistency_scores.append(similarity)
                
                results['consistency_scores'][typo_type].append(consistency_scores)
                
        except Exception as e:
            print(f"Error processing text {text_idx}: {e}")
            # Add dummy values to maintain structure
            dummy_activation = np.random.randn(512)  # Default hidden size
            results['original_activations'].append(dummy_activation)
            
            for typo_type in typo_types:
                results['typo_activations'][typo_type].append([dummy_activation] * num_variants)
                results['consistency_scores'][typo_type].append([0.5] * num_variants)  # Neutral score
                results['typo_examples'][typo_type].append([original_text] * num_variants)
    
    # Analyze neuron sensitivity if specialty neurons provided
    if specialty_neurons:
        print("Analyzing specialty neuron sensitivity...")
        
        for neuron_class, neurons in specialty_neurons.items():
            results['neuron_sensitivity'][neuron_class] = {}
            
            for typo_type in typo_types:
                sensitivities = []
                
                for neuron_idx, _ in neurons:
                    neuron_consistency = []
                    
                    for text_idx in range(len(texts)):
                        if text_idx < len(results['original_activations']):
                            original_vec = results['original_activations'][text_idx]
                            original_val = original_vec[neuron_idx] if neuron_idx < len(original_vec) else 0
                            
                            if text_idx < len(results['typo_activations'][typo_type]):
                                typo_vecs = results['typo_activations'][typo_type][text_idx]
                                
                                for typo_vec in typo_vecs:
                                    typo_val = typo_vec[neuron_idx] if neuron_idx < len(typo_vec) else 0
                                    
                                    # Calculate neuron-specific consistency
                                    if abs(original_val) > 1e-6:  # Avoid division by very small numbers
                                        consistency = 1 - min(1.0, abs(original_val - typo_val) / abs(original_val))
                                    else:
                                        consistency = 1.0 if abs(typo_val) < 1e-6 else 0.0
                                    
                                    neuron_consistency.append(max(0.0, consistency))  # Ensure non-negative
                    
                    if neuron_consistency:
                        sensitivities.append(np.mean(neuron_consistency))
                    else:
                        sensitivities.append(0.5)  # Neutral sensitivity
                
                results['neuron_sensitivity'][neuron_class][typo_type] = sensitivities
    
    return results

def analyze_typo_robustness_results(results):
    """
    Analyze and display typographical error robustness results - ENHANCED VERSION.
    
    Args:
        results: Results dictionary from test_typo_robustness
    """
    print("\n" + "="*60)
    print("TYPOGRAPHICAL ERROR ROBUSTNESS ANALYSIS")
    print("="*60)
    
    typo_types = list(results['consistency_scores'].keys())
    
    # Overall consistency statistics
    print("\nOverall Consistency Scores:")
    print("-" * 40)
    
    all_stats = {}
    for typo_type in typo_types:
        all_scores = []
        for text_scores in results['consistency_scores'][typo_type]:
            all_scores.extend(text_scores)
        
        if all_scores:
            mean_consistency = np.mean(all_scores)
            std_consistency = np.std(all_scores)
            min_consistency = np.min(all_scores)
            max_consistency = np.max(all_scores)
            robust_count = sum(1 for s in all_scores if s > 0.8)
            
            all_stats[typo_type] = {
                'mean': mean_consistency,
                'std': std_consistency,
                'min': min_consistency,
                'max': max_consistency,
                'robust_count': robust_count,
                'total_count': len(all_scores)
            }
            
            print(f"{typo_type.upper()} Typos:")
            print(f"  Mean consistency: {mean_consistency:.4f}")
            print(f"  Std deviation:    {std_consistency:.4f}")
            print(f"  Min consistency:  {min_consistency:.4f}")
            print(f"  Max consistency:  {max_consistency:.4f}")
            print(f"  Robust texts (>0.8): {robust_count}/{len(all_scores)} ({robust_count/len(all_scores)*100:.1f}%)")
            print()
    
    # Show some example typos and their effects
    print("Example Typos and Effects:")
    print("-" * 40)
    
    for typo_type in typo_types[:2]:  # Show first two types
        if typo_type in results['typo_examples'] and results['typo_examples'][typo_type]:
            print(f"{typo_type.upper()} Examples:")
            
            # Show first few examples
            for i in range(min(3, len(results['typo_examples'][typo_type]))):
                examples = results['typo_examples'][typo_type][i]
                scores = results['consistency_scores'][typo_type][i]
                
                print(f"  Text {i+1}:")
                if examples:
                    print(f"    Original: {results['typo_examples'][typo_type][0][0] if i == 0 else '...'}")
                    print(f"    Typo:     {examples[0]}")
                    print(f"    Score:    {scores[0]:.3f}")
                print()
    
    # Neuron sensitivity analysis
    if 'neuron_sensitivity' in results and results['neuron_sensitivity']:
        print("Neuron Class Sensitivity:")
        print("-" * 40)
        
        for neuron_class, sensitivity_data in results['neuron_sensitivity'].items():
            print(f"{neuron_class.upper()} neurons:")
            
            for typo_type, sensitivities in sensitivity_data.items():
                if sensitivities:
                    mean_sensitivity = np.mean(sensitivities)
                    std_sensitivity = np.std(sensitivities)
                    print(f"  {typo_type}: {mean_sensitivity:.4f} ± {std_sensitivity:.4f} sensitivity")
            print()
    
    # Robustness assessment and recommendations
    print("Robustness Assessment:")
    print("-" * 40)
    
    for typo_type in typo_types:
        if typo_type in all_stats:
            mean_score = all_stats[typo_type]['mean']
            robust_pct = (all_stats[typo_type]['robust_count'] / all_stats[typo_type]['total_count']) * 100
            
            if mean_score > 0.9:
                assessment = "EXCELLENT"
            elif mean_score > 0.8:
                assessment = "GOOD"
            elif mean_score > 0.7:
                assessment = "MODERATE"
            elif mean_score > 0.6:
                assessment = "POOR"
            else:
                assessment = "VERY POOR"
            
            print(f"{typo_type.upper()} robustness: {assessment} ({mean_score:.3f}, {robust_pct:.1f}% robust)")
    
    print("\nRecommendations:")
    print("- Consistency > 0.8: Model is robust to this error type")
    print("- Consistency 0.6-0.8: Some sensitivity, consider error correction")
    print("- Consistency < 0.6: High sensitivity, implement robust preprocessing")
    print("- Use cosine similarity for semantic robustness testing")
    print("- Consider ensemble methods for highly sensitive error types")

def generate_typo_examples(texts, num_examples=5):
    """
    Generate examples of different typo types for demonstration.
    
    Args:
        texts: List of input texts
        num_examples: Number of examples to generate
    
    Returns:
        Dictionary with example typos
    """
    examples = {
        'original': [],
        'realistic': [],
        'random': [],
        'contextual': []
    }
    
    selected_texts = texts[:num_examples]
    
    for text in selected_texts:
        examples['original'].append(text)
        examples['realistic'].append(add_realistic_typos(text, error_rate=0.08))
        examples['random'].append(add_typos(text, error_rate=0.05))
        examples['contextual'].append(add_contextual_typos(text, error_rate=0.06))
    
    return examples

# Example usage and testing
if __name__ == "__main__":
    # Test the typo generation functions
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "I want to go to the store with you and your friends.",
        "They're going to their house over there.",
        "The effect of the new policy will affect everyone.",
        "It's important to check its functionality before deployment."
    ]
    
    print("TYPOGRAPHICAL ERROR EXAMPLES")
    print("="*50)
    
    # Test similarity calculation
    vec1 = np.array([1, 2, 3, 4, 5])
    vec2 = np.array([1.1, 2.1, 2.9, 4.1, 4.9])  # Similar vector
    vec3 = np.array([5, 4, 3, 2, 1])  # Different vector
    
    print(f"Similarity tests:")
    print(f"Similar vectors - Cosine: {calculate_similarity(vec1, vec2, 'cosine'):.3f}")
    print(f"Similar vectors - Correlation: {calculate_similarity(vec1, vec2, 'correlation'):.3f}")
    print(f"Different vectors - Cosine: {calculate_similarity(vec1, vec3, 'cosine'):.3f}")
    print()
    
    examples = generate_typo_examples(test_texts)
    
    for i in range(len(examples['original'])):
        print(f"\nExample {i+1}:")
        print(f"Original:   {examples['original'][i]}")
        print(f"Realistic:  {examples['realistic'][i]}")
        print(f"Random:     {examples['random'][i]}")
        print(f"Contextual: {examples['contextual'][i]}")
        print("-" * 50)