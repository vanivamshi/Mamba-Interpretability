#!/usr/bin/env python3
"""
delta_sensitivity_transformer.py

Implements an equivalent analysis to Mamba's delta-sensitive neurons
for Transformer models by measuring neuron activation differences
when the input sequence order is permuted.

This reveals neurons sensitive to sequence order.
"""

import numpy as np
from typographical_errors_robustness import extract_activation_vector


def permute_text(text, strategy="reverse"):
    """
    Permute a text sequence according to the chosen strategy.
    Available strategies:
      - reverse: reverse token order
      - shuffle: randomly shuffle tokens
      - swap: swap first and last token
    """
    tokens = text.strip().split()
    if len(tokens) < 2:
        return text

    if strategy == "reverse":
        permuted_tokens = tokens[::-1]
    elif strategy == "shuffle":
        permuted_tokens = tokens[:]
        np.random.shuffle(permuted_tokens)
    elif strategy == "swap":
        permuted_tokens = tokens[:]
        permuted_tokens[0], permuted_tokens[-1] = permuted_tokens[-1], permuted_tokens[0]
    else:
        raise ValueError(f"Unknown permutation strategy: {strategy}")

    return " ".join(permuted_tokens)


def find_delta_sensitive_neurons_transformer(
    model,
    tokenizer,
    texts,
    layer_idx=0,
    top_k=8,
    permutation_strategy="reverse"
):
    """
    Identify "delta-sensitive" neurons in transformers by measuring
    how neuron activations change when input tokens are permuted.
    """

    deltas = []

    for text in texts:
        # Original activation
        vec_original = extract_activation_vector(
            model, tokenizer, text, layer_idx=layer_idx
        )

        # Permuted activation
        permuted_text = permute_text(text, strategy=permutation_strategy)
        vec_permuted = extract_activation_vector(
            model, tokenizer, permuted_text, layer_idx=layer_idx
        )

        # Compute delta
        delta_vec = vec_original - vec_permuted
        deltas.append(delta_vec)

    delta_matrix = np.vstack(deltas)

    # Variance per neuron across examples
    variance_per_neuron = np.var(delta_matrix, axis=0)

    # Find top-k delta-sensitive neurons
    top_indices = np.argsort(-variance_per_neuron)[:top_k]
    top_variances = variance_per_neuron[top_indices]

    # Return list of (neuron index, variance)
    return [(int(idx), float(var)) for idx, var in zip(top_indices, top_variances)]
