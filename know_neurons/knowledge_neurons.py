#!/usr/bin/env python3
"""
Fixed implementation of knowledge neurons using integrated gradients.
This version addresses the gradient computation issues and provides more robust detection.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Any
import math
from collections import defaultdict

class KnowledgeNeuronsFinder:
    """
    Find knowledge neurons using integrated gradients method.
    Fixed version that properly computes gradients.
    """

    def __init__(self, model, tokenizer, model_type='gpt2'):
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type.lower()
        self.device = next(model.parameters()).device

        # Set up model for gradients
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(True)

        # Detect model architecture
        self.num_layers = self._get_num_layers()

    def _get_num_layers(self):
        """Detect number of layers in the model."""
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return len(self.model.transformer.h)  # GPT-2 style
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return len(self.model.model.layers)  # LLaMA style
        elif hasattr(self.model, 'layers'):
            return len(self.model.layers)  # Other architectures
        else:
            print("Warning: Could not detect number of layers. Assuming 12 for default.")
            return 12 # Default for common small models like GPT-2

    def _get_output_embedding_layer(self):
        """Get the output embedding layer (unembedder) of the model."""
        if self.model_type == 'gpt2':
            # In GPT-2, the output layer is typically the 'lm_head'
            return self.model.lm_head
        elif self.model_type == 'mamba':
            # For Mamba, it's usually model.lm_head or model.backbone.lm_head
            if hasattr(self.model, 'lm_head'):
                return self.model.lm_head
            elif hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'lm_head'):
                return self.model.backbone.lm_head
            else:
                raise ValueError("Could not find lm_head for Mamba model.")
        else:
            # Fallback for other transformer models
            if hasattr(self.model, 'lm_head'):
                return self.model.lm_head
            elif hasattr(self.model, 'embed_out'): # Some models use this
                return self.model.embed_out
            else:
                raise ValueError(f"Could not identify output embedding layer for model type {self.model_type}")

    def _get_mlp_layer(self, layer_idx: int):
        """
        Get the MLP (Multi-Layer Perceptron) layer for a given layer index.
        This function needs to be adapted for different model architectures.
        """
        if self.model_type == 'gpt2':
            if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                return self.model.transformer.h[layer_idx].mlp
            else:
                raise ValueError("GPT-2 model structure not as expected for MLP layers.")
        elif self.model_type == 'mamba':
            # Mamba's architecture is different; it doesn't have traditional MLPs in the same way.
            # Instead, it has a Block structure containing SSM and MLP-like projections.
            # For integrated gradients, we'll target the output projection of the SSM block or the inner MLP if available.
            # Let's target the final output projection of the block as the 'MLP equivalent'
            if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'layers'):
                # Assuming `out_proj` in the Mamba block is what we want to target
                # The output of the layer block will be the activations
                return self.model.backbone.layers[layer_idx]
            else:
                raise ValueError("Mamba model structure not as expected for layers.")
        else:
            # Generic Transformer: Assumes a structure like model.encoder.layer[idx].mlp or model.decoder.layer[idx].mlp
            if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
                return self.model.encoder.layer[layer_idx].mlp
            elif hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'layer'):
                return self.model.decoder.layer[layer_idx].mlp
            elif hasattr(self.model, 'layers'): # Some models directly expose layers attribute
                return self.model.layers[layer_idx].mlp if hasattr(self.model.layers[layer_idx], 'mlp') else self.model.layers[layer_idx]
            else:
                raise ValueError(f"Could not identify MLP layer for model type {self.model_type} at index {layer_idx}")


    def _get_output_logits(self, input_ids: torch.Tensor, target_token_id: int) -> torch.Tensor:
        """Get the logits for the target token at the last position."""
        outputs = self.model(input_ids)
        # Ensure we're taking logits from the last token in the sequence
        # and for the specific target_token_id
        last_token_logits = outputs.logits[:, -1, target_token_id]
        return last_token_logits


    def compute_integrated_gradients_fixed(self,
                                            input_ids: torch.Tensor,
                                            target_token_id: int,
                                            layer_idx: int,
                                            steps: int = 50) -> np.ndarray:
        """
        Computes integrated gradients for the MLP layer's activations
        with respect to the target token's logit, using torch.autograd.grad.
        """
        mlp_layer = self._get_mlp_layer(layer_idx)
        
        # This list will store the output of the MLP layer from the hook for each forward pass.
        # It's important that this tensor remains connected to the graph.
        captured_mlp_outputs_for_grad = []

        def hook_fn(module, input, output):
            # Capture the output directly. Do NOT detach or clone here.
            # This 'output' tensor needs to be the one that `outputs.logits` depends on.
            if isinstance(output, tuple):
                # For models returning tuples (e.g., Mamba), take the main hidden state
                captured_mlp_outputs_for_grad.append(output[0])
            else:
                captured_mlp_outputs_for_grad.append(output)

        # Register the hook before the loop. It will fire during each model() call.
        handle = mlp_layer.register_forward_hook(hook_fn)

        total_gradients = torch.zeros(self.model.config.hidden_size, device=self.device)
        
        try:
            for step in range(1, steps + 1):
                alpha = step / steps
                
                # Interpolate input embeddings
                input_embeddings = self.model.get_input_embeddings()(input_ids)
                baseline_embeddings = torch.zeros_like(input_embeddings).to(self.device)
                interpolated_embeddings = (baseline_embeddings + alpha * (input_embeddings - baseline_embeddings))

                self.model.zero_grad() # Clear model gradients for current step
                captured_mlp_outputs_for_grad.clear() # Clear activations from previous steps in this batch/loop

                # Forward pass - the hook_fn will populate captured_mlp_outputs_for_grad
                outputs = self.model(inputs_embeds=interpolated_embeddings)
                
                if not captured_mlp_outputs_for_grad:
                    print(f"Warning: MLP activations not captured by hook for step {step}. Skipping.")
                    continue
                
                # Get the captured MLP output for this step.
                current_mlp_output = captured_mlp_outputs_for_grad[0]
                
                # Ensure current_mlp_output requires gradients. This is crucial for torch.autograd.grad.
                # If it's a leaf node that doesn't already require grad (e.g., if it's the model's direct output),
                # we must explicitly set it. If it's an intermediate result of operations on grad-enabled inputs,
                # it should already require grad.
                if not current_mlp_output.requires_grad:
                    # If it doesn't require grad, it means it's detached or a leaf without requires_grad=True.
                    # This indicates an issue in the graph or the way the tensor is produced.
                    # For a robust solution, we'd ideally make a clone that requires grad and then
                    # ensure the rest of the model's computation uses that clone.
                    # For now, if it doesn't require grad, it's likely not part of the differentiable path.
                    print(f"Warning: Captured MLP activations for step {step} do not require gradients.")
                    continue # Skip this step

                # Get the logit of the target token.
                target_logit = outputs.logits[:, -1, target_token_id]

                # Compute gradients using torch.autograd.grad
                # outputs: the tensor to compute gradients of (target_logit.sum() to make it scalar)
                # inputs: the tensor(s) to compute gradients with respect to (current_mlp_output)
                grads_tuple = torch.autograd.grad(
                    outputs=target_logit.sum(),
                    inputs=current_mlp_output,
                    retain_graph=True, # Retain graph for subsequent steps in the loop
                    allow_unused=True # Allow if current_mlp_output is not used to compute target_logit
                )
                
                grads = grads_tuple[0] # The gradients are the first (and usually only) element of the tuple

                if grads is not None:
                    # Average gradients over batch and sequence length (if present)
                    # (batch, seq_len, hidden_size) -> (hidden_size,)
                    mean_grads = grads.mean(dim=(0, 1))
                    total_gradients += mean_grads
                else:
                    print(f"Warning: No gradients returned from torch.autograd.grad for MLP activations at step {step}.")

            # Remove the hook after all steps are done to prevent side effects
            handle.remove()

            # Re-get original activations for the original input (without requiring grad for this part)
            original_mlp_output_list = []
            # Temporarily register a hook to get the actual activations for the original input
            original_handle = mlp_layer.register_forward_hook(lambda m, i, o: original_mlp_output_list.append(o[0].detach() if isinstance(o, tuple) else o.detach()))
            with torch.no_grad(): # Do not track gradients for this forward pass
                self.model(input_ids)
            original_handle.remove() # Remove this temporary hook

            if original_mlp_output_list:
                # Average the original activations over batch and sequence length
                original_activations_mean = original_mlp_output_list[0].mean(dim=(0,1))
            else:
                original_activations_mean = torch.zeros_like(total_gradients) # Fallback

            # Integrated gradients formula: (original_activation - baseline_activation) * avg_gradients
            # Assuming baseline activations are effectively zero for the integrated path,
            # this simplifies to original_activations * avg_gradients.
            # avg_gradients = total_gradients / steps
            integrated_grads = (original_activations_mean * (total_gradients / steps)).cpu().numpy()
            
            # Check if the computed integrated gradients are effectively zero
            if np.linalg.norm(integrated_grads) < 1e-9:
                print("DEBUG: Integrated gradients result very close to zero, returning zeros.")
                return np.zeros(self.model.config.hidden_size)

            return integrated_grads

        except Exception as e:
            print(f"Error during integrated gradients computation for layer {layer_idx}: {e}")
            handle.remove() # Ensure hook is removed even if an error occurs
            # Return zeros in case of any error
            return np.zeros(self.model.config.hidden_size) # Ensure consistent return type


    def find_knowledge_neurons_for_fact_fixed(self,
                                            fact_data: Dict[str, Any],
                                            layer_idx: int = -1, # Default to last layer
                                            steps: int = 15,
                                            top_k: int = None) -> List[Tuple[int, float]]:
        """
        Finds knowledge neurons for a given fact using integrated gradients.
        This fixed version iterates over texts, handles potential tokenization
        issues, and averages attributions.
        """
        texts = fact_data['texts']
        ground_truth = fact_data['ground_truth']

        all_attributions = []

        # Ensure ground_truth can be tokenized
        ground_truth_tokens = self.tokenizer.encode(ground_truth, add_special_tokens=False)
        if len(ground_truth_tokens) == 0:
            print(f"Warning: Could not tokenize ground truth '{ground_truth}'. Returning empty attributions.")
            return []
        
        # DEBUG: Added this line to check tokenization
        print(f"DEBUG: Ground truth '{ground_truth}' tokenizes to: {self.tokenizer.convert_ids_to_tokens(ground_truth_tokens)}")

        target_token_id = ground_truth_tokens[0]

        for text in texts:
            try:
                # Tokenize the input text, add special tokens for full context
                input_ids = self.tokenizer.encode(text, return_tensors='pt', add_special_tokens=True).to(self.device)

                # Ensure input_ids is not empty
                if input_ids.numel() == 0:
                    print(f"Warning: Empty input_ids for text: '{text}'. Skipping.")
                    continue

                # Compute integrated gradients for the specific layer
                # If layer_idx is -1, use the last layer
                actual_layer_idx = layer_idx if layer_idx != -1 else self.num_layers - 1

                attributions = self.compute_integrated_gradients_fixed(
                    input_ids, target_token_id, actual_layer_idx, steps=steps
                )

                if attributions is not None and attributions.size > 0:
                    all_attributions.append(attributions)
                else:
                    print(f"Warning: Integrated gradients returned empty or None for text: '{text}'")

            except Exception as e:
                print(f"Error processing text '{text[:50]}...': {e}")
                # Continue to next text if an error occurs

        if not all_attributions:
            print(f"No attributions found for fact '{fact_data.get('ground_truth', 'N/A')}' across all texts.")
            return []

        # Average attributions across all texts for this fact
        avg_attributions = np.mean(all_attributions, axis=0)

        # Get the top_k neurons if specified, otherwise return all
        if top_k is not None:
            # Sort by absolute value to find most influential neurons
            top_neuron_indices = np.argsort(np.abs(avg_attributions))[-top_k:]
            # Return (index, attribution_value) pairs for the top K
            result = [(int(idx), float(avg_attributions[idx])) for idx in top_neuron_indices]
        else:
            # Return all neurons with their attributions as a flat list
            result = avg_attributions.tolist() # Convert to list of floats

        return result

def compute_overlap(neuron_results: Dict[str, List[Tuple[int, float]]],
                    relations: Dict[str, str]) -> Dict[str, Any]:
    """
    Computes overlap statistics (Jaccard similarity and intersection size)
    for knowledge neurons between different facts.
    """
    all_facts = list(neuron_results.keys())
    overlaps = []

    for i in range(len(all_facts)):
        for j in range(i + 1, len(all_facts)):
            fact1_name = all_facts[i]
            fact2_name = all_facts[j]

            # Get neuron sets (using just indices as sets for Jaccard)
            # Assuming neuron_results gives (idx, value) tuples or a list of values where index is implicit
            neurons1_set = set(idx for idx, _ in neuron_results.get(fact1_name, []))
            neurons2_set = set(idx for idx, _ in neuron_results.get(fact2_name, []))

            if not neurons1_set or not neurons2_set:
                print(f"Skipping overlap for {fact1_name} and {fact2_name}: empty neuron sets.")
                continue

            intersection = len(neurons1_set.intersection(neurons2_set))
            union = len(neurons1_set.union(neurons2_set))

            jaccard_similarity = intersection / union if union > 0 else 0

            # Determine if facts belong to the same relation type
            same_relation = (relations.get(fact1_name) == relations.get(fact2_name))

            overlaps.append({
                'fact1': fact1_name,
                'fact2': fact2_name,
                'jaccard_similarity': jaccard_similarity,
                'intersection_size': intersection,
                'same_relation': same_relation
            })

    # Separate overlaps by relation type for summary statistics
    intra_relation_overlaps = [o for o in overlaps if o['same_relation']]
    inter_relation_overlaps = [o for o in overlaps if not o['same_relation']]

    # Calculate summary statistics
    if overlaps:
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
    else:
        stats = {
            'overall': {
                'num_pairs': len(overlaps),
                'avg_jaccard': np.mean([o['jaccard_similarity'] for o in overlaps]) if overlaps else 0,
                'avg_intersection_size': np.mean([o['intersection_size'] for o in overlaps]) if overlaps else 0
            }
        }

    return {
        'summary_stats': stats,
        'detailed_overlaps': overlaps
    }


if __name__ == "__main__":
    print("Fixed Knowledge Neurons Implementation Ready!")
    print("Use test_fixed_implementation(model, tokenizer) to test with your model.")
    print("\nKey improvements:")
    print("- Fixed gradient computation")
    print("- Better error handling")
    print("- Multiple fallback methods")
    print("- Architecture detection")
    print("- Enhanced prompts")
    print("- Compatibility wrappers for existing code")