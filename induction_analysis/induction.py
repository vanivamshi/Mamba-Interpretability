import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

class MambaInductionAnalyzer:
    """
    Analyzes induction head behavior in Mamba models by examining SSM states
    and gate activations during sequence processing.
    """
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
        # Storage for intermediate activations
        self.ssm_states = {}
        self.gate_activations = {}
        self.delta_params = {}
        
    def register_hooks(self, layer_indices: Optional[List[int]] = None):
        """Register forward hooks to capture SSM states and gates."""
        if layer_indices is None:
            layer_indices = range(len(self.model.backbone.layers))
        
        self.hooks = []
        
        for layer_idx in layer_indices:
            layer = self.model.backbone.layers[layer_idx]
            
            # Hook into the Mamba mixer
            if hasattr(layer, 'mixer'):
                hook = layer.mixer.register_forward_hook(
                    self._create_hook(layer_idx)
                )
                self.hooks.append(hook)
    
    def _create_hook(self, layer_idx):
        """Create a hook function for a specific layer."""
        def hook_fn(module, input, output):
            # Store the output (hidden states after SSM)
            self.ssm_states[layer_idx] = output.detach()
            
            # Try to capture intermediate computations
            # Mamba's SSM involves: x_proj, dt_proj, conv1d, SSM scan
            if hasattr(module, 'dt_proj'):
                # Delta (Δ) controls how much to update state
                # This is key for understanding what the model "attends" to
                try:
                    x = input[0]
                    # Get delta parameters
                    delta = module.dt_proj(x)
                    self.delta_params[layer_idx] = delta.detach()
                except:
                    pass
        
        return hook_fn
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def analyze_induction_sequence(self, sequence: str, return_tokens=False):
        """
        Analyze how Mamba processes an induction sequence.
        
        Args:
            sequence: Input sequence (e.g., "A B C A B C")
            return_tokens: Whether to return tokenized sequence
        
        Returns:
            Dictionary with induction analysis results
        """
        # Tokenize
        tokens = self.tokenizer(sequence, return_tensors="pt").to(self.device)
        input_ids = tokens['input_ids'][0]
        
        # Clear previous activations
        self.ssm_states.clear()
        self.gate_activations.clear()
        self.delta_params.clear()
        
        # Register hooks
        self.register_hooks()
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**tokens, output_hidden_states=True)
        
        # Remove hooks
        self.remove_hooks()
        
        # Analyze induction patterns
        results = {
            'input_ids': input_ids.cpu(),
            'tokens': [self.tokenizer.decode([tid]) for tid in input_ids],
            'hidden_states': outputs.hidden_states,
            'ssm_states': self.ssm_states,
            'delta_params': self.delta_params,
            'induction_scores': self._compute_induction_scores(input_ids, outputs.hidden_states),
            'state_similarity': self._compute_state_similarity(outputs.hidden_states),
        }
        
        if return_tokens:
            results['tokenized'] = tokens
        
        return results
    
    def _compute_induction_scores(self, input_ids, hidden_states):
        """
        Compute induction scores by measuring how much the model recognizes
        repeated patterns and predicts based on them.
        """
        seq_len = len(input_ids)
        induction_scores = torch.zeros(seq_len)
        
        # For each position, check if we've seen this token before
        for i in range(1, seq_len):
            current_token = input_ids[i]
            
            # Find previous occurrences of this token
            prev_positions = (input_ids[:i] == current_token).nonzero(as_tuple=True)[0]
            
            if len(prev_positions) > 0:
                # Compare hidden states at current position with previous occurrences
                current_hidden = hidden_states[-1][0, i]  # Last layer, current position
                
                similarities = []
                for prev_pos in prev_positions:
                    prev_hidden = hidden_states[-1][0, prev_pos]
                    # Cosine similarity
                    sim = torch.nn.functional.cosine_similarity(
                        current_hidden.unsqueeze(0),
                        prev_hidden.unsqueeze(0)
                    )
                    similarities.append(sim.item())
                
                # Induction score is the max similarity with previous occurrences
                induction_scores[i] = max(similarities) if similarities else 0.0
        
        return induction_scores.cpu().numpy()
    
    def _compute_state_similarity(self, hidden_states):
        """
        Compute similarity matrix between all positions in the sequence.
        This shows which positions the model treats as similar.
        """
        last_hidden = hidden_states[-1][0]  # Last layer, batch dim removed
        seq_len = last_hidden.shape[0]
        
        similarity_matrix = torch.zeros(seq_len, seq_len)
        
        for i in range(seq_len):
            for j in range(seq_len):
                sim = torch.nn.functional.cosine_similarity(
                    last_hidden[i].unsqueeze(0),
                    last_hidden[j].unsqueeze(0)
                )
                similarity_matrix[i, j] = sim.item()
        
        return similarity_matrix.cpu().numpy()
    
    def analyze_ssm_gates(self, layer_idx=0):
        """
        Analyze the SSM gate activations (delta parameters) for a specific layer.
        Delta controls how much to update the hidden state at each timestep.
        """
        if layer_idx not in self.delta_params:
            return None
        
        delta = self.delta_params[layer_idx]
        
        # Delta analysis
        analysis = {
            'mean_delta': delta.mean(dim=-1).squeeze().cpu().numpy(),
            'std_delta': delta.std(dim=-1).squeeze().cpu().numpy(),
            'max_delta': delta.max(dim=-1)[0].squeeze().cpu().numpy(),
            'min_delta': delta.min(dim=-1)[0].squeeze().cpu().numpy(),
        }
        
        return analysis
    
    def visualize_induction_analysis(self, results, save_path=None):
        """
        Create comprehensive visualization of induction head behavior.
        """
        tokens = results['tokens']
        induction_scores = results['induction_scores']
        state_similarity = results['state_similarity']
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.5, wspace=0.4)
        
        # Plot 1: Token sequence with induction scores
        ax1 = fig.add_subplot(gs[0, :])
        x_pos = np.arange(len(tokens))
        bars = ax1.bar(x_pos, induction_scores, color='skyblue', alpha=0.7)
        
        # Color bars based on induction strength
        colors = ['red' if score > 0.5 else 'skyblue' for score in induction_scores]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax1.set_xlabel('Token Position', fontsize=14)
        ax1.set_ylabel('Induction Score', fontsize=14)
        ax1.set_title('Induction Head Activation Across Sequence', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f"{i}:{t.strip()}" for i, t in enumerate(tokens)], rotation=45, ha='right')
        ax1.tick_params(axis='both', labelsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Strong Induction Threshold')
        ax1.legend(fontsize=14)
        
        # Plot 2: State similarity heatmap
        ax2 = fig.add_subplot(gs[1, 0])
        im = ax2.imshow(state_similarity, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
        ax2.set_xlabel('Token Position', fontsize=14)
        ax2.set_ylabel('Token Position', fontsize=14)
        ax2.set_title('Hidden State Similarity Matrix', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(tokens)))
        ax2.set_yticks(range(len(tokens)))
        ax2.set_xticklabels([t.strip() for t in tokens], rotation=45, ha='right')
        ax2.set_yticklabels([t.strip() for t in tokens])
        ax2.tick_params(axis='both', labelsize=14)
        cbar2 = plt.colorbar(im, ax=ax2, label='Cosine Similarity')
        cbar2.set_label('Cosine Similarity', fontsize=14)
        cbar2.ax.tick_params(labelsize=14)
        
        # Plot 3: Layer-wise hidden state norms
        ax3 = fig.add_subplot(gs[1, 1])
        hidden_states = results['hidden_states']
        layer_norms = []
        for layer_hidden in hidden_states:
            norms = torch.norm(layer_hidden[0], dim=-1).cpu().numpy()
            layer_norms.append(norms)
        
        layer_norms = np.array(layer_norms)
        im3 = ax3.imshow(layer_norms, cmap='viridis', aspect='auto')
        ax3.set_xlabel('Token Position', fontsize=14)
        ax3.set_ylabel('Layer', fontsize=14)
        ax3.set_title('Hidden State Norms Across Layers', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(len(tokens)))
        ax3.set_xticklabels([t.strip() for t in tokens], rotation=45, ha='right')
        ax3.tick_params(axis='both', labelsize=14)
        cbar3 = plt.colorbar(im3, ax=ax3, label='L2 Norm')
        cbar3.set_label('L2 Norm', fontsize=14)
        cbar3.ax.tick_params(labelsize=14)
        
        # Plot 4: SSM delta parameters (if available)
        if results['delta_params']:
            ax4 = fig.add_subplot(gs[2, 0])
            layer_idx = list(results['delta_params'].keys())[0]
            delta_analysis = self.analyze_ssm_gates(layer_idx)
            
            if delta_analysis:
                x = range(len(delta_analysis['mean_delta']))
                ax4.plot(x, delta_analysis['mean_delta'], 'o-', label='Mean Δ', linewidth=2)
                ax4.fill_between(x, 
                                delta_analysis['mean_delta'] - delta_analysis['std_delta'],
                                delta_analysis['mean_delta'] + delta_analysis['std_delta'],
                                alpha=0.3, label='± Std')
                ax4.set_xlabel('Token Position', fontsize=14)
                ax4.set_ylabel('Delta (Δ) Value', fontsize=14)
                ax4.set_title(f'SSM Gate Activations (Layer {layer_idx})', fontsize=14, fontweight='bold')
                ax4.set_xticks(range(len(tokens)))
                ax4.set_xticklabels([t.strip() for t in tokens], rotation=45, ha='right')
                ax4.tick_params(axis='both', labelsize=14)
                ax4.legend(fontsize=14)
                ax4.grid(True, alpha=0.3)
        
        # Plot 5: Induction pattern detection
        ax5 = fig.add_subplot(gs[2, 1])
        input_ids = results['input_ids']
        
        # Find repeated tokens and their patterns
        pattern_info = []
        for i in range(len(input_ids)):
            token = input_ids[i]
            prev_occurrences = (input_ids[:i] == token).nonzero(as_tuple=True)[0]
            
            if len(prev_occurrences) > 0:
                pattern_info.append({
                    'position': i,
                    'token': tokens[i].strip(),
                    'prev_positions': prev_occurrences.tolist(),
                    'induction_score': induction_scores[i]
                })
        
        if pattern_info:
            y_pos = np.arange(len(pattern_info))
            scores = [p['induction_score'] for p in pattern_info]
            ax5.barh(y_pos, scores, color='coral', alpha=0.7)
            ax5.set_yticks(y_pos)
            ax5.set_yticklabels([f"Pos {p['position']}: '{p['token']}'" for p in pattern_info])
            ax5.set_xlabel('Induction Score', fontsize=14)
            ax5.set_title('Detected Induction Patterns', fontsize=14, fontweight='bold')
            ax5.tick_params(axis='both', labelsize=14)
            ax5.grid(True, alpha=0.3, axis='x')
        else:
            ax5.text(0.5, 0.5, 'No repeated patterns detected', 
                    ha='center', va='center', fontsize=14)
            ax5.set_xlim(0, 1)
        
        plt.suptitle('Mamba Induction Head Analysis', fontsize=14, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
            print(f"Visualization saved to {save_path}")
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig
    
    def compare_predictions(self, sequence: str):
        """
        Compare model predictions at positions where induction should occur.
        """
        results = self.analyze_induction_sequence(sequence)
        input_ids = results['input_ids']
        tokens = results['tokens']
        
        predictions = []
        
        with torch.no_grad():
            for i in range(len(input_ids) - 1):
                # Get logits at position i
                partial_input = input_ids[:i+1].unsqueeze(0).to(self.device)
                outputs = self.model(partial_input)
                logits = outputs.logits[0, -1]
                
                # Get top predictions
                top_k = torch.topk(logits, k=5)
                predicted_tokens = [self.tokenizer.decode([tid]) for tid in top_k.indices]
                predicted_probs = torch.softmax(top_k.values, dim=0).cpu().numpy()
                
                actual_next = tokens[i+1].strip()
                
                predictions.append({
                    'position': i,
                    'current_token': tokens[i].strip(),
                    'actual_next': actual_next,
                    'top_predictions': list(zip(predicted_tokens, predicted_probs)),
                    'induction_score': results['induction_scores'][i],
                    'correct': actual_next in predicted_tokens
                })
        
        return predictions


def create_induction_test_sequences():
    """Generate test sequences with clear induction patterns."""
    return [
        "A B C A B C",
        "cat dog bird cat dog bird",
        "1 2 3 4 1 2 3 4",
        "The cat sat on the mat. The cat sat on",
        "red blue green red blue green",
        "hello world hello world hello",
    ]


# Integration function for your existing pipeline
def integrate_induction_analysis(model, tokenizer, sequences=None, 
                                 layer_indices=None, save_dir='plots'):
    """
    Integrate induction analysis into existing Mamba interpretability pipeline.
    """
    import os
    from datetime import datetime
    
    if sequences is None:
        sequences = create_induction_test_sequences()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize analyzer
    device = next(model.parameters()).device
    analyzer = MambaInductionAnalyzer(model, tokenizer, device)
    
    results = {}
    
    for idx, sequence in enumerate(sequences):
        print(f"\nAnalyzing sequence {idx+1}/{len(sequences)}: {sequence}")
        
        # Analyze induction behavior
        result = analyzer.analyze_induction_sequence(sequence)
        results[f"sequence_{idx}"] = result
        
        # Visualize
        save_path = os.path.join(save_dir, f'induction_analysis_seq{idx}_{timestamp}.png')
        analyzer.visualize_induction_analysis(result, save_path=save_path)
        
        # Get predictions
        predictions = analyzer.compare_predictions(sequence)
        results[f"sequence_{idx}_predictions"] = predictions
        
        # Print summary
        print(f"Sequence: {sequence}")
        print(f"Mean induction score: {result['induction_scores'].mean():.3f}")
        print(f"Max induction score: {result['induction_scores'].max():.3f}")
        
        # Show where strong induction occurs
        strong_induction = np.where(result['induction_scores'] > 0.5)[0]
        if len(strong_induction) > 0:
            print(f"Strong induction at positions: {strong_induction.tolist()}")
            for pos in strong_induction:
                print(f"  Position {pos}: '{result['tokens'][pos].strip()}' " +
                      f"(score: {result['induction_scores'][pos]:.3f})")
    
    return results, analyzer