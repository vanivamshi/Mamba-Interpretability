#!/usr/bin/env python3

"""
Investigate why Mamba shows strong induction scores but poor prediction accuracy.

This analyzes the logits and output head to understand the disconnect.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


class InductionPredictionAnalyzer:
    """
    Analyzes the gap between induction detection and actual predictions.
    """
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def analyze_logit_distribution(self, sequence: str, high_induction_positions: List[int]):
        """
        Analyze how logits are distributed at high vs low induction positions.
        """
        tokens = self.tokenizer(sequence, return_tensors="pt").to(self.device)
        input_ids = tokens['input_ids'][0]
        
        with torch.no_grad():
            outputs = self.model(**tokens, output_hidden_states=True)
            logits = outputs.logits[0]  # [seq_len, vocab_size]
        
        analysis = {
            'high_induction': [],
            'low_induction': [],
            'all_positions': []
        }
        
        for pos in range(len(input_ids) - 1):
            actual_next = input_ids[pos + 1].item()
            pos_logits = logits[pos]
            
            # Get probability distribution
            probs = torch.softmax(pos_logits, dim=0)
            
            # Entropy of distribution (how concentrated is it?)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            
            # Rank of correct token
            sorted_indices = torch.argsort(pos_logits, descending=True)
            rank = (sorted_indices == actual_next).nonzero(as_tuple=True)[0].item()
            
            # Probability assigned to correct token
            correct_prob = probs[actual_next].item()
            
            # Top-k accuracy
            top_5_indices = sorted_indices[:5]
            in_top_5 = actual_next in top_5_indices
            
            pos_data = {
                'position': pos,
                'entropy': entropy,
                'rank': rank,
                'correct_prob': correct_prob,
                'in_top_5': in_top_5,
                'actual_token': self.tokenizer.decode([actual_next]),
                'predicted_token': self.tokenizer.decode([sorted_indices[0]]),
            }
            
            analysis['all_positions'].append(pos_data)
            
            if pos in high_induction_positions:
                analysis['high_induction'].append(pos_data)
            else:
                analysis['low_induction'].append(pos_data)
        
        return analysis
    
    def analyze_hidden_to_logit_transform(self, sequence: str, position: int):
        """
        Analyze how hidden states are transformed into logits at a specific position.
        This helps understand if the issue is in the output head (lm_head).
        """
        tokens = self.tokenizer(sequence, return_tensors="pt").to(self.device)
        input_ids = tokens['input_ids'][0]
        
        with torch.no_grad():
            outputs = self.model(**tokens, output_hidden_states=True)
            
            # Get hidden state at this position (last layer)
            hidden = outputs.hidden_states[-1][0, position]  # [hidden_dim]
            
            # Get logits at this position
            logits = outputs.logits[0, position]  # [vocab_size]
        
        # Analyze the lm_head transformation
        if hasattr(self.model, 'lm_head'):
            lm_head = self.model.lm_head
            
            # Get weight matrix
            weight = lm_head.weight  # [vocab_size, hidden_dim]
            
            # Manual computation to verify
            manual_logits = torch.matmul(weight, hidden)
            
            # Check if they match
            match = torch.allclose(manual_logits, logits, atol=1e-5)
            
            return {
                'hidden_state': hidden.cpu(),
                'logits': logits.cpu(),
                'manual_logits': manual_logits.cpu(),
                'match': match,
                'lm_head_weight_norm': torch.norm(weight, dim=1).mean().item(),
                'hidden_norm': torch.norm(hidden).item(),
            }
        
        return None
    
    def compare_induction_contexts(self, sequence: str, 
                                   first_occurrence_pos: int,
                                   second_occurrence_pos: int):
        """
        Compare what happens at first vs second occurrence of a repeated sequence.
        This reveals why induction might fail.
        """
        tokens = self.tokenizer(sequence, return_tensors="pt").to(self.device)
        input_ids = tokens['input_ids'][0]
        
        with torch.no_grad():
            outputs = self.model(**tokens, output_hidden_states=True)
            logits = outputs.logits[0]
            hidden_states = outputs.hidden_states[-1][0]
        
        # First occurrence
        first_hidden = hidden_states[first_occurrence_pos]
        first_logits = logits[first_occurrence_pos]
        first_probs = torch.softmax(first_logits, dim=0)
        
        # Second occurrence (induction position)
        second_hidden = hidden_states[second_occurrence_pos]
        second_logits = logits[second_occurrence_pos]
        second_probs = torch.softmax(second_logits, dim=0)
        
        # What token should follow at second occurrence?
        # It should be the same as what followed the first occurrence
        if first_occurrence_pos + 1 < len(input_ids):
            expected_token = input_ids[first_occurrence_pos + 1].item()
            
            return {
                'first_occurrence': {
                    'position': first_occurrence_pos,
                    'hidden': first_hidden.cpu(),
                    'logits': first_logits.cpu(),
                    'top_5_tokens': [self.tokenizer.decode([i]) for i in torch.topk(first_logits, 5).indices],
                    'top_5_probs': torch.topk(first_probs, 5).values.cpu().numpy(),
                },
                'second_occurrence': {
                    'position': second_occurrence_pos,
                    'hidden': second_hidden.cpu(),
                    'logits': second_logits.cpu(),
                    'top_5_tokens': [self.tokenizer.decode([i]) for i in torch.topk(second_logits, 5).indices],
                    'top_5_probs': torch.topk(second_probs, 5).values.cpu().numpy(),
                },
                'expected_token': self.tokenizer.decode([expected_token]),
                'expected_token_id': expected_token,
                'expected_rank_at_second': (torch.argsort(second_logits, descending=True) == expected_token).nonzero(as_tuple=True)[0].item(),
                'expected_prob_at_second': second_probs[expected_token].item(),
                'hidden_similarity': torch.nn.functional.cosine_similarity(
                    first_hidden.unsqueeze(0), 
                    second_hidden.unsqueeze(0)
                ).item(),
                'logit_correlation': torch.corrcoef(torch.stack([first_logits, second_logits]))[0, 1].item(),
            }
        
        return None
    
    def visualize_prediction_gap(self, analysis_results, save_path=None):
        """
        Visualize why induction detection doesn't translate to predictions.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Induction Detection vs Prediction Gap Analysis', fontsize=14, fontweight='bold')
        
        # Extract data
        high_ind = analysis_results['high_induction']
        low_ind = analysis_results['low_induction']
        
        # Plot 1: Entropy comparison
        ax = axes[0, 0]
        if high_ind and low_ind:
            high_entropy = [p['entropy'] for p in high_ind]
            low_entropy = [p['entropy'] for p in low_ind]
            
            ax.boxplot([high_entropy, low_entropy], labels=['High\nInduction', 'Low\nInduction'])
            ax.set_ylabel('Entropy', fontsize=14)
            ax.set_title('Prediction Entropy\n(Lower = More Confident)', fontsize=14)
            ax.tick_params(axis='both', labelsize=14)
            ax.grid(True, alpha=0.3)
        
        # Plot 2: Correct token probability
        ax = axes[0, 1]
        if high_ind and low_ind:
            high_prob = [p['correct_prob'] for p in high_ind]
            low_prob = [p['correct_prob'] for p in low_ind]
            
            ax.boxplot([high_prob, low_prob], labels=['High\nInduction', 'Low\nInduction'])
            ax.set_ylabel('Probability', fontsize=14)
            ax.set_title('Probability of Correct Token', fontsize=14)
            ax.tick_params(axis='both', labelsize=14)
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Rank of correct token
        ax = axes[0, 2]
        if high_ind and low_ind:
            high_rank = [p['rank'] for p in high_ind]
            low_rank = [p['rank'] for p in low_ind]
            
            ax.hist([high_rank, low_rank], bins=20, label=['High Induction', 'Low Induction'], alpha=0.7)
            ax.set_xlabel('Rank of Correct Token', fontsize=14)
            ax.set_ylabel('Frequency', fontsize=14)
            ax.set_title('Distribution of Correct Token Ranks', fontsize=14)
            ax.tick_params(axis='both', labelsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Top-5 accuracy
        ax = axes[1, 0]
        if high_ind and low_ind:
            high_top5 = sum(p['in_top_5'] for p in high_ind) / len(high_ind) * 100 if high_ind else 0
            low_top5 = sum(p['in_top_5'] for p in low_ind) / len(low_ind) * 100 if low_ind else 0
            
            ax.bar(['High\nInduction', 'Low\nInduction'], [high_top5, low_top5], 
                   color=['coral', 'skyblue'], alpha=0.7)
            ax.set_ylabel('Accuracy (%)', fontsize=14)
            ax.set_title('Top-5 Accuracy', fontsize=14)
            ax.tick_params(axis='both', labelsize=14)
            ax.set_ylim(0, 100)
            for i, v in enumerate([high_top5, low_top5]):
                ax.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 5: Position-wise entropy
        ax = axes[1, 1]
        all_pos = analysis_results['all_positions']
        if all_pos:
            positions = [p['position'] for p in all_pos]
            entropies = [p['entropy'] for p in all_pos]
            
            ax.plot(positions, entropies, 'o-', color='purple', alpha=0.7, markersize=6)
            ax.set_xlabel('Position', fontsize=14)
            ax.set_ylabel('Entropy', fontsize=14)
            ax.set_title('Entropy Across Sequence', fontsize=14)
            ax.tick_params(axis='both', labelsize=14)
            ax.grid(True, alpha=0.3)
        
        # Plot 6: Summary statistics table
        ax = axes[1, 2]
        ax.axis('off')
        
        if high_ind and low_ind:
            summary_data = [
                ['Metric', 'High Induction', 'Low Induction'],
                ['', '', ''],
                ['Mean Entropy', f'{np.mean([p["entropy"] for p in high_ind]):.3f}', 
                 f'{np.mean([p["entropy"] for p in low_ind]):.3f}'],
                ['Mean Correct Prob', f'{np.mean([p["correct_prob"] for p in high_ind]):.3f}',
                 f'{np.mean([p["correct_prob"] for p in low_ind]):.3f}'],
                ['Median Rank', f'{np.median([p["rank"] for p in high_ind]):.0f}',
                 f'{np.median([p["rank"] for p in low_ind]):.0f}'],
                ['Top-5 Accuracy', f'{high_top5:.1f}%', f'{low_top5:.1f}%'],
                ['', '', ''],
                ['Sample Count', f'{len(high_ind)}', f'{len(low_ind)}'],
            ]
            
            table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                           bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Style header row
            for i in range(3):
                table[(0, i)].set_facecolor('#4472C4')
                table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved prediction gap analysis to {save_path}")
        
        return fig


def investigate_mamba_induction_mystery(model, tokenizer, timestamp, plots_dir='plots'):
    """
    Main investigation function to understand why Mamba detects induction but doesn't predict correctly.
    """
    import os
    
    analyzer = InductionPredictionAnalyzer(model, tokenizer)
    
    # Test sequences with clear induction patterns
    test_sequences = [
        "1 2 3 4 1 2 3 4",  # Your best sequence
        "A B C A B C",
        "cat dog bird cat dog bird",
    ]
    
    results = {}
    
    for idx, sequence in enumerate(test_sequences):
        print(f"\n{'='*60}")
        print(f"Investigating: {sequence}")
        print('='*60)
        
        # First, identify high induction positions
        # Based on your results, positions 5, 6, 7 for "1 2 3 4 1 2 3 4"
        tokens = tokenizer(sequence, return_tensors="pt")['input_ids'][0]
        
        # Find repeated tokens
        high_induction_pos = []
        for i in range(1, len(tokens)):
            current_token = tokens[i]
            # Check if this token appeared before
            if current_token in tokens[:i]:
                # This is a repeated token, likely high induction
                high_induction_pos.append(i - 1)  # The position BEFORE the repeat
        
        print(f"High induction positions: {high_induction_pos}")
        
        # Analyze logit distribution
        print("\n--- Analyzing Logit Distribution ---")
        logit_analysis = analyzer.analyze_logit_distribution(sequence, high_induction_pos)
        
        # Visualize
        save_path = os.path.join(plots_dir, f'prediction_gap_seq{idx}_{timestamp}.png')
        analyzer.visualize_prediction_gap(logit_analysis, save_path=save_path)
        
        # Compare first vs second occurrence
        print("\n--- Comparing First vs Second Occurrence ---")
        comparisons = []
        if len(high_induction_pos) > 0:
            # For "1 2 3 4 1 2 3 4", compare position 1 (first '2') vs position 5 (second '2')
            for pos in high_induction_pos[:2]:  # Check first 2 induction positions
                # Find the first occurrence of this token
                current_token = tokens[pos + 1]  # Token at induction position
                first_positions = (tokens == current_token).nonzero(as_tuple=True)[0]
                
                if len(first_positions) > 1:  # Make sure it's actually a repeat
                    first_pos = first_positions[0].item()
                    
                    if first_pos != pos + 1:  # Make sure it's not the same position
                        comparison = analyzer.compare_induction_contexts(
                            sequence, first_pos - 1, pos
                        )
                        
                        if comparison:
                            comparisons.append(comparison)
                            print(f"\nToken '{comparison['expected_token']}' comparison:")
                            print(f"  First occurrence (pos {first_pos - 1}):")
                            print(f"    Top predictions: {comparison['first_occurrence']['top_5_tokens']}")
                            print(f"    Top probs: {comparison['first_occurrence']['top_5_probs']}")
                            
                            print(f"  Second occurrence (pos {pos}) - INDUCTION POSITION:")
                            print(f"    Top predictions: {comparison['second_occurrence']['top_5_tokens']}")
                            print(f"    Top probs: {comparison['second_occurrence']['top_5_probs']}")
                            
                            print(f"  Expected token: '{comparison['expected_token']}'")
                            print(f"  Expected token rank at induction: {comparison['expected_rank_at_second']}")
                            print(f"  Expected token prob at induction: {comparison['expected_prob_at_second']:.4f}")
                            print(f"  Hidden state similarity: {comparison['hidden_similarity']:.4f}")
                            print(f"  Logit correlation: {comparison['logit_correlation']:.4f}")
                            
                            # KEY INSIGHT: If hidden states are similar but logits are different,
                            # the problem is in the output head (lm_head)
                            if comparison['hidden_similarity'] > 0.8 and abs(comparison['logit_correlation']) < 0.5:
                                print("\n  ⚠️  KEY FINDING: High hidden similarity but low logit correlation!")
                                print("      → The LM head is NOT preserving induction information!")
        
        results[f'seq_{idx}'] = {
            'sequence': sequence,
            'logit_analysis': logit_analysis,
            'high_induction_positions': high_induction_pos,
            'comparisons': comparisons,
        }
    
    return results

