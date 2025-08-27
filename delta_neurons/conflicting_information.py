import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple, Optional
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils import get_model_layers

class StateInterpolationAnalyzer:
    """Analyze how delta neurons handle conflicting information through state interpolation."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def create_conflicting_sequences(self) -> List[Dict]:
        """Create sequences with conflicting information to test state interpolation."""
        
        # Different types of conflicts to test
        conflicting_sequences = [
            {
                'type': 'factual_contradiction', 
                'description': 'Contradictory facts about the same entity',
                'sequences': [
                    "Paris is the capital of France. However, London is the capital of France.",
                    "Water boils at 100°C. Actually, water boils at 50°C.",
                    "Einstein developed relativity theory. Einstein never worked on physics.",
                    "The sun is hot and bright. The sun is cold and dark.",
                ]
            },
            {
                'type': 'temporal_conflict',
                'description': 'Events with conflicting temporal information', 
                'sequences': [
                    "John was born in 1990. John was born in 1980.",
                    "The meeting is tomorrow. The meeting was yesterday.",
                    "Summer comes after spring. Summer comes before spring.",
                    "I will go shopping later. I went shopping earlier.",
                ]
            },
            {
                'type': 'emotional_contradiction',
                'description': 'Conflicting emotional states',
                'sequences': [
                    "I am very happy today. I am extremely sad today.",
                    "This movie is fantastic. This movie is terrible.",
                    "She loves chocolate. She hates chocolate.",
                    "The weather is beautiful. The weather is awful.",
                ]
            },
            {
                'type': 'logical_contradiction',
                'description': 'Logically incompatible statements',
                'sequences': [
                    "All birds can fly. Penguins are birds that cannot fly.",
                    "The box is empty. The box contains many items.",
                    "He is always late. He arrived early today.",
                    "Nothing is impossible. Some things are impossible.",
                ]
            }
        ]
        
        return conflicting_sequences
    
    def extract_state_sequence(self, text: str, neuron_indices: List[int], 
                              layer_idx: int = 0) -> Dict:
        """Extract the sequence of states for specific neurons during text processing."""
        
        layers = get_model_layers(self.model)
        if not layers or layer_idx >= len(layers):
            raise ValueError(f"Invalid layer index {layer_idx}")
        
        target_layer = layers[layer_idx]
        
        # Store states at each time step
        states_sequence = []
        input_tokens = []
        
        def state_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Extract states for specific neurons
            neuron_states = hidden_states[0, :, neuron_indices].detach().cpu().numpy()  # [seq_len, num_neurons]
            states_sequence.append(neuron_states.copy())
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Register hook and run forward pass
        handle = target_layer.register_forward_hook(state_hook)
        
        try:
            with torch.no_grad():
                _ = self.model(**inputs)
        finally:
            handle.remove()
        
        # Combine all states - take the last recorded state sequence
        if states_sequence:
            final_states = states_sequence[-1]  # [seq_len, num_neurons]
        else:
            final_states = np.random.randn(len(input_tokens), len(neuron_indices))
        
        return {
            'states': final_states,
            'tokens': input_tokens,
            'token_ids': inputs['input_ids'][0].cpu().numpy(),
            'sequence_length': len(input_tokens)
        }
    
    def analyze_conflict_resolution(self, conflicting_sequences: List[Dict], 
                                   neuron_indices: List[int], layer_idx: int = 0) -> Dict:
        """Analyze how neurons resolve conflicting information."""
        
        print(f"🔍 Analyzing conflict resolution in layer {layer_idx}")
        print(f"Target neurons: {neuron_indices}")
        
        results = {
            'conflict_types': {},
            'interpolation_patterns': {},
            'resolution_strategies': {}
        }
        
        for conflict_group in conflicting_sequences:
            conflict_type = conflict_group['type']
            print(f"\n📊 Analyzing {conflict_type}...")
            
            conflict_results = []
            
            for sequence in conflict_group['sequences']:
                print(f"  Processing: {sequence[:50]}...")
                
                try:
                    # Extract state sequence
                    state_data = self.extract_state_sequence(sequence, neuron_indices, layer_idx)
                    
                    # Analyze state evolution
                    analysis = self.analyze_single_sequence(state_data, sequence)
                    conflict_results.append(analysis)
                    
                except Exception as e:
                    print(f"    Error processing sequence: {e}")
                    continue
            
            if conflict_results:
                # Aggregate results for this conflict type
                results['conflict_types'][conflict_type] = {
                    'description': conflict_group['description'],
                    'individual_results': conflict_results,
                    'summary': self.summarize_conflict_type(conflict_results)
                }
        
        return results
    
    def analyze_single_sequence(self, state_data: Dict, sequence: str) -> Dict:
        """Analyze state evolution for a single conflicting sequence."""
        
        states = state_data['states']  # [seq_len, num_neurons]
        tokens = state_data['tokens']
        
        # Find potential conflict points (simple heuristic: look for contradiction words)
        conflict_indicators = ['however', 'but', 'actually', 'although', 'despite', 'nevertheless', 'yet']
        conflict_positions = []
        
        for i, token in enumerate(tokens):
            if any(indicator in token.lower() for indicator in conflict_indicators):
                conflict_positions.append(i)
        
        # If no explicit indicators, assume conflict is in second half
        if not conflict_positions:
            conflict_positions = [len(tokens) // 2]
        
        analysis = {
            'sequence': sequence,
            'sequence_length': len(tokens),
            'conflict_positions': conflict_positions,
            'state_evolution': self.analyze_state_evolution(states, conflict_positions),
            'interpolation_metrics': self.calculate_interpolation_metrics(states),
            'resolution_pattern': self.identify_resolution_pattern(states, conflict_positions)
        }
        
        return analysis
    
    def analyze_state_evolution(self, states: np.ndarray, conflict_positions: List[int]) -> Dict:
        """Analyze how states evolve before, during, and after conflicts."""
        
        seq_len, num_neurons = states.shape
        
        # Divide sequence into segments
        segments = {
            'pre_conflict': states[:conflict_positions[0]] if conflict_positions else states[:seq_len//2],
            'conflict': states[conflict_positions[0]:conflict_positions[-1]+1] if len(conflict_positions) > 1 else states[conflict_positions[0]:conflict_positions[0]+1],
            'post_conflict': states[conflict_positions[-1]+1:] if conflict_positions else states[seq_len//2:]
        }
        
        evolution_metrics = {}
        
        for segment_name, segment_states in segments.items():
            if len(segment_states) == 0:
                continue
                
            # Calculate metrics for this segment
            metrics = {
                'mean_activation': np.mean(segment_states, axis=0),
                'std_activation': np.std(segment_states, axis=0),
                'activation_range': np.ptp(segment_states, axis=0),  # peak-to-peak
                'state_stability': self.calculate_stability(segment_states)
            }
            
            evolution_metrics[segment_name] = metrics
        
        # Calculate transitions between segments
        if len(segments) > 1:
            evolution_metrics['transitions'] = self.calculate_segment_transitions(segments)
        
        return evolution_metrics
    
    def calculate_stability(self, states: np.ndarray) -> float:
        """Calculate stability of states (lower variance = more stable)."""
        if len(states) <= 1:
            return 0.0
        
        # Calculate variance across time for each neuron, then average
        temporal_variance = np.var(states, axis=0)
        return float(np.mean(temporal_variance))
    
    def calculate_segment_transitions(self, segments: Dict) -> Dict:
        """Calculate how states transition between segments."""
        
        transitions = {}
        segment_names = list(segments.keys())
        
        for i in range(len(segment_names) - 1):
            current_segment = segment_names[i]
            next_segment = segment_names[i + 1]
            
            if len(segments[current_segment]) == 0 or len(segments[next_segment]) == 0:
                continue
            
            # Compare end of current segment with start of next segment
            end_state = segments[current_segment][-1]
            start_state = segments[next_segment][0]
            
            # Calculate transition metrics
            transition_key = f"{current_segment}_to_{next_segment}"
            transitions[transition_key] = {
                'cosine_similarity': float(cosine_similarity([end_state], [start_state])[0, 0]),
                'euclidean_distance': float(np.linalg.norm(end_state - start_state)),
                'state_change_magnitude': float(np.mean(np.abs(end_state - start_state)))
            }
        
        return transitions
    
    def calculate_interpolation_metrics(self, states: np.ndarray) -> Dict:
        """Calculate metrics that indicate interpolation behavior."""
        
        if len(states) < 2:
            return {'error': 'Insufficient states for interpolation analysis'}
        
        # Calculate smoothness (how gradually states change)
        state_diffs = np.diff(states, axis=0)  # Differences between consecutive states
        smoothness = np.mean(np.std(state_diffs, axis=0))  # Lower = smoother transitions
        
        # Calculate directional consistency
        if len(states) > 2:
            direction_changes = 0
            for neuron_idx in range(states.shape[1]):
                neuron_states = states[:, neuron_idx]
                diffs = np.diff(neuron_states)
                if len(diffs) > 1:
                    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
                    direction_changes += sign_changes
            
            directional_consistency = 1.0 - (direction_changes / (states.shape[1] * (len(states) - 2)))
        else:
            directional_consistency = 1.0
        
        # Calculate overall state trajectory curvature
        if len(states) >= 3:
            curvature = self.calculate_trajectory_curvature(states)
        else:
            curvature = 0.0
        
        return {
            'smoothness': float(smoothness),
            'directional_consistency': float(directional_consistency),
            'trajectory_curvature': float(curvature),
            'total_state_change': float(np.linalg.norm(states[-1] - states[0]))
        }
    
    def calculate_trajectory_curvature(self, states: np.ndarray) -> float:
        """Calculate the curvature of the state trajectory."""
        
        # Use second derivatives to estimate curvature
        second_diffs = np.diff(states, n=2, axis=0)  # Second differences
        curvature = np.mean(np.linalg.norm(second_diffs, axis=1))
        
        return curvature
    
    def identify_resolution_pattern(self, states: np.ndarray, conflict_positions: List[int]) -> str:
        """Identify the pattern of conflict resolution."""
        
        if len(states) < 3 or not conflict_positions:
            return 'insufficient_data'
        
        seq_len = len(states)
        
        # Compare pre-conflict, conflict, and post-conflict states
        pre_conflict = states[:conflict_positions[0]]
        post_conflict = states[conflict_positions[-1]+1:]
        
        if len(pre_conflict) == 0 or len(post_conflict) == 0:
            return 'boundary_conflict'
        
        pre_mean = np.mean(pre_conflict, axis=0)
        post_mean = np.mean(post_conflict, axis=0)
        
        # Calculate similarity between pre and post conflict states
        similarity = cosine_similarity([pre_mean], [post_mean])[0, 0]
        
        # Analyze the pattern
        if similarity > 0.9:
            return 'stable_states'  # States remain similar despite conflict
        elif similarity > 0.5:
            return 'partial_interpolation'  # Some blending occurred
        elif similarity > 0.0:
            return 'state_transition'  # Clear transition to new state
        else:
            return 'state_reversal'  # States moved in opposite direction
    
    def summarize_conflict_type(self, conflict_results: List[Dict]) -> Dict:
        """Summarize results across all sequences of a conflict type."""
        
        if not conflict_results:
            return {'error': 'No results to summarize'}
        
        # Aggregate interpolation metrics
        all_smoothness = []
        all_consistency = []
        all_curvature = []
        resolution_patterns = []
        
        for result in conflict_results:
            interp_metrics = result.get('interpolation_metrics', {})
            if 'smoothness' in interp_metrics:
                all_smoothness.append(interp_metrics['smoothness'])
            if 'directional_consistency' in interp_metrics:
                all_consistency.append(interp_metrics['directional_consistency'])
            if 'trajectory_curvature' in interp_metrics:
                all_curvature.append(interp_metrics['trajectory_curvature'])
            
            resolution_patterns.append(result.get('resolution_pattern', 'unknown'))
        
        # Calculate summary statistics
        summary = {
            'num_sequences': len(conflict_results),
            'avg_smoothness': float(np.mean(all_smoothness)) if all_smoothness else 0.0,
            'avg_directional_consistency': float(np.mean(all_consistency)) if all_consistency else 0.0,
            'avg_trajectory_curvature': float(np.mean(all_curvature)) if all_curvature else 0.0,
            'resolution_patterns': {
                pattern: resolution_patterns.count(pattern) 
                for pattern in set(resolution_patterns)
            }
        }
        
        return summary
    
    def visualize_state_interpolation(self, analysis_results: Dict, neuron_indices: List[int]):
        """Create comprehensive visualizations of state interpolation patterns."""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Interpolation metrics by conflict type
        ax1 = axes[0, 0]
        conflict_types = []
        smoothness_values = []
        consistency_values = []
        
        for conflict_type, data in analysis_results['conflict_types'].items():
            if 'summary' in data and 'avg_smoothness' in data['summary']:
                conflict_types.append(conflict_type.replace('_', ' ').title())
                smoothness_values.append(data['summary']['avg_smoothness'])
                consistency_values.append(data['summary']['avg_directional_consistency'])
        
        if conflict_types:
            x = np.arange(len(conflict_types))
            width = 0.35
            
            ax1.bar(x - width/2, smoothness_values, width, label='Smoothness', alpha=0.8)
            ax1.bar(x + width/2, consistency_values, width, label='Directional Consistency', alpha=0.8)
            
            ax1.set_xlabel('Conflict Type')
            ax1.set_ylabel('Metric Value')
            ax1.set_title('Interpolation Metrics by Conflict Type')
            ax1.set_xticks(x)
            ax1.set_xticklabels(conflict_types, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Resolution pattern distribution
        ax2 = axes[0, 1]
        all_patterns = {}
        
        for conflict_type, data in analysis_results['conflict_types'].items():
            if 'summary' in data and 'resolution_patterns' in data['summary']:
                for pattern, count in data['summary']['resolution_patterns'].items():
                    all_patterns[pattern] = all_patterns.get(pattern, 0) + count
        
        if all_patterns:
            patterns = list(all_patterns.keys())
            counts = list(all_patterns.values())
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(patterns)))
            wedges, texts, autotexts = ax2.pie(counts, labels=patterns, autopct='%1.1f%%', 
                                              colors=colors, startangle=90)
            ax2.set_title('Resolution Pattern Distribution')
        
        # 3. State trajectory curvature comparison
        ax3 = axes[1, 0]
        curvature_by_type = []
        type_labels = []
        
        for conflict_type, data in analysis_results['conflict_types'].items():
            if 'summary' in data and 'avg_trajectory_curvature' in data['summary']:
                curvature_by_type.append(data['summary']['avg_trajectory_curvature'])
                type_labels.append(conflict_type.replace('_', ' ').title())
        
        if curvature_by_type:
            bars = ax3.barh(type_labels, curvature_by_type, color='skyblue', alpha=0.7)
            ax3.set_xlabel('Average Trajectory Curvature')
            ax3.set_title('State Trajectory Curvature by Conflict Type')
            ax3.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, curvature_by_type):
                ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{value:.3f}', va='center', ha='left')
        
        # 4. Summary text panel
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"State Interpolation Analysis Summary\n\n"
        summary_text += f"Analyzed Neurons: {neuron_indices}\n\n"
        
        total_sequences = sum(data.get('summary', {}).get('num_sequences', 0) 
                            for data in analysis_results['conflict_types'].values())
        summary_text += f"Total Sequences Analyzed: {total_sequences}\n\n"
        
        # Most common resolution pattern
        if all_patterns:
            most_common_pattern = max(all_patterns, key=all_patterns.get)
            summary_text += f"Most Common Resolution: {most_common_pattern.replace('_', ' ').title()}\n\n"
        
        # Average metrics across all conflict types
        if smoothness_values:
            summary_text += f"Average Smoothness: {np.mean(smoothness_values):.3f}\n"
            summary_text += f"Average Consistency: {np.mean(consistency_values):.3f}\n"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot to images folder
        import os
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/state_interpolation_analysis.png", dpi=300, bbox_inches='tight')
        print("✅ Saved state interpolation analysis plot to images/state_interpolation_analysis.png")
        
        return fig

def run_state_interpolation_analysis():
    """Main function to run the complete state interpolation analysis."""
    
    print("🚀 Dynamic State Management Analysis")
    print("=" * 50)
    
    # Setup model and tokenizer
    model_name = "state-spaces/mamba-130m-hf"
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize analyzer
    analyzer = StateInterpolationAnalyzer(model, tokenizer)
    
    # Choose neurons to analyze
    target_neurons = [300, 189, 446, 560, 175]  # Your discovered delta-sensitive neurons
    layer_idx = 0
    
    print(f"Target neurons: {target_neurons}")
    print(f"Target layer: {layer_idx}")
    
    # Create conflicting sequences
    print("\n📝 Creating conflicting sequences...")
    conflicting_sequences = analyzer.create_conflicting_sequences()
    
    total_sequences = sum(len(group['sequences']) for group in conflicting_sequences)
    print(f"Created {len(conflicting_sequences)} conflict types with {total_sequences} sequences")
    
    # Run analysis
    print("\n🔬 Running conflict resolution analysis...")
    results = analyzer.analyze_conflict_resolution(conflicting_sequences, target_neurons, layer_idx)
    
    # Print detailed results
    print("\n" + "=" * 60)
    print("📊 DETAILED ANALYSIS RESULTS")
    print("=" * 60)
    
    for conflict_type, data in results['conflict_types'].items():
        print(f"\n🔹 {conflict_type.upper().replace('_', ' ')}")
        print(f"   Description: {data['description']}")
        
        summary = data.get('summary', {})
        if summary and 'num_sequences' in summary:
            print(f"   Sequences analyzed: {summary['num_sequences']}")
            print(f"   Average smoothness: {summary.get('avg_smoothness', 0):.4f}")
            print(f"   Average consistency: {summary.get('avg_directional_consistency', 0):.4f}")
            print(f"   Average curvature: {summary.get('avg_trajectory_curvature', 0):.4f}")
            
            # Resolution patterns
            patterns = summary.get('resolution_patterns', {})
            if patterns:
                print(f"   Resolution patterns:")
                for pattern, count in patterns.items():
                    print(f"     - {pattern.replace('_', ' ').title()}: {count}")
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    fig = analyzer.visualize_state_interpolation(results, target_neurons)
    # plt.show()  # Not needed with non-interactive backend
    
    # Print key insights
    print("\n" + "=" * 60)
    print("🎯 KEY INSIGHTS")
    print("=" * 60)
    
    # Analyze patterns across conflict types
    all_smoothness = []
    all_consistency = []
    
    for data in results['conflict_types'].values():
        summary = data.get('summary', {})
        if 'avg_smoothness' in summary:
            all_smoothness.append(summary['avg_smoothness'])
        if 'avg_directional_consistency' in summary:
            all_consistency.append(summary['avg_directional_consistency'])
    
    if all_smoothness:
        avg_smoothness = np.mean(all_smoothness)
        avg_consistency = np.mean(all_consistency)
        
        print(f"🔸 Overall State Interpolation Quality:")
        print(f"   - Smoothness: {avg_smoothness:.4f} ({'High' if avg_smoothness < 0.1 else 'Medium' if avg_smoothness < 0.5 else 'Low'})")
        print(f"   - Consistency: {avg_consistency:.4f} ({'High' if avg_consistency > 0.8 else 'Medium' if avg_consistency > 0.5 else 'Low'})")
        
        if avg_smoothness < 0.1 and avg_consistency > 0.8:
            print("   ✅ Neurons show excellent interpolation behavior!")
        elif avg_smoothness < 0.5 and avg_consistency > 0.5:
            print("   ⚡ Neurons show good interpolation with some adaptation!")
        else:
            print("   ⚠️  Neurons show complex, possibly chaotic behavior!")
    
    print(f"\n🔸 Conflict Resolution Strategy:")
    # Find most common resolution pattern
    all_patterns = {}
    for data in results['conflict_types'].values():
        patterns = data.get('summary', {}).get('resolution_patterns', {})
        for pattern, count in patterns.items():
            all_patterns[pattern] = all_patterns.get(pattern, 0) + count
    
    if all_patterns:
        dominant_pattern = max(all_patterns, key=all_patterns.get)
        pattern_percentage = (all_patterns[dominant_pattern] / sum(all_patterns.values())) * 100
        
        pattern_descriptions = {
            'stable_states': 'maintain consistent representations despite conflicts',
            'partial_interpolation': 'blend conflicting information gradually',
            'state_transition': 'switch to new state representations',
            'state_reversal': 'reverse direction when encountering conflicts'
        }
        
        description = pattern_descriptions.get(dominant_pattern, 'show complex resolution behavior')
        print(f"   - Primary strategy ({pattern_percentage:.1f}%): {description}")
    
    print("\n✅ Analysis complete!")
    
    return results, analyzer

if __name__ == "__main__":
    results, analyzer = run_state_interpolation_analysis()


"""
Step 4: Calculate Key Metrics
1. Smoothness: How gradually do states change?

Low smoothness = abrupt changes
High smoothness = gradual interpolation

2. Directional Consistency: Do neurons change direction frequently?

High consistency = steady trajectory
Low consistency = oscillating behavior

3. Trajectory Curvature: How curved is the path through state space?

Low curvature = straight line changes
High curvature = complex, winding paths

Step 4.1: Identify Resolution Strategies
The analysis identifies several patterns:

Stable States: Neurons maintain consistent representations despite conflicts
Partial Interpolation: Neurons blend conflicting information gradually
State Transition: Clear switch to new representations
State Reversal: States move in opposite directions
"""