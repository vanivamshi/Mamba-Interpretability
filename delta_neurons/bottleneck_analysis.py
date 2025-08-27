# bottleneck_analysis.py

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
import matplotlib.pyplot as plt
from typing import List, Dict
from scipy.stats import entropy
from sklearn.decomposition import PCA
from utils import get_model_layers
from delta_extraction import evaluate_perplexity, register_perturbation_hook


def evaluate_perturbation_effect_single_batch(
    model, tokenizer,
    texts,
    neuron_indices,
    layer_idx=0,
    mode="zero",
    std=1.0
):
    """
    Evaluate perplexity before and after perturbation on a single text set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Ensure model is on the correct device
    model = model.to(device)

    baseline_ppl = evaluate_perplexity(model, tokenizer, texts, device)

    if neuron_indices:
        layers = get_model_layers(model)
        target_layer = layers[layer_idx]
        hook = register_perturbation_hook(
            target_layer, neuron_indices, mode=mode, std=std
        )
        perturbed_ppl = evaluate_perplexity(model, tokenizer, texts, device)
        hook.remove()
    else:
        perturbed_ppl = baseline_ppl

    return baseline_ppl, perturbed_ppl


class BottleneckAnalyzer:
    """Analyze whether neurons are bottlenecks or redundant."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # Ensure model is on the correct device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def comprehensive_perturbation_analysis(self, texts: List[str], neuron_indices: List[int],
                                            layer_idx: int = 0) -> Dict:
        """Run comprehensive perturbation analysis with different intensities."""

        results = {
            'zero_perturbation': {},
            'noise_perturbations': {},
            'mean_perturbation': {},
            'partial_perturbations': {},
            'individual_effects': {}
        }

        # Baseline
        baseline_ppl, _ = evaluate_perturbation_effect_single_batch(
            self.model, self.tokenizer, texts, [], layer_idx, "zero"
        )
        results['baseline_ppl'] = baseline_ppl

        # 1. Zero perturbation (complete ablation)
        _, zero_ppl = evaluate_perturbation_effect_single_batch(
            self.model, self.tokenizer, texts, neuron_indices, layer_idx, "zero"
        )
        results['zero_perturbation'] = {
            'perplexity': zero_ppl,
            'effect_size': (zero_ppl - baseline_ppl) / baseline_ppl if baseline_ppl > 0 else 0.0
        }

        # 2. Noise perturbations with different intensities
        noise_levels = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        for noise_std in noise_levels:
            _, noise_ppl = evaluate_perturbation_effect_single_batch(
                self.model, self.tokenizer, texts, neuron_indices, layer_idx, "noise", noise_std
            )
            results['noise_perturbations'][noise_std] = {
                'perplexity': noise_ppl,
                'effect_size': (noise_ppl - baseline_ppl) / baseline_ppl if baseline_ppl > 0 else 0.0
            }

        # 3. Mean perturbation
        _, mean_ppl = evaluate_perturbation_effect_single_batch(
            self.model, self.tokenizer, texts, neuron_indices, layer_idx, "mean"
        )
        results['mean_perturbation'] = {
            'perplexity': mean_ppl,
            'effect_size': (mean_ppl - baseline_ppl) / baseline_ppl if baseline_ppl > 0 else 0.0
        }

        # 4. Partial perturbations (subset of neurons)
        if len(neuron_indices) > 1:
            for subset_size in [1, len(neuron_indices)//2, len(neuron_indices)-1]:
                if subset_size < len(neuron_indices):
                    subset = neuron_indices[:subset_size]
                    _, partial_ppl = evaluate_perturbation_effect_single_batch(
                        self.model, self.tokenizer, texts, subset, layer_idx, "zero"
                    )
                    results['partial_perturbations'][subset_size] = {
                        'neurons': subset,
                        'perplexity': partial_ppl,
                        'effect_size': (partial_ppl - baseline_ppl) / baseline_ppl if baseline_ppl > 0 else 0.0
                    }

        # 5. Individual neuron effects
        for neuron in neuron_indices:
            _, individual_ppl = evaluate_perturbation_effect_single_batch(
                self.model, self.tokenizer, texts, [neuron], layer_idx, "zero"
            )
            results['individual_effects'][neuron] = {
                'perplexity': individual_ppl,
                'effect_size': (individual_ppl - baseline_ppl) / baseline_ppl if baseline_ppl > 0 else 0.0
            }

        return results

    def classify_neuron_type(self, analysis_results: Dict) -> Dict:
        """Classify neurons as bottleneck, redundant, or other based on perturbation patterns."""

        zero_effect = analysis_results['zero_perturbation']['effect_size']
        noise_effects = [data['effect_size'] for data in analysis_results['noise_perturbations'].values()]
        mean_effect = analysis_results['mean_perturbation']['effect_size']

        noise_sensitivity = np.mean(noise_effects) if noise_effects else 0.0
        zero_to_noise_ratio = zero_effect / (noise_sensitivity + 1e-8) if noise_sensitivity > 0 else float('inf')

        classification = {
            'type': 'unknown',
            'confidence': 0.0,
            'metrics': {
                'zero_effect': zero_effect,
                'noise_sensitivity': noise_sensitivity,
                'mean_effect': mean_effect,
                'zero_to_noise_ratio': zero_to_noise_ratio
            }
        }

        if zero_effect > 0.2:
            if noise_sensitivity < 0.1:
                classification['type'] = 'bottleneck'
                classification['confidence'] = min(1.0, zero_effect * 2)
            elif noise_sensitivity > 0.15:
                classification['type'] = 'critical_sensitive'
                classification['confidence'] = min(1.0, (zero_effect + noise_sensitivity) / 2)
            else:
                classification['type'] = 'important'
                classification['confidence'] = zero_effect

        elif zero_effect < 0.05:
            if noise_sensitivity < 0.05:
                classification['type'] = 'redundant'
                classification['confidence'] = 1.0 - zero_effect
            else:
                classification['type'] = 'noise_sensitive'
                classification['confidence'] = noise_sensitivity

        else:
            classification['type'] = 'moderate'
            classification['confidence'] = 0.5

        return classification

    def analyze_information_flow(self, texts: List[str], neuron_indices: List[int],
                                 layer_idx: int = 0) -> Dict:
        """Analyze how information flows through these neurons."""

        activations_normal = self.extract_layer_activations(texts, layer_idx)
        activations_perturbed = self.extract_layer_activations(texts, layer_idx, perturb_neurons=neuron_indices)

        info_metrics = {}

        for i, (normal, perturbed) in enumerate(zip(activations_normal, activations_perturbed)):
            normal_flat = normal.flatten()
            perturbed_flat = perturbed.flatten()

            try:
                bins = np.linspace(min(normal_flat.min(), perturbed_flat.min()),
                                   max(normal_flat.max(), perturbed_flat.max()), 50)
                normal_hist, _ = np.histogram(normal_flat, bins=bins, density=True)
                perturbed_hist, _ = np.histogram(perturbed_flat, bins=bins, density=True)

                normal_hist += 1e-10
                perturbed_hist += 1e-10

                kl_div = entropy(normal_hist, perturbed_hist)
                info_metrics[f'kl_divergence_text_{i}'] = kl_div
            except Exception as e:
                print(f"Error calculating KL divergence: {e}")
                info_metrics[f'kl_divergence_text_{i}'] = 0.0

        try:
            all_normal = np.concatenate([act.reshape(-1, act.shape[-1]) for act in activations_normal])
            all_perturbed = np.concatenate([act.reshape(-1, act.shape[-1]) for act in activations_perturbed])

            pca = PCA(n_components=min(10, all_normal.shape[1]))
            normal_pca = pca.fit_transform(all_normal)
            perturbed_pca = pca.transform(all_perturbed)

            pca_distance = np.mean(np.linalg.norm(normal_pca - perturbed_pca, axis=1))
            info_metrics['pca_distance'] = pca_distance
            info_metrics['explained_variance'] = pca.explained_variance_ratio_.tolist()

        except Exception as e:
            print(f"Error in PCA analysis: {e}")
            info_metrics['pca_distance'] = 0.0

        return info_metrics

    def extract_layer_activations(self, texts: List[str], layer_idx: int,
                                  perturb_neurons: List[int] = None) -> List[np.ndarray]:
        """Extract activations from a specific layer."""

        layers = get_model_layers(self.model)
        layer = layers[layer_idx]
        activations = []

        def activation_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            if perturb_neurons:
                for neuron_idx in perturb_neurons:
                    if neuron_idx < hidden_states.shape[-1]:
                        hidden_states[..., neuron_idx] = 0

            activations.append(hidden_states.detach().cpu().numpy())

        handle = layer.register_forward_hook(activation_hook)

        try:
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                # Move inputs to the same device as model
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    _ = self.model(**inputs)
        finally:
            handle.remove()

        return activations

    def visualize_analysis(self, analysis_results: Dict, classification: Dict):
        """Visualize the analysis results."""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        ax1 = axes[0, 0]
        effects = [
            analysis_results['zero_perturbation']['effect_size'],
            analysis_results['mean_perturbation']['effect_size']
        ]
        noise_effects = [data['effect_size'] for data in analysis_results['noise_perturbations'].values()]
        effects.extend(noise_effects)

        labels = ['Zero', 'Mean'] + [f'Noise {std}' for std in analysis_results['noise_perturbations'].keys()]

        bars = ax1.bar(range(len(effects)), effects,
                       color=['red', 'orange'] + ['blue'] * len(noise_effects))
        ax1.set_xlabel('Perturbation Type')
        ax1.set_ylabel('Effect Size (Relative PPL Change)')
        ax1.set_title('Perturbation Effects')
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45)
        ax1.set_yscale('log')

        ax2 = axes[0, 1]
        if analysis_results['individual_effects']:
            neurons = list(analysis_results['individual_effects'].keys())
            individual_effects = [analysis_results['individual_effects'][n]['effect_size'] for n in neurons]

            ax2.bar(range(len(neurons)), individual_effects)
            ax2.set_xlabel('Neuron Index')
            ax2.set_ylabel('Individual Effect Size')
            ax2.set_title('Individual Neuron Contributions')
            ax2.set_xticks(range(len(neurons)))
            ax2.set_xticklabels([str(n) for n in neurons])

        ax3 = axes[1, 0]
        noise_stds = list(analysis_results['noise_perturbations'].keys())
        noise_effects = [analysis_results['noise_perturbations'][std]['effect_size']
                         for std in noise_stds]

        ax3.plot(noise_stds, noise_effects, 'o-', linewidth=2, markersize=8)
        ax3.set_xlabel('Noise Standard Deviation')
        ax3.set_ylabel('Effect Size')
        ax3.set_title('Noise Sensitivity Curve')
        ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        ax4.text(0.1, 0.8, f"Classification: {classification['type']}",
                 fontsize=14, fontweight='bold', transform=ax4.transAxes)
        ax4.text(0.1, 0.7, f"Confidence: {classification['confidence']:.3f}",
                 fontsize=12, transform=ax4.transAxes)

        metrics_text = "Key Metrics:\n"
        for key, value in classification['metrics'].items():
            metrics_text += f"{key}: {value:.4f}\n"

        ax4.text(0.1, 0.1, metrics_text, fontsize=10, transform=ax4.transAxes,
                 verticalalignment='bottom')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Classification Results')

        plt.tight_layout()
        
        # Save the plot to images folder
        import os
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/bottleneck_analysis.png", dpi=300, bbox_inches='tight')
        print("✅ Saved bottleneck analysis plot to images/bottleneck_analysis.png")
        
        return fig


def analyze_your_neurons(neuron_indices=None):
    """Analyze the specific neurons you found."""

    print(f"Using top {len(neuron_indices)} neurons: {neuron_indices}")

    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = "state-spaces/mamba-130m-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Ensure model is on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Machine learning models require large datasets.",
        "Natural language processing involves understanding text.",
        "Deep learning has revolutionized computer vision."
    ]

    analyzer = BottleneckAnalyzer(model, tokenizer)

    print("Running comprehensive perturbation analysis...")
    analysis_results = analyzer.comprehensive_perturbation_analysis(texts, neuron_indices)

    print("Classifying neuron type...")
    classification = analyzer.classify_neuron_type(analysis_results)

    print(f"\n=== ANALYSIS RESULTS ===")
    print(f"Neuron Classification: {classification['type'].upper()}")
    print(f"Confidence: {classification['confidence']:.3f}")
    print(f"\nKey Evidence:")
    print(f"- Zero perturbation effect: {classification['metrics']['zero_effect']:.3f}")
    print(f"- Noise sensitivity: {classification['metrics']['noise_sensitivity']:.3f}")
    print(f"- Zero-to-noise ratio: {classification['metrics']['zero_to_noise_ratio']:.1f}")

    print("\nAnalyzing information flow...")
    info_metrics = analyzer.analyze_information_flow(texts, neuron_indices)
    print("Information flow metrics:", info_metrics)

    fig = analyzer.visualize_analysis(analysis_results, classification)
    # plt.show()  # Not needed with non-interactive backend
    
    return analysis_results, classification


if __name__ == "__main__":
    analyze_your_neurons()
