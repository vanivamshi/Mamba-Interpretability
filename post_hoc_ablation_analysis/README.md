# Mamba Mechanistic Interpretability Framework

A comprehensive experimental framework for opening Mamba's black box through systematic mechanistic interpretability analysis.

## Overview

This framework implements a step-by-step experimental recipe for understanding the internal mechanisms of Mamba models, following rigorous scientific methodology. The approach combines multiple interpretability techniques to discover, validate, and understand the circuits that implement specific functions in Mamba models.

## Key Features

- **🔬 Systematic Analysis**: Step-by-step experimental methodology
- **🎯 Sparse Autoencoders (SAE)**: Discover interpretable latent features
- **🔍 Activation Patching**: Test necessity and sufficiency of circuits
- **⏰ Temporal Causality**: Analyze long-range dependencies with Jacobian maps
- **📊 Comprehensive Visualization**: Rich visualizations and reporting
- **🎲 Reproducible**: Deterministic seeding and detailed logging
- **📈 Scalable**: From small models (0.5M) to large models (1B+)

## Experimental Framework

The framework follows a 15-step methodology:

### Core Steps
1. **Setup**: Reproducible environment with deterministic seeding
2. **Activation Collection**: Gather activation data with baseline statistics
3. **SAE Discovery**: Find sparse, interpretable features
4. **Hypothesis Probes**: Use Lasso/ElasticNet for causal dimension discovery
5. **Circuit Selection**: Combine SAE units, probe dims, and clustering
6. **Activation Patching**: Test necessity and sufficiency
7. **Perturbation Robustness**: Measure circuit granularity
8. **Temporal Causality**: Jacobian and influence maps
9. **Dynamic Analysis**: Eigenmodes, timescales, controllability
10. **Reproducibility**: Cross-seed validation
11. **Cross-task Transfer**: Test reusable circuits
12. **Visualization**: Comprehensive reporting
13. **Controls**: Statistical rigor and baselines
14. **Scale-up**: Apply to larger models
15. **Deliverables**: Complete mechanistic claims

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd circuit_1

# Install dependencies
pip install -r requirements.txt

# Optional: Install development dependencies
pip install -e .
```

## Quick Start

### Basic Analysis

```bash
# Run complete analysis on Mamba-130M
python mamba_mechanistic_analysis.py --model state-spaces/mamba-130m-hf --layer 0 --samples 100

# Use synthetic toy data for controlled experiments
python mamba_mechanistic_analysis.py --model state-spaces/mamba-130m-hf --use_toy_data --samples 200

# Skip certain steps for faster iteration
python mamba_mechanistic_analysis.py --skip_steps 8 9 10  # Skip temporal analysis
```

### Advanced Usage

```python
from experimental_framework import ExperimentConfig, MambaMechanisticAnalyzer

# Create custom configuration
config = ExperimentConfig(
    model_name="state-spaces/mamba-130m-hf",
    layer_idx=0,
    num_samples=500,
    sae_latent_dim=0.3,
    sae_l1_weight=1e-3,
    seed=42
)

# Initialize analyzer
analyzer = MambaMechanisticAnalyzer(config)
analyzer.setup()

# Run specific analysis steps
activations = analyzer.collect_activations(texts)
sae_results = analyzer.discover_interpretable_features()
circuits = analyzer.select_candidate_circuits()
patching_results = analyzer.test_circuit_causality()
```

## Framework Components

### 1. Experimental Framework (`experimental_framework.py`)
- Deterministic setup and seeding
- Activation collection and instrumentation
- Experiment logging and result management
- Toy dataset generation for controlled experiments

### 2. Sparse Autoencoder (`sparse_autoencoder.py`)
- SAE implementation with sparsity penalties
- Feature correlation analysis
- Interpretable feature discovery
- Sparse probing encoders (Lasso/ElasticNet)

### 3. Activation Patching (`activation_patching.py`)
- Necessity testing (ablation)
- Sufficiency testing (patching)
- Control tests with random subspaces
- Statistical significance testing

### 4. Temporal Causality (`temporal_causality.py`)
- Jacobian computation for temporal dependencies
- Influence maps showing attention-like patterns
- Long-range dependency analysis
- Temporal decay analysis

### 5. Main Analysis Script (`mamba_mechanistic_analysis.py`)
- Orchestrates the complete pipeline
- Integrates all components
- Generates comprehensive reports
- Command-line interface

## Usage Examples

### Example 1: Basic Circuit Discovery

```python
from mamba_mechanistic_analysis import MambaMechanisticAnalyzer
from experimental_framework import ExperimentConfig

# Setup
config = ExperimentConfig(model_name="state-spaces/mamba-130m-hf")
analyzer = MambaMechanisticAnalyzer(config)
analyzer.setup()

# Collect activations
texts = ["The quick brown fox jumps over the lazy dog."] * 50
activations = analyzer.collect_activations(texts)

# Discover interpretable features
sae_results = analyzer.discover_interpretable_features()

# Select candidate circuits
circuits = analyzer.select_candidate_circuits()

# Test causality
patching_results = analyzer.test_circuit_causality()
```

### Example 2: Temporal Analysis

```python
from temporal_causality import run_temporal_causality_analysis

# Analyze temporal dependencies
temporal_results = run_temporal_causality_analysis(
    model=model,
    inputs=input_tokens,
    layer_idx=0,
    circuit_indices=[100, 200, 300],  # Circuit dimensions
    max_lag=10
)
```

### Example 3: SAE Analysis

```python
from sparse_autoencoder import run_sae_analysis

# Run SAE analysis
sae_results = run_sae_analysis(
    activations=activation_tensor,
    task_labels=task_labels,
    config={
        'latent_dim_ratio': 0.3,
        'sparsity_weight': 1e-3,
        'num_epochs': 100
    }
)
```

## Configuration Options

### ExperimentConfig Parameters

```python
@dataclass
class ExperimentConfig:
    # Model parameters
    model_name: str = "state-spaces/mamba-130m-hf"
    hidden_size: int = 768
    num_layers: int = 24
    
    # Analysis parameters
    layer_idx: int = 0
    top_k: int = 10
    num_samples: int = 50
    
    # SAE parameters
    sae_latent_dim: float = 0.3  # Fraction of hidden size
    sae_l1_weight: float = 1e-3
    sae_sparsity_target: float = 0.05
    
    # Reproducibility
    seed: int = 42
    device: str = "cuda"
    
    # Logging
    use_wandb: bool = False
    log_dir: str = "experiment_logs"
```

## Output Structure

The framework generates comprehensive outputs:

```
experiment_logs/
├── experiment_YYYYMMDD_HHMMSS/
│   ├── config.json                    # Experiment configuration
│   ├── experiment.log                 # Detailed logging
│   ├── activations.pt                 # Raw activation data
│   ├── baseline_stats.json           # Activation statistics
│   ├── sae_results_layer_X.json       # SAE analysis results
│   ├── probe_results_layer_X.json     # Probing results
│   ├── candidate_circuits_layer_X.json # Circuit candidates
│   ├── patching_results_layer_X.json  # Patching test results
│   ├── temporal_results_layer_X.json  # Temporal analysis
│   └── comprehensive_report.json      # Final report
```

## Visualization Features

The framework includes comprehensive visualization capabilities:

- **SAE Feature Analysis**: Activation patterns, sparsity, correlations
- **Activation Patching Results**: Effect sizes, significance testing
- **Temporal Influence Maps**: Attention-like patterns, decay analysis
- **Circuit Analysis**: Long-range vs short-range dependencies
- **Statistical Summaries**: Effect distributions, significance plots

## Best Practices

### 1. Start Small
- Begin with small models (0.5-2M parameters)
- Use synthetic toy tasks for controlled experiments
- Validate methodology before scaling up

### 2. Reproducibility
- Always use deterministic seeding
- Save all configurations and random seeds
- Document hyperparameters and model versions

### 3. Statistical Rigor
- Run control tests with random subspaces
- Use multiple random seeds for validation
- Report effect sizes and statistical significance

### 4. Interpretation
- Focus on circuits with strong correlations (>0.6)
- Validate with both necessity and sufficiency tests
- Check temporal consistency across timesteps

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or sequence length
   - Use CPU for smaller experiments
   - Process data in smaller chunks

2. **No Significant Circuits Found**
   - Increase sample size
   - Try different SAE hyperparameters
   - Check if model is properly trained

3. **Slow Jacobian Computation**
   - Reduce max_lag parameter
   - Use fewer timesteps
   - Consider approximate methods

### Performance Tips

- Use mixed precision training for SAE
- Cache activations to avoid recomputation
- Parallelize across multiple GPUs for large models
- Use gradient checkpointing for memory efficiency

## Contributing

We welcome contributions! Please see our contributing guidelines for:

- Code style and standards
- Testing requirements
- Documentation standards
- Pull request process

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{mamba_mechanistic_interpretability,
  title={Opening Mamba's Black Box: A Mechanistic Interpretability Framework},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on top of the transformers library
- Inspired by mechanistic interpretability research
- Thanks to the Mamba team for the original model

## Support

For questions and support:

- Open an issue on GitHub
- Check the documentation
- Join our discussion forum

---

**Happy analyzing! 🔬✨**
