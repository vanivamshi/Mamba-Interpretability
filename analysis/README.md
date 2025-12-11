# Multi-Layer Neuron Analysis

This directory contains scripts for comprehensive multi-layer analysis of all neuron types in the Mamba model.

## Main Scripts

### `main_all_layers.py`
**Purpose**: Main orchestration script that runs all neuron type analyses across all 24 layers of Mamba-130M.

**Usage**:
```bash
python analysis/main_all_layers.py
```

**What it does**:
- Loads the Mamba-130M model
- Runs analysis for 6 neuron types:
  - Universality Neurons
  - Delta-Sensitive Neurons
  - Projection-Dominant Neurons
  - Speciality Neurons
  - Dead Neurons
  - Causal Neurons
- Generates comprehensive visualizations:
  - 6 heatmaps (one per neuron type)
  - 144 individual layer plots (24 layers × 6 types)
  - 1 summary trend plot
- Saves all results to `plots/all_layers_TIMESTAMP/`

**Output Structure**:
```
plots/all_layers_TIMESTAMP/
├── universality_heatmap.png
├── universality_results.json
├── universality_layers/
│   ├── layer_0.png
│   ├── layer_1.png
│   └── ...
├── delta_variance_heatmap.png
├── delta_results.json
├── delta_layers/
├── projection_heatmap.png
├── projection_results.json
├── projection_layers/
├── speciality_heatmap.png
├── speciality_results.json
├── speciality_layers/
├── dead_heatmap.png
├── dead_results.json
├── dead_layers/
├── causal_heatmap.png
├── causal_results.json
├── causal_layers/
└── layer_trends.png
```

### `efficient_analysis.py`
**Purpose**: Contains all multi-layer analysis functions for each neuron type.

**Key Functions**:
- `analyze_universality_all_layers()` - Analyzes universality across tasks
- `find_delta_sensitive_neurons_all_layers()` - Finds delta-sensitive neurons
- `find_projection_dominant_neurons_all_layers()` - Finds projection-dominant neurons
- `find_specialty_neurons_all_layers()` - Finds class-specific neurons
- `find_dead_neurons_all_layers()` - Identifies rarely-active neurons
- `find_causal_neurons_all_layers()` - Measures causal impact of neurons

### `utils.py`
**Purpose**: Utility functions for model layer access and activation hooks.

**Key Functions**:
- `get_model_layers()` - Extracts layers from different model architectures
- `get_activation_hook_target()` - Determines correct module for activation hooks

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- NumPy
- Matplotlib
- Seaborn (optional, for better heatmaps)
- tqdm
- datasets

## Notes

- **Knowledge Neurons**: Currently skipped due to compatibility issues with Mamba's SSM architecture
- **Device Support**: Automatically detects and uses CUDA, MPS (Mac), or CPU
- **Performance**: Full analysis takes ~20 minutes on Mac with MPS
- **Memory**: Requires ~4GB GPU/unified memory for Mamba-130M

## Related Directories

Individual neuron type implementations are in their respective directories:
- `universality_neurons/` - Universality neuron analysis
- `delta_extraction/` - Delta parameter extraction
- `projection_neurons/` - Projection neuron analysis
- `speciality_neurons/` - Speciality neuron analysis
- `knowledge_neurons/` - Knowledge neuron analysis
- `dead_neurons/` - Dead neuron detection
- `causal_neurons/` - Causal neuron analysis
