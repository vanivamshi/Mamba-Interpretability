# Attention Neurons Integration Summary

## Overview
Successfully integrated `attention_neurons.py` into `main.py` following the pattern from `dead_9/main.py`.

## What Was Integrated

### 1. Import Statement
```python
from attention_neurons import integrate_mamba_attention_neurons, MambaAttentionNeurons
```

### 2. New Function: `run_attention_neurons_analysis()`
- **Purpose**: Runs attention neurons analysis for Mamba models
- **Parameters**: 
  - `model`: The Mamba model to analyze
  - `tokenizer`: Tokenizer for text processing
  - `texts`: List of texts for analysis
  - `layer_idx`: Layer index to analyze (default: 0)
  - `methods`: List of analysis methods (default: ['attention_weighted', 'gradient_guided', 'rollout'])
  - `plots_dir`: Directory to save plots (default: 'plots')

### 3. Command-Line Arguments
Added new command-line options:
- `--enable_attention`: Enable attention neurons analysis (default: True)
- `--attention_methods`: Specify methods for attention analysis
- `--layer`: Layer index to analyze (default: 0)
- `--text_limit`: Limit number of texts to process (default: 100)
- `--save_results`: Save analysis results to JSON file
- `--plots_dir`: Directory to save plots (default: 'plots')

### 4. Analysis Pipeline Integration
Added attention neurons analysis as **Stage 4** in the main analysis pipeline:
1. Basic Mamba analysis
2. Inter-layer analysis  
3. Cross-layer analysis
4. **Attention neurons analysis** ← NEW
5. Efficiency analysis
6. Comparison analysis

### 5. Results Storage
- Attention neurons results are stored in `analysis_results['attention_neurons']`
- Visualizations are automatically saved to the plots directory
- Results can be saved to JSON files using `--save_results` flag

### 6. Enhanced Summary Output
The final summary now includes:
- Attention neurons analysis completion status
- Key metrics (neurons analyzed, mean activation, neuron diversity)
- Information about saved visualizations

## Usage Examples

### Basic Usage
```bash
python main.py --enable_attention
```

### Custom Configuration
```bash
python main.py \
  --model "state-spaces/mamba-130m-hf" \
  --layer 2 \
  --enable_attention \
  --attention_methods attention_weighted gradient_guided \
  --text_limit 200 \
  --save_results \
  --plots_dir "my_plots"
```

### Disable Attention Analysis
```bash
python main.py --enable_attention false
```

## Features

### Analysis Methods
1. **attention_weighted**: Creates neurons weighted by attention vectors
2. **gradient_guided**: Creates neurons guided by gradients (XAI vectors)
3. **rollout**: Creates neurons using rollout attention method

### Visualizations
- Neuron activations bar chart
- Neuron importance scores
- Attention heatmaps
- Top neurons comparison
- All plots saved as high-resolution PNG files

### Output Files
- `attention_neurons_layer_X.png`: Attention neurons visualizations
- `neuron_analysis_results_YYYYMMDD_HHMMSS.json`: Analysis results (if `--save_results` used)

## Integration Pattern
Followed the successful integration pattern from `dead_9/main.py`:
- Same function structure and error handling
- Consistent command-line argument handling
- Similar results storage and visualization approach
- Compatible with existing analysis pipeline

## Testing
- All imports work correctly
- Functions can be instantiated and called
- No syntax errors in the integrated code
- Ready for production use

## Dependencies
- `torch`: PyTorch for deep learning operations
- `numpy`: Numerical computing
- `matplotlib`: Plotting and visualization
- `transformers`: Hugging Face transformers library
- `datasets`: Hugging Face datasets library

The integration is complete and ready to use!
