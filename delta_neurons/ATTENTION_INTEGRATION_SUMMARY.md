# Attention Weights Integration Summary

## Overview
This document summarizes the integration of attention weights functionality from `attention_neurons.py` into `main.py`, following the pattern used in the reference project `LLM_paper/know_2`.

## What Was Integrated

### 1. Import Statements
- Added import for `MambaAttentionNeurons` and `integrate_mamba_attention_neurons` from `attention_neurons.py`

### 2. New Functions Added

#### `run_attention_analysis(model, tokenizer, texts, layer_indices, methods)`
- **Purpose**: Runs attention analysis for Mamba models using attention weights
- **Parameters**:
  - `model`: The model to analyze
  - `tokenizer`: The tokenizer
  - `texts`: List of texts to analyze
  - `layer_indices`: List of layer indices to analyze (default: [0, 6, 12, 18])
  - `methods`: List of methods for neuron creation (default: ['attention_weighted', 'gradient_guided', 'rollout'])
- **Returns**: Dictionary containing attention analysis results

#### `visualize_attention_neurons(attention_results, save_dir)`
- **Purpose**: Visualizes attention neurons and saves plots
- **Parameters**:
  - `attention_results`: Results from attention analysis
  - `save_dir`: Directory to save visualization plots (default: "images")
- **Features**: Creates visualizations for each method and layer, saves as PNG files

#### `analyze_attention_knowledge_extraction(attention_results, layer_idx)`
- **Purpose**: Analyzes how attention weights contribute to knowledge extraction
- **Parameters**:
  - `attention_results`: Results from attention analysis
  - `layer_idx`: Layer index to analyze
- **Returns**: Dictionary containing knowledge extraction analysis metrics

#### `print_attention_analysis_summary(attention_results, model_name)`
- **Purpose**: Prints a formatted summary of attention analysis results
- **Parameters**:
  - `attention_results`: Results from attention analysis
  - `model_name`: Name of the model analyzed

### 3. Enhanced Main Analysis Function

#### `run_comprehensive_analysis()`
- **Added**: Attention analysis for Mamba models
- **Condition**: Only runs when model has Mamba architecture (layers with mixer attribute) and attention is enabled
- **Integration**: Results are added to the main analysis results dictionary
- **Return Value**: Now returns `(results, high_var_neurons, attention_results)`

### 4. Command Line Arguments

#### New Arguments Added:
- `--enable_attention`: Enable attention weight analysis for Mamba models (default: True)
- `--attention_layers`: Layer indices for attention analysis (default: [0, 6, 12, 18])
- `--attention_methods`: Methods for attention neuron creation (default: ['attention_weighted', 'gradient_guided', 'rollout'])

### 5. Main Function Integration

#### Attention Analysis Flow:
1. **Detection**: Automatically detects Mamba models by checking for `mixer` attribute in layers
2. **Execution**: Runs attention analysis using specified layers and methods
3. **Storage**: Stores results in both `analysis_results` and `attention_results` sections
4. **Summary**: Prints attention analysis summary for each model
5. **Visualization**: Creates and saves attention neuron visualizations
6. **Results**: Includes attention analysis in saved results

#### Visualization Pipeline:
- Creates attention neuron plots for each method and layer
- Saves visualizations to `images/` directory
- Generates comprehensive image summary

## Usage Examples

### Basic Usage with Attention Analysis:
```bash
python main.py --models state-spaces/mamba-130m-hf gpt2 --enable_attention
```

### Custom Attention Analysis:
```bash
python main.py --models state-spaces/mamba-130m-hf \
    --enable_attention \
    --attention_layers 0 3 6 9 \
    --attention_methods attention_weighted gradient_guided
```

### Disable Attention Analysis:
```bash
python main.py --models state-spaces/mamba-130m-hf --no-enable_attention
```

## Output Files

### Attention Visualizations:
- `attention_neurons_attention_weighted_layer0.png`
- `attention_neurons_gradient_guided_layer0.png`
- `attention_neurons_rollout_layer0.png`
- (and similar for other layers)

### Enhanced Results JSON:
- `attention_analysis`: Summary of attention analysis
- `attention_knowledge_extraction`: Knowledge extraction metrics
- `has_attention_analysis`: Boolean flag indicating if attention analysis was performed

## Key Features

### 1. Automatic Mamba Detection
- Automatically detects Mamba models by checking architecture
- Only runs attention analysis on compatible models

### 2. Multiple Analysis Methods
- **attention_weighted**: Neurons weighted by attention vectors
- **gradient_guided**: Neurons guided by gradients (XAI vectors)
- **rollout**: Neurons using rollout attention method

### 3. Comprehensive Analysis
- Extracts attention vectors from multiple layers
- Creates mamba neurons using different methods
- Analyzes neuron behavior and importance
- Provides knowledge extraction insights

### 4. Rich Visualizations
- Neuron activation plots
- Attention heatmaps
- Importance score visualizations
- Top neuron comparisons

## Integration Benefits

### 1. Enhanced Analysis
- Provides deeper insights into Mamba model behavior
- Combines traditional neuron analysis with attention weights
- Enables knowledge extraction analysis

### 2. Consistent Interface
- Follows the same pattern as the reference project
- Integrates seamlessly with existing analysis pipeline
- Maintains backward compatibility

### 3. Flexible Configuration
- Configurable layers and methods via command line
- Easy to enable/disable attention analysis
- Customizable analysis parameters

### 4. Comprehensive Output
- Rich visualizations for analysis
- Detailed results for further processing
- Integration with existing result saving

## Testing

A test script `test_attention_integration.py` is provided to verify:
- Import functionality
- Function availability
- Integration correctness

Run with: `python test_attention_integration.py`

## Conclusion

The attention weights integration successfully brings the advanced attention analysis capabilities from `attention_neurons.py` into the main analysis pipeline, following the established patterns from the reference project. This integration provides comprehensive attention-based neuron analysis for Mamba models while maintaining the existing functionality and adding new insights into model behavior.
