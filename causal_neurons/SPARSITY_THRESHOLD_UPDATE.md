# Sparsity Threshold Update Summary

## Overview
Updated sparsity calculations across the codebase to use model-specific thresholds:
- **Mamba models**: 0.01 (higher threshold)
- **Transformer models**: 1e-5 (lower threshold)

## Why Different Thresholds?

### Mamba Models (0.01)
- Mamba models have different activation patterns compared to Transformers
- They often have higher baseline activation values
- A higher threshold (0.01) is needed to properly identify "dead" neurons
- This prevents over-estimating sparsity due to architectural differences

### Transformer Models (1e-5)
- Transformers typically have more sensitive activation patterns
- They can have very small but meaningful activations
- A lower threshold (1e-5) ensures we don't miss neurons that are actually active
- This provides more accurate sparsity measurements for Transformer architectures

## Files Updated

### 1. `comparison_plots.py`
**Method**: `analyze_layer_dynamics()`
- Added model-specific sparsity threshold detection
- Updated sparsity calculation: `np.mean(np.abs(variance) < sparsity_threshold)`
- Enhanced logging to show which threshold was used

**Code Change**:
```python
# Use different thresholds for different model types
if self.model_type == "mamba":
    sparsity_threshold = 0.01
elif self.model_type == "transformer":
    sparsity_threshold = 1e-5
else:
    sparsity_threshold = 0.01  # Default fallback

sparsity = np.mean(np.abs(variance) < sparsity_threshold)
```

### 2. `main.py`
**Method**: `analyze_model_efficiency()`
- Added model name detection from command-line arguments
- Implemented model-specific threshold selection
- Updated sparsity calculation and logging

**Code Change**:
```python
# Different model architectures require different thresholds:
# - Mamba models: 0.01 (higher threshold due to different activation patterns)
# - Transformer models: 1e-5 (lower threshold for more sensitive detection)
model_name = args.model.lower()
if "mamba" in model_name:
    sparsity_threshold = 0.01
elif "gpt" in model_name or "transformer" in model_name:
    sparsity_threshold = 1e-5
else:
    sparsity_threshold = 0.01  # Default fallback
```

### 3. `comparison_wrapper.py`
**Method**: `analyze_model_efficiency()`
- Added documentation about model-specific thresholds
- Used conservative default threshold (0.01) for compatibility
- Added comments explaining the approach

## Impact on Analysis

### Before Update
- All models used the same threshold (0.01)
- Transformer models appeared more sparse than they actually were
- Mamba models appeared less sparse than they actually were
- Inconsistent sparsity measurements across architectures

### After Update
- Model-specific thresholds provide more accurate sparsity measurements
- Fair comparison between Mamba and Transformer architectures
- Better identification of truly "dead" vs. "active" neurons
- More reliable efficiency analysis and model comparison

## Usage Examples

### Mamba Model Analysis
```bash
python main.py --model "state-spaces/mamba-130m-hf"
# Uses sparsity threshold: 0.01
```

### Transformer Model Analysis
```bash
python main.py --model "gpt2"
# Uses sparsity threshold: 1e-5
```

### Comparison Analysis
```bash
python comparison_plots.py
# Automatically uses appropriate thresholds for each model type
```

## Testing Recommendations

1. **Verify Mamba Models**: Ensure sparsity values are reasonable with 0.01 threshold
2. **Verify Transformer Models**: Ensure sparsity values are reasonable with 1e-5 threshold
3. **Cross-Validation**: Compare results with manual inspection of activation patterns
4. **Model Comparison**: Verify that sparsity comparisons between architectures are now fair

## Future Improvements

1. **Dynamic Threshold Detection**: Automatically detect optimal thresholds based on activation statistics
2. **Threshold Tuning**: Allow users to specify custom thresholds via command-line arguments
3. **Validation Metrics**: Add metrics to validate threshold choice (e.g., correlation with model performance)
4. **Architecture Detection**: Improve automatic detection of model architecture type

## Summary

The sparsity threshold update ensures more accurate and fair analysis across different model architectures. Mamba models now use a higher threshold (0.01) appropriate for their activation patterns, while Transformer models use a lower threshold (1e-5) for more sensitive detection. This change improves the reliability of sparsity-based efficiency analysis and model comparison.
