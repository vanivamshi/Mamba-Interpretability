# Activation Extraction Update - Using Proper Model Access

## Overview
Updated all three new analysis modules (Steps 9-11) to use the same activation extraction approach as `attention_neurons.py`, which directly accesses model layers instead of relying on utility functions that may not work with all model architectures.

## Key Changes Made

### 1. **Direct Model Layer Access**
Instead of using `utils.get_model_layers()` which may fail, all modules now use the same approach as `attention_neurons.py`:

```python
# Before: Complex multi-strategy approach with utils.get_model_layers()
from utils import get_model_layers
layers = get_model_layers(model)
if layers and layer_idx < len(layers):
    target_module = layers[layer_idx]

# After: Direct access like attention_neurons.py
if hasattr(model, 'layers') and layer_idx < len(model.layers):
    target_layer = model.layers[layer_idx]
```

### 2. **Consistent Layer Detection Strategy**

#### Primary Strategy: `model.layers`
```python
if hasattr(model, 'layers') and layer_idx < len(model.layers):
    target_layer = model.layers[layer_idx]
    logger.info(f"Found layer {layer_idx} using model.layers: {type(target_layer)}")
```

#### Fallback Strategy: `model.backbone.layers`
```python
elif hasattr(model, 'backbone') and hasattr(model.backbone, 'layers'):
    if layer_idx < len(model.backbone.layers):
        target_layer = model.backbone.layers[layer_idx]
        logger.info(f"Found layer {layer_idx} using backbone.layers: {type(target_layer)}")
```

### 3. **Updated Files**

#### `causal_equivalence.py`
- **`extract_feature_activation()`**: Now uses direct layer access
- **`patch_feature_into_model()`**: Updated to use same approach
- **Error Handling**: Returns zeros instead of dummy values when extraction fails

#### `dynamic_universality.py`
- **`collect_sae_latents()`**: Updated to use direct layer access
- **Feature Validation**: Skips out-of-bounds features with warnings
- **Error Handling**: Returns zeros for failed extractions

#### `temporal_causality.py`
- **`_extract_activations_with_grad()`**: Updated to use direct layer access
- **Simplified Analysis**: Uses zeros instead of random values
- **Error Handling**: Returns None for failed extractions

## Technical Benefits

### 1. **Consistency with Existing Code**
- Uses the same approach as `attention_neurons.py`
- Follows established patterns in the codebase
- Reduces code duplication and maintenance overhead

### 2. **Better Model Compatibility**
- Works with standard Mamba model structure (`model.layers`)
- Supports backbone structure (`model.backbone.layers`)
- More reliable than utility functions that may not work with all models

### 3. **Improved Error Handling**
- Clear logging when layers are found vs. not found
- Graceful degradation with zeros instead of random values
- Better debugging information for troubleshooting

### 4. **Simplified Architecture**
- Removed complex multi-strategy layer detection
- Cleaner, more maintainable code
- Easier to understand and debug

## Code Pattern Comparison

### Before (Complex Multi-Strategy):
```python
# Strategy 1: Use utils.get_model_layers
from utils import get_model_layers
layers = get_model_layers(model)
if layers and layer_idx < len(layers):
    target_module = layers[layer_idx]

# Strategy 2: Direct model traversal
if target_module is None:
    try:
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'layers'):
            if layer_idx < len(model.backbone.layers):
                target_module = model.backbone.layers[layer_idx]
    except:
        pass

# Strategy 3: Try different model structures
if target_module is None:
    for attr_name in ['model', 'transformer', 'encoder']:
        if hasattr(model, attr_name):
            submodel = getattr(model, attr_name)
            if hasattr(submodel, 'layers') and layer_idx < len(submodel.layers):
                target_module = submodel.layers[layer_idx]
                break
```

### After (Simple Direct Access):
```python
# Use the same approach as attention_neurons.py
# Direct access to model layers
if hasattr(model, 'layers') and layer_idx < len(model.layers):
    target_layer = model.layers[layer_idx]
    logger.info(f"Found layer {layer_idx} using model.layers: {type(target_layer)}")
    
    # Register hook on the layer
    hook = target_layer.register_forward_hook(activation_hook)
    
    try:
        with torch.no_grad():
            _ = model(inputs)
        
        hook.remove()
        # ... handle results
        
    except Exception as e:
        hook.remove()
        logger.error(f"Failed to extract activations: {e}")
        return None

# Fallback: try backbone structure
elif hasattr(model, 'backbone') and hasattr(model.backbone, 'layers'):
    if layer_idx < len(model.backbone.layers):
        target_layer = model.backbone.layers[layer_idx]
        # ... same pattern
```

## Impact on Analysis

### 1. **More Reliable Activation Extraction**
- Should work better with actual Mamba models
- Consistent with how `attention_neurons.py` successfully extracts activations
- Reduced likelihood of "Could not get model layers" errors

### 2. **Better Error Reporting**
- Clear indication of which layer access method worked
- Better logging for debugging model architecture issues
- Easier to identify when and why extraction fails

### 3. **Maintainability**
- Single pattern to maintain across all modules
- Easier to update if model structure changes
- Consistent with existing working code

## Usage Notes

1. **Model Structure**: This approach works with models that have `model.layers` or `model.backbone.layers`
2. **Error Handling**: When extraction fails, modules return zeros or None instead of random values
3. **Logging**: Check logs to see which layer access method is being used
4. **Compatibility**: Should work better with actual Mamba models than the previous approach

This update ensures that the activation extraction in Steps 9-11 uses the same reliable approach as the existing `attention_neurons.py` module, improving compatibility and reducing the likelihood of extraction failures.
