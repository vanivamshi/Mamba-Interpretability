# How to Ensure Activation Collection is Correct

## Overview

This guide provides comprehensive methods to validate and debug activation collection in your Mamba mechanistic analysis framework. Activation collection is the foundation of mechanistic interpretability, so ensuring it's correct is crucial.

## Quick Start

**Run the test script first:**
```bash
python test_activation_collection.py
```

This will quickly identify if your activation collection is working correctly.

## Common Issues and Solutions

### 1. **Model Layer Access Issues** (Most Common)

**Problem:** `utils.get_model_layers()` fails to find layers
**Symptoms:** "Could not find model layers" errors

**Solutions:**
- Use direct layer access: `model.layers` or `model.backbone.layers`
- Check model structure with debugging tools
- Use the improved ActivationCollector

**Code Fix:**
```python
# Instead of:
from utils import get_model_layers
layers = get_model_layers(model)

# Use:
if hasattr(model, 'layers'):
    layers = model.layers
elif hasattr(model, 'backbone') and hasattr(model.backbone, 'layers'):
    layers = model.backbone.layers
```

### 2. **Hook Registration Failures**

**Problem:** Hooks can't be registered on target layers
**Symptoms:** "Failed to register hook" errors

**Solutions:**
- Verify layer exists before registering hooks
- Use proper error handling
- Test with simple inputs first

### 3. **Activation Shape Inconsistencies**

**Problem:** Activations have unexpected shapes
**Symptoms:** Dimension mismatch errors in downstream analysis

**Solutions:**
- Validate activation shapes before processing
- Handle variable sequence lengths properly
- Use consistent padding/truncation

### 4. **Memory Issues**

**Problem:** Out of memory during activation collection
**Symptoms:** CUDA out of memory errors

**Solutions:**
- Process texts in smaller batches
- Use gradient checkpointing
- Clear activations between runs

## Validation Methods

### 1. **Basic Validation**

Use the test script to quickly check:
```bash
python test_activation_collection.py
```

### 2. **Comprehensive Validation**

For thorough validation:
```python
from activation_validation import run_comprehensive_validation

report = run_comprehensive_validation(model, tokenizer, texts)
```

### 3. **Debugging Suite**

For detailed debugging:
```python
from activation_debugging import run_activation_debugging_suite

debug_results = run_activation_debugging_suite(model, tokenizer)
```

## Validation Checklist

### ✅ Pre-Collection Checks
- [ ] Model loads successfully
- [ ] Tokenizer works with test texts
- [ ] Model layers are accessible
- [ ] Device is properly set

### ✅ During Collection Checks
- [ ] Hooks register successfully
- [ ] Forward pass completes without errors
- [ ] Activations are captured
- [ ] Activation shapes are reasonable
- [ ] No NaN or Inf values

### ✅ Post-Collection Checks
- [ ] Activations have expected properties
- [ ] Memory usage is reasonable
- [ ] Activations are consistent across runs
- [ ] Hooks are properly removed

## Best Practices

### 1. **Always Test First**
```python
# Test with simple inputs before full analysis
test_input = torch.tensor([[1, 2, 3, 4, 5]], device=device)
activations = collector.collect_activations(test_input)
```

### 2. **Use Error Handling**
```python
try:
    activations = collector.collect_activations(inputs)
    if not activations:
        logger.warning("No activations collected")
except Exception as e:
    logger.error(f"Activation collection failed: {e}")
```

### 3. **Validate Shapes**
```python
for layer_idx, activation in activations.items():
    if len(activation.shape) < 2:
        logger.warning(f"Unexpected activation shape: {activation.shape}")
```

### 4. **Monitor Memory**
```python
memory_usage = activation.element_size() * activation.nelement() / (1024**2)
if memory_usage > 1000:  # 1GB
    logger.warning(f"Large activation: {memory_usage:.1f}MB")
```

## Troubleshooting Guide

### Issue: "Could not find model layers"
**Solution:**
1. Check model structure: `print(dir(model))`
2. Try different access methods
3. Use the debugging tools

### Issue: "Hook registration failed"
**Solution:**
1. Verify layer exists: `layer_idx < len(layers)`
2. Check layer type compatibility
3. Test with simple inputs

### Issue: "No activations captured"
**Solution:**
1. Verify forward pass completes
2. Check hook is properly registered
3. Ensure model is in eval mode

### Issue: "Activation shapes inconsistent"
**Solution:**
1. Use consistent text preprocessing
2. Handle variable lengths properly
3. Validate shapes before concatenation

## Advanced Validation

### Consistency Testing
```python
# Test consistency across multiple runs
for run in range(3):
    activations = collector.collect_activations(inputs)
    # Compare with previous runs
```

### Property Validation
```python
# Check activation properties
assert not torch.isnan(activation).any()
assert not torch.isinf(activation).any()
assert activation.shape[0] > 0
```

### Performance Monitoring
```python
import time
start_time = time.time()
activations = collector.collect_activations(inputs)
collection_time = time.time() - start_time
logger.info(f"Collection took {collection_time:.2f}s")
```

## Integration with Your Analysis

### Update Your Main Script
```python
# In mamba_mechanistic_analysis.py, add validation:
def collect_activations(self, texts: List[str], layer_indices: List[int] = None):
    # ... existing code ...
    
    # Add validation
    from activation_validation import ActivationValidator
    validator = ActivationValidator(self.model, self.tokenizer, self.config.device)
    
    # Validate before collection
    validator.validate_model_structure()
    validator.validate_hook_registration(layer_indices)
    
    # ... rest of collection code ...
    
    # Validate after collection
    validator.validate_activation_properties(final_activations)
```

### Use Improved Collector
```python
# Replace ActivationCollector with improved version
from activation_debugging import create_improved_activation_collector
ImprovedCollector = create_improved_activation_collector()
collector = ImprovedCollector(self.model, self.config)
```

## Files Created

1. **`activation_validation.py`** - Comprehensive validation framework
2. **`activation_debugging.py`** - Debugging utilities and improved collector
3. **`test_activation_collection.py`** - Quick test script

## Next Steps

1. **Run the test script** to check your current setup
2. **Fix any issues** identified by the tests
3. **Integrate validation** into your main analysis pipeline
4. **Use the improved collector** for better reliability
5. **Monitor activation properties** during analysis

## Summary

Ensuring activation collection is correct involves:

1. **Testing** - Always test before running full analysis
2. **Validation** - Use comprehensive validation frameworks
3. **Debugging** - Debug issues systematically
4. **Monitoring** - Monitor properties and performance
5. **Error Handling** - Handle errors gracefully

The tools provided will help you identify and fix activation collection issues, ensuring reliable mechanistic analysis results.

