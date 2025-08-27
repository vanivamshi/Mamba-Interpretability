#!/usr/bin/env python3
"""
Test script to verify the attention weight integration is working properly.
"""

import torch
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_attention_neurons_import():
    """Test that we can import the attention neurons module."""
    try:
        from attention_neurons import MambaAttentionNeurons, integrate_mamba_attention_neurons
        print("✅ Successfully imported attention_neurons module")
        return True
    except ImportError as e:
        print(f"❌ Failed to import attention_neurons: {e}")
        return False

def test_attention_neurons_class():
    """Test that the MambaAttentionNeurons class can be instantiated."""
    try:
        from attention_neurons import MambaAttentionNeurons
        
        # Create a dummy model-like object for testing
        class DummyModel:
            def __init__(self):
                self.layers = []
                for i in range(20):
                    layer = DummyLayer()
                    self.layers.append(layer)
            
            def __call__(self, inputs):
                return torch.randn(inputs.shape[0], 512)
        
        class DummyLayer:
            def __init__(self):
                self.mixer = DummyMixer()
        
        class DummyMixer:
            def __init__(self):
                self.compute_attn_matrix = True
                self.attn_matrix_a = torch.randn(1, 8, 64, 64)  # batch, heads, seq, seq
                self.attn_matrix_b = torch.randn(1, 8, 64, 64)
                self.xai_b = torch.randn(1, 64, 512)
        
        model = DummyModel()
        analyzer = MambaAttentionNeurons(model, enable_attention_computation=True)
        print("✅ Successfully created MambaAttentionNeurons instance")
        return True
    except Exception as e:
        print(f"❌ Failed to create MambaAttentionNeurons instance: {e}")
        return False

def test_attention_analysis():
    """Test that the attention analysis can run."""
    try:
        from attention_neurons import integrate_mamba_attention_neurons
        
        # Create a dummy model-like object for testing
        class DummyModel:
            def __init__(self):
                self.layers = []
                for i in range(20):
                    layer = DummyLayer()
                    self.layers.append(layer)
            
            def __call__(self, inputs):
                return torch.randn(inputs.shape[0], 512)
        
        class DummyLayer:
            def __init__(self):
                self.mixer = DummyMixer()
        
        class DummyMixer:
            def __init__(self):
                self.compute_attn_matrix = True
                # Fix the tensor dimensions to match what the rollout method expects
                # The attention vectors need to be 4D: (batch, heads, seq, seq)
                self.attn_matrix_a = torch.randn(1, 8, 64, 64)  # batch, heads, seq, seq
                self.attn_matrix_b = torch.randn(1, 8, 64, 64)
                self.xai_b = torch.randn(1, 64, 512)
        
        model = DummyModel()
        inputs = torch.randint(0, 1000, (1, 64))
        
        results = integrate_mamba_attention_neurons(
            model=model,
            inputs=inputs,
            layer_indices=[0, 6, 12, 18],
            methods=['attention_weighted', 'gradient_guided', 'rollout']
        )
        
        print("✅ Successfully ran attention analysis")
        print(f"   - Results keys: {list(results.keys())}")
        if 'mamba_neurons' in results:
            print(f"   - Methods: {list(results['mamba_neurons'].keys())}")
        return True
    except Exception as e:
        print(f"❌ Failed to run attention analysis: {e}")
        return False

def test_main_integration():
    """Test that the main.py integration functions can be imported."""
    try:
        from main import run_attention_weight_analysis
        print("✅ Successfully imported run_attention_weight_analysis from main")
        return True
    except ImportError as e:
        print(f"❌ Failed to import from main: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Attention Weight Integration")
    print("=" * 50)
    
    tests = [
        ("Import attention_neurons", test_attention_neurons_import),
        ("Create MambaAttentionNeurons", test_attention_neurons_class),
        ("Run attention analysis", test_attention_analysis),
        ("Import main integration", test_main_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"   ❌ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Attention weight integration is working.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
