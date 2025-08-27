#!/usr/bin/env python3
"""
Test script to demonstrate multi-layer comprehensive sparsity plotting
"""

import sys
import os

def test_multi_layer_sparsity():
    """Test the new multi-layer comprehensive sparsity plotting."""
    print("🔬 Testing Multi-Layer Comprehensive Sparsity Plotting")
    print("=" * 60)
    
    try:
        from comparison_plots import load_models, create_comparison_plots
        
        # Load models
        print("Loading models...")
        models = load_models()
        
        if not models:
            print("❌ No models loaded successfully")
            return False
        
        print(f"✅ Loaded {len(models)} models: {list(models.keys())}")
        
        # Test texts
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world.",
            "Machine learning models require large datasets.",
            "Natural language processing involves understanding text.",
            "Deep learning has revolutionized computer vision."
        ]
        
        # Test multi-layer analysis for each model
        for model_name, analyzer in models.items():
            print(f"\n{'='*50}")
            print(f"Testing {model_name.upper()} model multi-layer analysis")
            print(f"{'='*50}")
            
            try:
                # Test layer dynamics (this will now analyze multiple layers)
                print(f"\n🔬 Running multi-layer analysis...")
                layer_dynamics = analyzer.analyze_layer_dynamics(texts)
                
                print(f"  Layer variances: {[f'{v:.2e}' for v in layer_dynamics['layer_variances']]}")
                print(f"  Layer sparsities: {[f'{s:.4f}' for s in layer_dynamics['layer_sparsity']]}")
                
                if 'comprehensive_sparsity' in layer_dynamics:
                    print(f"  Comprehensive sparsity data available for layers: {list(layer_dynamics['comprehensive_sparsity'].keys())}")
                    
                    for layer_idx, comp_data in layer_dynamics['comprehensive_sparsity'].items():
                        print(f"\n    Layer {layer_idx} comprehensive analysis:")
                        if 'percentile' in comp_data and 'p50' in comp_data['percentile']:
                            print(f"      P50 sparsity: {comp_data['percentile']['p50']['sparsity']:.4f}")
                        if 'entropy' in comp_data:
                            print(f"      Entropy sparsity: {comp_data['entropy']['sparsity_from_entropy']:.4f}")
                        if 'gini' in comp_data:
                            print(f"      Gini sparsity: {comp_data['gini']['sparsity_from_gini']:.4f}")
                else:
                    print("  ⚠️  No comprehensive sparsity data available")
                
            except Exception as e:
                print(f"❌ Error testing {model_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Run the full comparison to generate all plots
        print(f"\n{'='*50}")
        print("Running full comparison to generate plots...")
        print(f"{'='*50}")
        create_comparison_plots(models, texts)
        
        print(f"\n🎉 Generated plots:")
        print(f"  • Comprehensive Sparsity Analysis Multi-Layer")
        print(f"  • Comprehensive Sparsity Comparison Multi-Layer Simple")
        print(f"  • Plus all existing plots")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the multi-layer sparsity test."""
    success = test_multi_layer_sparsity()
    
    if success:
        print("\n🎉 Multi-layer comprehensive sparsity test completed successfully!")
        print("\n💡 What's New:")
        print("  • Multi-layer comprehensive sparsity analysis (layers 0, 1, 2)")
        print("  • New plots showing sparsity across layers using different methods")
        print("  • P50 percentile sparsity (most reliable) across multiple layers")
        print("  • Enhanced visualizations for model comparison")
    else:
        print("\n❌ Multi-layer comprehensive sparsity test failed!")
        print("Check the error messages above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
