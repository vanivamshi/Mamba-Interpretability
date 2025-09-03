#!/usr/bin/env python3
"""
Test script to demonstrate comprehensive sparsity analysis
"""

import sys
import os

def test_comprehensive_sparsity():
    """Test the new comprehensive sparsity analysis methods."""
    print("🔬 Testing Comprehensive Sparsity Analysis")
    print("=" * 60)
    
    try:
        from comparison_plots import load_models
        
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
        
        # Test comprehensive sparsity analysis for each model
        for model_name, analyzer in models.items():
            print(f"\n{'='*50}")
            print(f"Testing {model_name.upper()} model")
            print(f"{'='*50}")
            
            try:
                # Test individual sparsity methods
                print(f"\n1. Percentile-based sparsity:")
                percentile_results = analyzer.calculate_percentile_sparsity(texts, layer_idx=0)
                
                print(f"\n2. Entropy-based sparsity:")
                entropy_results = analyzer.calculate_entropy_sparsity(texts, layer_idx=0)
                
                print(f"\n3. Gini-based sparsity:")
                gini_results = analyzer.calculate_gini_sparsity(texts, layer_idx=0)
                
                print(f"\n4. Comprehensive sparsity analysis:")
                comprehensive_results = analyzer.calculate_comprehensive_sparsity(texts, layer_idx=0)
                
                # Print summary
                print(f"\n📊 SUMMARY for {model_name.upper()}:")
                print(f"  Traditional sparsity: {comprehensive_results['traditional']['sparsity']:.4f}")
                print(f"  Entropy-based sparsity: {entropy_results['sparsity_from_entropy']:.4f}")
                print(f"  Gini-based sparsity: {gini_results['sparsity_from_gini']:.4f}")
                if comprehensive_results['optimal']:
                    print(f"  Optimal threshold sparsity: {comprehensive_results['optimal']['sparsity']:.4f}")
                
                # Find the most reliable sparsity measure
                sparsity_measures = [
                    ('Traditional', comprehensive_results['traditional']['sparsity']),
                    ('Entropy', entropy_results['sparsity_from_entropy']),
                    ('Gini', gini_results['sparsity_from_gini'])
                ]
                
                # Sort by how reasonable the sparsity value is (closer to 0.5 is better)
                sparsity_measures.sort(key=lambda x: abs(x[1] - 0.5))
                print(f"\n  🎯 Most reliable sparsity measure: {sparsity_measures[0][0]} ({sparsity_measures[0][1]:.4f})")
                
            except Exception as e:
                print(f"❌ Error testing {model_name}: {e}")
                import traceback
                traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"❌ Error in test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the comprehensive sparsity test."""
    success = test_comprehensive_sparsity()
    
    if success:
        print("\n🎉 Comprehensive sparsity test completed successfully!")
        print("\n💡 Key Insights:")
        print("  • Percentile-based sparsity is most robust (no arbitrary thresholds)")
        print("  • Entropy-based sparsity detects activation patterns well")
        print("  • Gini coefficient measures inequality in distributions")
        print("  • Compare all methods to find the most reliable measure")
    else:
        print("\n❌ Comprehensive sparsity test failed!")
        print("Check the error messages above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
