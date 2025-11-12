#!/usr/bin/env python3
"""
Simple verification script to test Mamba2 functionality.
"""

import torch
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mamba_model_loader import load_mamba_model_and_tokenizer
from mamba2_layer import attach_mamba2_layers
from mamba2_verification import comprehensive_mamba2_verification

class SimpleAnalyzer:
    """Simple analyzer class for verification"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

def main():
    """Test Mamba2 verification with simple setup"""
    
    print("🔬 Simple Mamba2 Verification Test")
    print("=" * 50)
    
    # Load model and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        model, tokenizer = load_mamba_model_and_tokenizer(
            model_name="state-spaces/mamba-130m-hf",
            device=device,
            use_mamba_class=True,
            fallback_to_auto=True
        )
        
        # Attach Mamba2 layers
        num_added = attach_mamba2_layers(model)
        print(f"Attached Mamba2 to {num_added} layers")
        
        # Create simple analyzer
        analyzer = SimpleAnalyzer(model, tokenizer, device)
        
        print(f"Model loaded: state-spaces/mamba-130m-hf")
        print(f"Device: {device}")
        print(f"Model type: {type(model).__name__}")
        
        # Run comprehensive verification
        verification_result = comprehensive_mamba2_verification(analyzer)
        
        if verification_result:
            print("\n🎉 Mamba2 verification successful!")
            print("The improved Mamba2Layer is working correctly.")
            
            # Additional test: Check if we can access the verification function directly
            print("\n🔧 Testing direct verification function...")
            contribution_ratio = verify_mamba2_is_active(analyzer)
            print(f"Direct verification result: {contribution_ratio:.1%} contribution")
            
        else:
            print("\n⚠️ Mamba2 verification failed!")
            print("Check the output above for details.")
            
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()

def verify_mamba2_is_active(analyzer):
    """Check if Mamba2 is actually being used - direct version"""
    
    print("🔍 Verifying Mamba2 is active...")
    
    # 1. Check if grad_scale is being learned
    try:
        grad_scale = analyzer.model.backbone.layers[0].mamba2.grad_scale.item()
        print(f"📊 Grad scale: {grad_scale:.4f}")
    except AttributeError as e:
        print(f"❌ Could not access grad_scale: {e}")
        return False
    
    # 2. Test Mamba2 vs Original Mamba contribution
    test_text = "The quick brown fox jumps over the lazy dog."
    inputs = analyzer.tokenizer(test_text, return_tensors="pt").to(analyzer.device)
    
    with torch.no_grad():
        # Get full output
        full_out = analyzer.model(**inputs).logits
        
        # Disable Mamba2 temporarily
        original_grad_scale = analyzer.model.backbone.layers[0].mamba2.grad_scale.item()
        analyzer.model.backbone.layers[0].mamba2.grad_scale.data = torch.tensor(0.0)
        
        no_mamba2_out = analyzer.model(**inputs).logits
        
        # Restore
        analyzer.model.backbone.layers[0].mamba2.grad_scale.data = torch.tensor(original_grad_scale)
    
    # Calculate Mamba2 contribution
    mamba2_contribution = torch.norm(full_out - no_mamba2_out).item()
    total_norm = torch.norm(full_out).item()
    
    contribution_ratio = mamba2_contribution / total_norm
    
    print(f"\n📈 Mamba2 Contribution Analysis:")
    print(f"  - Contribution ratio: {contribution_ratio:.1%}")
    print(f"  - Logit difference: {mamba2_contribution:.4f}")
    print(f"  - Total logit norm: {total_norm:.4f}")
    
    if contribution_ratio > 0.01:  # More than 1%
        print("✅ Mamba2 is actively contributing!")
        return contribution_ratio
    else:
        print("⚠️ Mamba2 contribution is very small")
        return contribution_ratio

if __name__ == "__main__":
    main()
