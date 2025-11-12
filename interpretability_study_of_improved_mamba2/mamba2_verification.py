import torch

def verify_mamba2_is_active(analyzer):
    """Check if Mamba2 is actually being used - integrated version"""
    
    print("🔍 Verifying Mamba2 is active...")
    
    # 1. Check if grad_scale is being learned
    try:
        grad_scale = analyzer.model.backbone.layers[0].mamba2.grad_scale.item()
        print(f"📊 Grad scale: {grad_scale:.4f}")
        # ✅ Should be close to 1.0 (default) or different if trained
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
        return True
    else:
        print("⚠️ Mamba2 contribution is very small")
        return False

def test_mamba2_parameters(analyzer):
    """Test Mamba2 parameter accessibility and values"""
    
    print("\n🔧 Testing Mamba2 parameters...")
    
    try:
        layer = analyzer.model.backbone.layers[0].mamba2
        
        # Check key parameters
        params_to_check = [
            'gate_weights',
            'timescale_weights', 
            'memory_gate',
            'grad_scale'
        ]
        
        for param_name in params_to_check:
            if hasattr(layer, param_name):
                param = getattr(layer, param_name)
                print(f"  ✅ {param_name}: shape={param.shape}, dtype={param.dtype}")
                print(f"     Value: {param.data.flatten()[:3].tolist()}...")
            else:
                print(f"  ❌ {param_name}: Not found")
        
        # Check SSM blocks
        ssm_blocks = ['ssm_fast', 'ssm_medium', 'ssm_slow']
        for ssm_name in ssm_blocks:
            if hasattr(layer, ssm_name):
                ssm = getattr(layer, ssm_name)
                print(f"  ✅ {ssm_name}: {type(ssm).__name__}")
                if hasattr(ssm, 'A'):
                    print(f"     A matrix shape: {ssm.A.shape}")
            else:
                print(f"  ❌ {ssm_name}: Not found")
                
        return True
        
    except Exception as e:
        print(f"❌ Error testing Mamba2 parameters: {e}")
        return False

def comprehensive_mamba2_verification(analyzer):
    """Comprehensive verification of Mamba2 functionality"""
    
    print("🔬 Comprehensive Mamba2 Verification")
    print("=" * 50)
    
    # Test 1: Parameter accessibility
    param_test = test_mamba2_parameters(analyzer)
    
    # Test 2: Active contribution
    active_test = verify_mamba2_is_active(analyzer)
    
    # Test 3: Forward pass with different texts
    print("\n🚀 Testing Mamba2 forward pass...")
    
    test_texts = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating."
    ]
    
    try:
        layer = analyzer.model.backbone.layers[0].mamba2
        
        for i, text in enumerate(test_texts):
            inputs = analyzer.tokenizer(text, return_tensors="pt").to(analyzer.device)
            
            # Get hidden states from the model
            with torch.no_grad():
                outputs = analyzer.model.backbone(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[0]  # First layer
                
                # Test Mamba2 forward pass
                mamba2_output = layer(hidden_states, layer_idx=0)
                
                print(f"  Text {i+1}: '{text[:20]}...'")
                print(f"    Input shape: {hidden_states.shape}")
                print(f"    Output shape: {mamba2_output.shape}")
                print(f"    Output mean: {mamba2_output.mean().item():.4f}")
                print(f"    Output std: {mamba2_output.std().item():.4f}")
        
        forward_test = True
        
    except Exception as e:
        print(f"❌ Error testing Mamba2 forward pass: {e}")
        forward_test = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Verification Summary:")
    print(f"  Parameter test: {'✅ PASS' if param_test else '❌ FAIL'}")
    print(f"  Forward pass test: {'✅ PASS' if forward_test else '❌ FAIL'}")
    print(f"  Active contribution test: {'✅ PASS' if active_test else '❌ FAIL'}")
    
    if param_test and forward_test and active_test:
        print("\n🎉 All tests passed! Mamba2 is working correctly.")
        return True
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
        return False
