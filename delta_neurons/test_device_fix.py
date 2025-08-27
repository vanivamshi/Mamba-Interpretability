#!/usr/bin/env python3
"""
Test script to verify device mismatch issues are fixed.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from delta_extraction import evaluate_perplexity, find_delta_sensitive_neurons_fixed, extract_deltas_fixed
from bottleneck_analysis import BottleneckAnalyzer

def test_device_consistency():
    """Test that device consistency is maintained throughout the analysis."""
    print("🔧 Testing device consistency...")
    
    # Test with a small model
    model_name = "gpt2"
    
    try:
        # Setup model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"✅ Model loaded and moved to {device}")
        
        # Test perplexity evaluation
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world."
        ]
        
        print("📊 Testing perplexity evaluation...")
        ppl = evaluate_perplexity(model, tokenizer, test_texts, device)
        print(f"✅ Perplexity evaluation successful: {ppl:.2f}")
        
        # Test delta extraction
        print("🔍 Testing delta extraction...")
        inputs = tokenizer(test_texts[0], return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        delta = extract_deltas_fixed(model, 0, input_ids)
        print(f"✅ Delta extraction successful: {delta.shape}")
        
        # Test finding delta-sensitive neurons
        print("🧠 Testing delta-sensitive neuron finding...")
        neurons = find_delta_sensitive_neurons_fixed(model, tokenizer, test_texts[:3], layer_idx=0, top_k=5)
        print(f"✅ Found {len(neurons)} delta-sensitive neurons")
        
        # Test bottleneck analyzer
        print("🔬 Testing bottleneck analyzer...")
        analyzer = BottleneckAnalyzer(model, tokenizer)
        print("✅ Bottleneck analyzer initialization successful")
        
        print("🎉 All device consistency tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Device consistency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_device_consistency() 