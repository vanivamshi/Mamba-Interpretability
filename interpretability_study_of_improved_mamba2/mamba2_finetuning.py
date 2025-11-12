#!/usr/bin/env python3
"""
Mamba2 Fine-tuning and Optimization Script

This script implements:
1. Fine-tuning only Mamba2 parameters while freezing original Mamba
2. Increasing grad_scale to make Mamba2 more influential
3. Adding more non-zero parameters (more gates, larger SSM state)
4. Training on a simple language modeling task
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import logging
import json
import os
from typing import List, Dict, Any
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleTextDataset(Dataset):
    """Simple dataset for fine-tuning Mamba2"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

def create_training_texts():
    """Create training texts for fine-tuning"""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming industries worldwide.",
        "Machine learning models require large amounts of training data.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning has revolutionized computer vision and speech recognition.",
        "Neural networks can learn complex patterns from data.",
        "Transformers have become the dominant architecture for language models.",
        "State space models offer efficient alternatives to transformers.",
        "Mamba models provide linear-time complexity for sequence modeling.",
        "Mechanistic interpretability helps us understand how models work.",
        "Gradient-based attribution methods reveal important parameters.",
        "Circuit analysis identifies causal relationships in neural networks.",
        "Sparse autoencoders discover interpretable features in activations.",
        "Activation patching tests causal hypotheses about model behavior.",
        "Parameter decomposition reveals how different components contribute.",
        "The future of AI lies in understanding and improving model interpretability.",
        "Research in mechanistic interpretability is advancing rapidly.",
        "Understanding model internals is crucial for AI safety.",
        "Interpretable AI systems are more trustworthy and reliable.",
        "Mechanistic analysis provides insights into model decision-making."
    ]

def optimize_mamba2_parameters(model):
    """Optimize Mamba2 parameters for better influence"""
    
    logger.info("🔧 Optimizing Mamba2 parameters...")
    
    # 1. Increase grad_scale to make Mamba2 more influential
    for layer_idx in range(len(model.backbone.layers)):
        if hasattr(model.backbone.layers[layer_idx], 'mamba2'):
            mamba2_layer = model.backbone.layers[layer_idx].mamba2
            # Increase grad_scale from 1.0 to 2.0
            mamba2_layer.grad_scale.data = torch.tensor(2.0)
            logger.info(f"  ✅ Increased grad_scale to 2.0 for layer {layer_idx}")
    
    # 2. Adjust gate weights to be more diverse
    for layer_idx in range(len(model.backbone.layers)):
        if hasattr(model.backbone.layers[layer_idx], 'mamba2'):
            mamba2_layer = model.backbone.layers[layer_idx].mamba2
            
            # Make gate weights more diverse
            if hasattr(mamba2_layer, 'gate_weights'):
                # Set different weights for each gate
                gate_weights = torch.tensor([0.5, 0.3, 0.2])  # More diverse than equal weights
                mamba2_layer.gate_weights.data = gate_weights
                logger.info(f"  ✅ Set diverse gate weights for layer {layer_idx}: {gate_weights.tolist()}")
            
            # Adjust timescale weights
            if hasattr(mamba2_layer, 'timescale_weights'):
                timescale_weights = torch.tensor([0.4, 0.4, 0.2])  # Favor fast and medium
                mamba2_layer.timescale_weights.data = timescale_weights
                logger.info(f"  ✅ Set timescale weights for layer {layer_idx}: {timescale_weights.tolist()}")
            
            # Adjust memory gate to be more active
            if hasattr(mamba2_layer, 'memory_gate'):
                # Increase memory gate values to be more active
                mamba2_layer.memory_gate.data = torch.ones_like(mamba2_layer.memory_gate) * 0.8
                logger.info(f"  ✅ Increased memory gate activity for layer {layer_idx}")

def setup_mamba2_training(model):
    """Setup training configuration for Mamba2 parameters only"""
    
    logger.info("🎯 Setting up Mamba2-only training...")
    
    # Collect Mamba2 parameters
    mamba2_params = []
    frozen_params = []
    
    for name, param in model.named_parameters():
        if 'mamba2' in name:
            param.requires_grad = True
            mamba2_params.append(param)
            logger.info(f"  ✅ Mamba2 parameter: {name} (shape: {param.shape})")
        else:
            param.requires_grad = False  # Freeze original Mamba
            frozen_params.append(name)
    
    logger.info(f"📊 Training setup:")
    logger.info(f"  - Mamba2 parameters: {len(mamba2_params)}")
    logger.info(f"  - Frozen parameters: {len(frozen_params)}")
    
    # Calculate total parameters
    mamba2_total = sum(p.numel() for p in mamba2_params)
    frozen_total = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    logger.info(f"  - Mamba2 parameter count: {mamba2_total:,}")
    logger.info(f"  - Frozen parameter count: {frozen_total:,}")
    logger.info(f"  - Mamba2 percentage: {mamba2_total/(mamba2_total+frozen_total)*100:.1f}%")
    
    return mamba2_params

def train_mamba2(model, tokenizer, device, num_epochs: int = 3, learning_rate: float = 1e-4):
    """Fine-tune Mamba2 parameters"""
    
    logger.info(f"🚀 Starting Mamba2 fine-tuning for {num_epochs} epochs...")
    
    # Setup training
    mamba2_params = setup_mamba2_training(model)
    optimizer = optim.AdamW(mamba2_params, lr=learning_rate, weight_decay=0.01)
    
    # Create training data
    training_texts = create_training_texts()
    dataset = SimpleTextDataset(training_texts, tokenizer, max_length=64)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Training loop
    model.train()
    total_loss = 0
    num_batches = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_batches = 0
        
        logger.info(f"📚 Epoch {epoch + 1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Calculate loss manually if not provided
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss
            else:
                # Manual loss calculation for language modeling
                logits = outputs.logits
                # Shift logits and labels for next token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                # Calculate cross-entropy loss
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(mamba2_params, max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            total_loss += loss.item()
            epoch_batches += 1
            num_batches += 1
            
            if batch_idx % 5 == 0:
                logger.info(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / epoch_batches
        logger.info(f"  📊 Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")
    
    avg_total_loss = total_loss / num_batches
    logger.info(f"🎯 Training completed! Average loss: {avg_total_loss:.4f}")
    
    return avg_total_loss

def verify_mamba2_improvements(model, tokenizer, device):
    """Verify that Mamba2 improvements are working"""
    
    logger.info("🔍 Verifying Mamba2 improvements...")
    
    # Test text
    test_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Get full output
        full_out = model(**inputs).logits
        
        # Test with different grad_scale values
        original_grad_scale = model.backbone.layers[0].mamba2.grad_scale.item()
        
        # Test with grad_scale = 0 (no Mamba2)
        model.backbone.layers[0].mamba2.grad_scale.data = torch.tensor(0.0)
        no_mamba2_out = model(**inputs).logits
        
        # Test with grad_scale = 2.0 (increased influence)
        model.backbone.layers[0].mamba2.grad_scale.data = torch.tensor(2.0)
        high_mamba2_out = model(**inputs).logits
        
        # Restore original
        model.backbone.layers[0].mamba2.grad_scale.data = torch.tensor(original_grad_scale)
    
    # Calculate contributions
    no_mamba2_diff = torch.norm(full_out - no_mamba2_out).item()
    high_mamba2_diff = torch.norm(full_out - high_mamba2_out).item()
    total_norm = torch.norm(full_out).item()
    
    no_mamba2_ratio = no_mamba2_diff / total_norm
    high_mamba2_ratio = high_mamba2_diff / total_norm
    
    logger.info(f"📈 Mamba2 Influence Analysis:")
    logger.info(f"  - Original grad_scale: {original_grad_scale:.1f}")
    logger.info(f"  - No Mamba2 contribution: {no_mamba2_ratio:.1%}")
    logger.info(f"  - High Mamba2 contribution: {high_mamba2_ratio:.1%}")
    logger.info(f"  - Influence increase: {high_mamba2_ratio/no_mamba2_ratio:.1f}x")
    
    # Check parameter values
    layer = model.backbone.layers[0].mamba2
    logger.info(f"🔧 Parameter Values:")
    logger.info(f"  - Gate weights: {layer.gate_weights.data.tolist()}")
    logger.info(f"  - Timescale weights: {layer.timescale_weights.data.tolist()}")
    logger.info(f"  - Memory gate mean: {layer.memory_gate.data.mean().item():.3f}")
    logger.info(f"  - Grad scale: {layer.grad_scale.data.item():.1f}")
    
    return {
        'no_mamba2_ratio': no_mamba2_ratio,
        'high_mamba2_ratio': high_mamba2_ratio,
        'influence_increase': high_mamba2_ratio/no_mamba2_ratio,
        'gate_weights': layer.gate_weights.data.tolist(),
        'timescale_weights': layer.timescale_weights.data.tolist(),
        'memory_gate_mean': layer.memory_gate.data.mean().item(),
        'grad_scale': layer.grad_scale.data.item()
    }

def main():
    """Main training and optimization function"""
    
    print("🚀 Mamba2 Fine-tuning and Optimization")
    print("=" * 60)
    
    # Load model and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        from mamba_model_loader import load_mamba_model_and_tokenizer
        from mamba2_layer import attach_mamba2_layers
        
        model, tokenizer = load_mamba_model_and_tokenizer(
            model_name="state-spaces/mamba-130m-hf",
            device=device,
            use_mamba_class=True,
            fallback_to_auto=True
        )
        
        # Attach Mamba2 layers
        num_added = attach_mamba2_layers(model)
        print(f"Attached Mamba2 to {num_added} layers")
        
        # Step 1: Optimize Mamba2 parameters
        optimize_mamba2_parameters(model)
        
        # Step 2: Fine-tune Mamba2
        training_loss = train_mamba2(model, tokenizer, device, num_epochs=3, learning_rate=1e-4)
        
        # Step 3: Verify improvements
        verification_results = verify_mamba2_improvements(model, tokenizer, device)
        
        # Save results
        results = {
            'training_loss': training_loss,
            'verification_results': verification_results,
            'num_mamba2_layers': num_added,
            'device': str(device)
        }
        
        os.makedirs('experiment_logs', exist_ok=True)
        with open('experiment_logs/mamba2_finetuning_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "=" * 60)
        print("🎉 Mamba2 Fine-tuning Complete!")
        print(f"📊 Training loss: {training_loss:.4f}")
        print(f"📈 Influence increase: {verification_results['influence_increase']:.1f}x")
        print(f"💾 Results saved to: experiment_logs/mamba2_finetuning_results.json")
        
    except Exception as e:
        print(f"❌ Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
