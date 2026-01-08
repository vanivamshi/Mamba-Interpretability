#!/usr/bin/env python3
# train mamba model on wikitext-2-v1 dataset. then run induction analysis on the model.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from induction import (
    MambaInductionAnalyzer, 
    integrate_induction_analysis,
    create_induction_test_sequences
)
from induction_investigation import investigate_mamba_induction_mystery
from datetime import datetime
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create timestamp for plot filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Analysis timestamp: {timestamp}")

# Ensure plots directory exists
plots_dir = 'plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    print(f"Created plots directory: {plots_dir}")

# Define summary file path early so it's available throughout
summary_file = f'plots/analysis_summary_{timestamp}.txt'

# ---------------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------------

print("\n=== Loading Dataset ===")
try:
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")
    texts = [item["text"] for item in dataset if item["text"].strip() != ""]
    # Limit dataset size for faster training (adjust as needed)
    texts = texts[:1000]  # Use first 1000 samples for training
    print(f"Loaded {len(texts)} non-empty samples from Wikitext.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Machine learning models require large datasets.",
        "Natural language processing involves understanding text."
    ]
    print(f"Using {len(texts)} dummy texts instead.")

# ---------------------------------------------------------------
# Load Models
# ---------------------------------------------------------------

print("\nLoading Mamba model...")
mamba_model_name = "state-spaces/mamba-130m-hf"
mamba_tokenizer = AutoTokenizer.from_pretrained(mamba_model_name)
# Add padding token if it doesn't exist
if mamba_tokenizer.pad_token is None:
    mamba_tokenizer.pad_token = mamba_tokenizer.eos_token
mamba_model = AutoModelForCausalLM.from_pretrained(mamba_model_name).to(device)
print("Mamba model loaded.")

# ---------------------------------------------------------------
# Fine-tune Model on Dataset
# ---------------------------------------------------------------

print("\n" + "="*60)
print("=== Fine-tuning Mamba Model on Dataset ===")
print("="*60)

def fine_tune_model(model, tokenizer, texts, num_epochs=1, batch_size=1, learning_rate=5e-5, max_length=256, gradient_accumulation_steps=4):
    """
    Fine-tune the model on the provided texts.
    Optimized for memory efficiency.
    """
    from torch.utils.data import Dataset, DataLoader
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    class TextDataset(Dataset):
        def __init__(self, texts, tokenizer, max_length=256):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = self.texts[idx]
            # Tokenize and truncate (no padding here, will pad in collate)
            encoding = tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze() if 'attention_mask' in encoding else torch.ones_like(encoding['input_ids'].squeeze())
            }
    
    def collate_fn(batch):
        """Custom collate function to pad sequences to same length"""
        input_ids = [item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]
        
        # Pad sequences
        max_len = max(len(ids) for ids in input_ids)
        padded_input_ids = []
        padded_attention_masks = []
        
        for ids, mask in zip(input_ids, attention_masks):
            pad_length = max_len - len(ids)
            padded_input_ids.append(torch.cat([ids, torch.zeros(pad_length, dtype=ids.dtype)]))
            padded_attention_masks.append(torch.cat([mask, torch.zeros(pad_length, dtype=mask.dtype)]))
        
        return {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_attention_masks)
        }
    
    # Create dataset and dataloader
    train_dataset = TextDataset(texts, tokenizer, max_length=max_length)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=False  # Disable pin_memory to save memory
    )
    
    # Setup optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Set model to training mode
    model.train()
    
    print(f"Training on {len(texts)} samples for {num_epochs} epoch(s)...")
    print(f"Batch size: {batch_size}, Max length: {max_length}, Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    
    total_loss = 0
    num_batches = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                
                # Scale loss by gradient accumulation steps
                loss = loss / gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update weights only after accumulating gradients
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Clear cache periodically
                    if (batch_idx + 1) % (gradient_accumulation_steps * 10) == 0:
                        torch.cuda.empty_cache()
                
                epoch_loss += loss.item() * gradient_accumulation_steps
                total_loss += loss.item() * gradient_accumulation_steps
                num_batches += 1
                
                # Print progress every 50 batches
                if (batch_idx + 1) % 50 == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {avg_loss:.4f}")
                    
            except torch.cuda.OutOfMemoryError as e:
                print(f"\n⚠️  CUDA OOM at batch {batch_idx+1}. Clearing cache and reducing batch size...")
                torch.cuda.empty_cache()
                # Try with smaller batch or skip this batch
                if batch_size > 1:
                    print("Consider reducing batch_size further or max_length")
                continue
        
        # Final gradient update if there are remaining gradients
        if (batch_idx + 1) % gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        avg_epoch_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
        print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Clear cache after each epoch
        torch.cuda.empty_cache()
    
    # Set model back to evaluation mode
    model.eval()
    
    # Final cache clear
    torch.cuda.empty_cache()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    print(f"\nFine-tuning complete! Average loss: {avg_loss:.4f}")
    
    return model

# Fine-tune the model (memory-optimized settings)
mamba_model = fine_tune_model(
    mamba_model,
    mamba_tokenizer,
    texts,
    num_epochs=1,  # Adjust number of epochs as needed
    batch_size=1,  # Reduced batch size to avoid OOM
    learning_rate=5e-5,
    max_length=256,  # Reduced sequence length to save memory
    gradient_accumulation_steps=4  # Accumulate gradients to simulate larger batch
)

print("Model fine-tuning completed. Switching to evaluation mode.")
mamba_model.eval()

# ---------------------------------------------------------------
# Induction Head Analysis (Add after model loading)
# ---------------------------------------------------------------

print("\n" + "="*60)
print("=== Mamba Induction Head Analysis ===")
print("="*60)

# Create test sequences with clear induction patterns
induction_sequences = create_induction_test_sequences()

# Also test on actual data samples
print("\nTesting on synthetic induction sequences...")
induction_results, induction_analyzer = integrate_induction_analysis(
    mamba_model, 
    mamba_tokenizer, 
    sequences=induction_sequences,
    save_dir=plots_dir
)

# Analyze real text samples for natural induction patterns
print("\nTesting on real text samples for natural induction patterns...")
real_text_samples = [
    "The cat sat on the mat. The cat jumped on the table.",
    "Alice went to the store. Bob went to the park. Alice went to the library.",
    "First we add two and two. Then we multiply by three. First we add five and five.",
]

real_text_results, _ = integrate_induction_analysis(
    mamba_model,
    mamba_tokenizer,
    sequences=real_text_samples,
    save_dir=plots_dir
)

# ---------------------------------------------------------------
# Detailed Prediction Analysis
# ---------------------------------------------------------------

print("\n" + "="*60)
print("Analyzing prediction accuracy on induction patterns...")
print("="*60)

def analyze_induction_predictions(results):
    """Analyze how well the model predicts based on induction."""
    all_predictions = []
    
    for key, value in results.items():
        if key.endswith('_predictions'):
            all_predictions.extend(value)
    
    # Separate by induction score
    high_induction = [p for p in all_predictions if p['induction_score'] > 0.5]
    low_induction = [p for p in all_predictions if p['induction_score'] <= 0.5]
    
    high_acc = sum(p['correct'] for p in high_induction) / len(high_induction) if high_induction else 0
    low_acc = sum(p['correct'] for p in low_induction) / len(low_induction) if low_induction else 0
    
    return {
        'high_induction_accuracy': high_acc,
        'low_induction_accuracy': low_acc,
        'num_high_induction': len(high_induction),
        'num_low_induction': len(low_induction),
        'high_induction_samples': high_induction[:5],
        'low_induction_samples': low_induction[:5]
    }

pred_analysis = analyze_induction_predictions(induction_results)

print(f"\nPrediction Accuracy Analysis:")
print(f"  High induction (score > 0.5): {pred_analysis['high_induction_accuracy']:.2%} " +
      f"({pred_analysis['num_high_induction']} samples)")
print(f"  Low induction (score ≤ 0.5): {pred_analysis['low_induction_accuracy']:.2%} " +
      f"({pred_analysis['num_low_induction']} samples)")

# ---------------------------------------------------------------
# Layer-wise Induction Analysis
# ---------------------------------------------------------------

print("\n" + "="*60)
print("Analyzing induction behavior across layers...")
print("="*60)

def analyze_layer_wise_induction(model, tokenizer, sequence, layer_indices=None):
    """Analyze how induction patterns emerge across layers."""
    if layer_indices is None:
        layer_indices = [0, 6, 12, 18, 23]  # Sample layers for 130M model
    
    device = next(model.parameters()).device
    tokens = tokenizer(sequence, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
    
    input_ids = tokens['input_ids'][0]
    
    layer_induction_scores = {}
    
    for layer_idx in layer_indices:
        if layer_idx >= len(outputs.hidden_states):
            continue
            
        hidden = outputs.hidden_states[layer_idx][0]
        seq_len = hidden.shape[0]
        
        # Compute induction scores for this layer
        scores = torch.zeros(seq_len)
        for i in range(1, seq_len):
            current_token = input_ids[i]
            prev_positions = (input_ids[:i] == current_token).nonzero(as_tuple=True)[0]
            
            if len(prev_positions) > 0:
                current_hidden = hidden[i]
                similarities = []
                for prev_pos in prev_positions:
                    prev_hidden = hidden[prev_pos]
                    sim = torch.nn.functional.cosine_similarity(
                        current_hidden.unsqueeze(0),
                        prev_hidden.unsqueeze(0)
                    )
                    similarities.append(sim.item())
                scores[i] = max(similarities) if similarities else 0.0
        
        layer_induction_scores[layer_idx] = scores.cpu().numpy()
    
    return layer_induction_scores

# Analyze a sample sequence across layers
sample_sequence = "A B C D A B C D"
layer_scores = analyze_layer_wise_induction(
    mamba_model, 
    mamba_tokenizer, 
    sample_sequence
)

# Visualize layer-wise induction
plt.figure(figsize=(14, 6))
tokens = mamba_tokenizer(sample_sequence, return_tensors="pt")['input_ids'][0]
token_strs = [mamba_tokenizer.decode([t]) for t in tokens]

for layer_idx, scores in layer_scores.items():
    plt.plot(range(len(scores)), scores, 'o-', label=f'Layer {layer_idx}', 
             linewidth=4, markersize=6, alpha=0.7)

ax = plt.gca()
plt.xlabel('Token Position', fontsize=18)
plt.ylabel('Induction Score', fontsize=18)
plt.title('Induction Head Emergence Across Layers', fontsize=18, fontweight='bold')
legend = plt.legend(loc='best', fontsize=20, frameon=True)
for line in legend.get_lines():
    line.set_linewidth(4)
plt.xticks(range(len(token_strs)), 
           [f"{i}:{t.strip()}" for i, t in enumerate(token_strs)], 
           rotation=45, ha='right')
ax.tick_params(axis='both', labelsize=18)
for label in ax.get_xticklabels():
    label.set_fontsize(18)
for label in ax.get_yticklabels():
    label.set_fontsize(18)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{plots_dir}/layer_wise_induction_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved layer-wise induction plot to {plots_dir}/layer_wise_induction_{timestamp}.png")
print("\nUpdating analysis summary with induction results...")
print(f"Induction analysis complete! Summary will be updated in {summary_file}")

# ---------------------------------------------------------------
# Investigation: Why Strong Induction Scores but Poor Predictions?
# ---------------------------------------------------------------

print("\n" + "="*60)
print("=== INVESTIGATING INDUCTION-PREDICTION GAP ===")
print("="*60)

investigation_results = investigate_mamba_induction_mystery(
    mamba_model, 
    mamba_tokenizer, 
    timestamp, 
    plots_dir=plots_dir
)

# ---------------------------------------------------------------
# Save Analysis Summary
# ---------------------------------------------------------------

print("\nSaving analysis summary...")
with open(summary_file, 'w') as f:
    f.write(f"Induction Head Analysis Summary - {timestamp}\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Device used: {device}\n")
    f.write(f"Mamba model: {mamba_model_name}\n\n")
    
    # Add Induction Head Analysis Results
    f.write("Induction Head Analysis Results:\n")
    f.write("-" * 40 + "\n")
    
    f.write(f"Number of sequences tested: {len(induction_sequences)}\n")
    f.write(f"Test sequences: {induction_sequences}\n\n")
    
    # Summary statistics
    all_induction_scores = []
    for key, value in induction_results.items():
        if not key.endswith('_predictions'):
            all_induction_scores.extend(value['induction_scores'])
    
    all_induction_scores = np.array(all_induction_scores)
    
    f.write(f"Overall Statistics:\n")
    f.write(f"  Mean induction score: {all_induction_scores.mean():.4f}\n")
    f.write(f"  Std induction score: {all_induction_scores.std():.4f}\n")
    f.write(f"  Max induction score: {all_induction_scores.max():.4f}\n")
    f.write(f"  Min induction score: {all_induction_scores.min():.4f}\n")
    f.write(f"  Positions with strong induction (>0.5): {(all_induction_scores > 0.5).sum()}\n")
    f.write(f"  Total positions analyzed: {len(all_induction_scores)}\n\n")
    
    f.write(f"Prediction Accuracy:\n")
    f.write(f"  High induction positions: {pred_analysis['high_induction_accuracy']:.2%}\n")
    f.write(f"  Low induction positions: {pred_analysis['low_induction_accuracy']:.2%}\n")
    f.write(f"  High induction samples: {pred_analysis['num_high_induction']}\n")
    f.write(f"  Low induction samples: {pred_analysis['num_low_induction']}\n\n")
    
    # Per-sequence detailed results with graph values
    f.write("=" * 50 + "\n")
    f.write("Per-Sequence Detailed Results (Graph Values):\n")
    f.write("=" * 50 + "\n\n")
    
    for idx, sequence in enumerate(induction_sequences):
        seq_key = f"sequence_{idx}"
        if seq_key in induction_results:
            result = induction_results[seq_key]
            tokens = result['tokens']
            induction_scores = result['induction_scores']
            state_similarity = result['state_similarity']
            
            f.write(f"Sequence {idx+1}: {sequence}\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Tokens: {tokens}\n")
            f.write(f"  Sequence length: {len(tokens)}\n\n")
            
            # Induction scores per position (graph values)
            f.write(f"  Induction Scores per Position:\n")
            for pos, (token, score) in enumerate(zip(tokens, induction_scores)):
                f.write(f"    Position {pos:2d} ({token.strip():>10s}): {score:.4f}\n")
            
            f.write(f"\n  Sequence Statistics:\n")
            f.write(f"    Mean: {induction_scores.mean():.4f}\n")
            f.write(f"    Std:  {induction_scores.std():.4f}\n")
            f.write(f"    Max:  {induction_scores.max():.4f} (at position {np.argmax(induction_scores)})\n")
            f.write(f"    Min:  {induction_scores.min():.4f} (at position {np.argmin(induction_scores)})\n")
            
            # Strong induction positions
            strong_positions = np.where(induction_scores > 0.5)[0]
            if len(strong_positions) > 0:
                f.write(f"\n  Strong Induction Positions (>0.5):\n")
                for pos in strong_positions:
                    f.write(f"    Position {pos}: '{tokens[pos].strip()}' (score: {induction_scores[pos]:.4f})\n")
            
            # State similarity matrix summary (key values)
            f.write(f"\n  State Similarity Matrix (Key Values):\n")
            f.write(f"    Matrix shape: {state_similarity.shape}\n")
            f.write(f"    Mean similarity: {state_similarity.mean():.4f}\n")
            f.write(f"    Max similarity: {state_similarity.max():.4f}\n")
            f.write(f"    Min similarity: {state_similarity.min():.4f}\n")
            
            # Diagonal and off-diagonal statistics
            diagonal = np.diag(state_similarity)
            mask = ~np.eye(state_similarity.shape[0], dtype=bool)
            off_diagonal = state_similarity[mask]
            f.write(f"    Diagonal mean (self-similarity): {diagonal.mean():.4f}\n")
            f.write(f"    Off-diagonal mean: {off_diagonal.mean():.4f}\n")
            
            # Top similarity pairs
            if len(state_similarity) > 1:
                # Get top 5 similarity pairs (excluding diagonal)
                triu_indices = np.triu_indices(len(state_similarity), k=1)
                similarities_flat = state_similarity[triu_indices]
                top_indices = np.argsort(similarities_flat)[-5:][::-1]
                f.write(f"\n    Top 5 Similarity Pairs:\n")
                for i, idx in enumerate(top_indices):
                    row, col = triu_indices[0][idx], triu_indices[1][idx]
                    f.write(f"      {i+1}. Position {row} <-> Position {col}: {similarities_flat[idx]:.4f}\n")
                    f.write(f"         ('{tokens[row].strip()}' <-> '{tokens[col].strip()}')\n")
            
            f.write("\n")
    
    # Layer-wise induction analysis results
    if 'layer_scores' in locals():
        f.write("=" * 50 + "\n")
        f.write("Layer-wise Induction Analysis (Graph Values):\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Sample sequence: {sample_sequence}\n")
        f.write(f"Layers analyzed: {list(layer_scores.keys())}\n\n")
        
        for layer_idx, scores in layer_scores.items():
            f.write(f"Layer {layer_idx}:\n")
            f.write("-" * 30 + "\n")
            for pos, score in enumerate(scores):
                f.write(f"  Position {pos:2d}: {score:.4f}\n")
            f.write(f"  Mean: {scores.mean():.4f}\n")
            f.write(f"  Max:  {scores.max():.4f} (at position {np.argmax(scores)})\n")
            f.write(f"  Min:  {scores.min():.4f} (at position {np.argmin(scores)})\n\n")
    
    # Prediction details
    f.write("=" * 50 + "\n")
    f.write("Prediction Details:\n")
    f.write("=" * 50 + "\n\n")
    
    for idx, sequence in enumerate(induction_sequences):
        pred_key = f"sequence_{idx}_predictions"
        if pred_key in induction_results:
            predictions = induction_results[pred_key]
            f.write(f"Sequence {idx+1}: {sequence}\n")
            f.write("-" * 40 + "\n")
            
            for pred in predictions[:10]:  # Show first 10 predictions
                f.write(f"  Position {pred['position']}: '{pred['current_token']}' -> '{pred['actual_next']}'\n")
                f.write(f"    Induction score: {pred['induction_score']:.4f}\n")
                f.write(f"    Correct: {pred['correct']}\n")
                top_preds = pred['top_predictions'][:3]
                f.write(f"    Top 3 predictions: {[(tok, f'{prob:.3f}') for tok, prob in top_preds]}\n")
                f.write("\n")
            f.write("\n")
    
    # Add Investigation Results
    if 'investigation_results' in locals():
        f.write("=" * 50 + "\n")
        f.write("Induction-Prediction Gap Investigation Results:\n")
        f.write("=" * 50 + "\n\n")
        
        for seq_key, result in investigation_results.items():
            f.write(f"Sequence: {result['sequence']}\n")
            f.write("-" * 40 + "\n")
            f.write(f"High induction positions: {result['high_induction_positions']}\n\n")
            
            logit_analysis = result['logit_analysis']
            
            # High vs Low induction statistics
            high_ind = logit_analysis['high_induction']
            low_ind = logit_analysis['low_induction']
            
            if high_ind and low_ind:
                f.write("Logit Distribution Analysis:\n")
                f.write(f"  High Induction Positions:\n")
                f.write(f"    Mean entropy: {np.mean([p['entropy'] for p in high_ind]):.4f}\n")
                f.write(f"    Mean correct prob: {np.mean([p['correct_prob'] for p in high_ind]):.4f}\n")
                f.write(f"    Median rank: {np.median([p['rank'] for p in high_ind]):.0f}\n")
                high_top5 = sum(p['in_top_5'] for p in high_ind) / len(high_ind) * 100
                f.write(f"    Top-5 accuracy: {high_top5:.2f}%\n")
                f.write(f"    Sample count: {len(high_ind)}\n\n")
                
                f.write(f"  Low Induction Positions:\n")
                f.write(f"    Mean entropy: {np.mean([p['entropy'] for p in low_ind]):.4f}\n")
                f.write(f"    Mean correct prob: {np.mean([p['correct_prob'] for p in low_ind]):.4f}\n")
                f.write(f"    Median rank: {np.median([p['rank'] for p in low_ind]):.0f}\n")
                low_top5 = sum(p['in_top_5'] for p in low_ind) / len(low_ind) * 100
                f.write(f"    Top-5 accuracy: {low_top5:.2f}%\n")
                f.write(f"    Sample count: {len(low_ind)}\n\n")
            
            # First vs Second occurrence comparisons
            if result['comparisons']:
                f.write("First vs Second Occurrence Comparisons:\n")
                for i, comp in enumerate(result['comparisons']):
                    f.write(f"  Comparison {i+1}:\n")
                    f.write(f"    Expected token: '{comp['expected_token']}'\n")
                    f.write(f"    Hidden state similarity: {comp['hidden_similarity']:.4f}\n")
                    f.write(f"    Logit correlation: {comp['logit_correlation']:.4f}\n")
                    f.write(f"    Expected token rank at induction: {comp['expected_rank_at_second']}\n")
                    f.write(f"    Expected token prob at induction: {comp['expected_prob_at_second']:.4f}\n")
                    
                    if comp['hidden_similarity'] > 0.8 and abs(comp['logit_correlation']) < 0.5:
                        f.write(f"    ⚠️  KEY FINDING: High hidden similarity but low logit correlation!\n")
                        f.write(f"       → The LM head is NOT preserving induction information!\n")
                    f.write("\n")
            
            f.write("\n")

print(f"Analysis summary saved to {summary_file}")
print("\nAll plots complete!")
