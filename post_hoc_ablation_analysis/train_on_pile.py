"""
Train Mamba Model on The Pile Dataset

This script trains the Mamba-130M model on The Pile dataset,
then saves the trained model for evaluation on query datasets.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
import logging
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    MambaForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from pathlib import Path
import json
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PileDataset(IterableDataset):
    """
    Iterable dataset wrapper for The Pile.
    """
    
    def __init__(self, tokenizer, max_length=512, num_samples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = num_samples
        self.dataset = None
    
    def __iter__(self):
        """Iterate over The Pile dataset."""
        # Try alternative first (more reliable)
        dataset = None
        dataset_name = None
        
        # First try: monology/pile-uncopyrighted (more reliable)
        try:
            logger.info("📚 Trying to load: monology/pile-uncopyrighted")
            dataset = load_dataset(
                "monology/pile-uncopyrighted",
                split="train",
                streaming=True,
                trust_remote_code=True
            )
            dataset_name = "monology/pile-uncopyrighted"
            logger.info("✅ Loaded monology/pile-uncopyrighted dataset")
        except ModuleNotFoundError as e:
            if 'zstandard' in str(e):
                logger.error("❌ Missing dependency: zstandard")
                logger.error("   Install it with: pip install zstandard")
                raise
            raise
        except Exception as e:
            logger.warning(f"⚠️ Could not load monology/pile-uncopyrighted: {e}")
            logger.info("   Trying alternative: EleutherAI/pile")
            
            # Second try: EleutherAI/pile
            try:
                dataset = load_dataset(
                    "EleutherAI/pile",
                    split="train",
                    streaming=True,
                    trust_remote_code=True
                )
                dataset_name = "EleutherAI/pile"
                logger.info("✅ Loaded EleutherAI/pile dataset")
            except Exception as e2:
                logger.error(f"❌ Error loading EleutherAI/pile: {e2}")
                logger.error("   The Pile dataset servers may be unavailable.")
                logger.error("   Options:")
                logger.error("   1. Try again later (servers may be down)")
                logger.error("   2. Use a different dataset")
                logger.error("   3. Download The Pile locally and use local path")
                raise
        
        if dataset is None:
            raise RuntimeError("Failed to load any Pile dataset variant")
        
        count = 0
        for example in dataset:
            if self.num_samples and count >= self.num_samples:
                break
            
            text = example.get('text', '')
            if not text or len(text.strip()) < 50:
                continue
            
            # Tokenize
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            yield {
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0)
            }
            
            count += 1
            if count % 1000 == 0:
                logger.info(f"   Processed {count} samples...")


def load_model_for_training(model_name: str, device: str = "cuda", use_fp16: bool = True):
    """
    Load model for training.
    
    Args:
        model_name: Model name or path
        device: Device to load on
        use_fp16: Whether to use FP16 (for mixed precision training)
                  Note: Load in FP32, let Trainer handle FP16 casting
    """
    logger.info(f"📦 Loading model: {model_name}")
    
    # Load in FP32 for training (Trainer will handle FP16 casting)
    # This avoids "Attempting to unscale FP16 gradients" error
    try:
        model = MambaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Always FP32 for training
            device_map="auto" if device == "cuda" else None
        )
        logger.info("✅ Loaded with MambaForCausalLM (FP32)")
    except Exception as e:
        logger.warning(f"⚠️ MambaForCausalLM failed: {e}")
        logger.info("   Falling back to AutoModelForCausalLM")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Always FP32 for training
            device_map="auto" if device == "cuda" else None
        )
        logger.info("✅ Loaded with AutoModelForCausalLM (FP32)")
    
    if device == "cpu":
        model = model.to(device)
    
    return model


def train_on_pile(
    model_name: str = "state-spaces/mamba-130m-hf",
    output_dir: str = "./models/mamba_trained_on_pile",
    num_train_samples: int = 10000,
    max_length: int = 512,
    batch_size: int = 4,
    num_epochs: int = 1,
    learning_rate: float = 5e-5,
    device: str = "cuda",
    save_steps: int = 500,
    logging_steps: int = 100
):
    """
    Train Mamba model on The Pile dataset.
    
    Args:
        model_name: Pretrained model to start from
        output_dir: Directory to save trained model
        num_train_samples: Number of training samples
        max_length: Maximum sequence length
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        save_steps: Save checkpoint every N steps
        logging_steps: Log every N steps
    """
    logger.info("="*80)
    logger.info("🚀 TRAINING MAMBA ON THE PILE DATASET")
    logger.info("="*80)
    
    # Load tokenizer
    logger.info("\n📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("✅ Tokenizer loaded")
    
    # Load model
    logger.info("\n📦 Loading model...")
    # Load in FP32 - Trainer will handle FP16 mixed precision
    model = load_model_for_training(model_name, device, use_fp16=(device == "cuda"))
    logger.info("✅ Model loaded")
    
    # Create dataset
    logger.info("\n📚 Creating Pile dataset...")
    train_dataset = PileDataset(
        tokenizer=tokenizer,
        max_length=max_length,
        num_samples=num_train_samples
    )
    logger.info(f"✅ Dataset created (max {num_train_samples} samples)")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )
    
    # Calculate max_steps for IterableDataset
    # Since IterableDataset doesn't have __len__, we need to specify max_steps
    max_steps = (num_train_samples // batch_size) * num_epochs if num_train_samples else None
    
    # Calculate warmup steps (10% of max_steps or 100, whichever is smaller)
    warmup_steps_value = min(100, max_steps // 10) if max_steps else 100
    
    # Training arguments
    # When using max_steps with IterableDataset, we don't set num_train_epochs
    # Build args dict conditionally to avoid None comparison issues
    training_args_dict = {
        'output_dir': output_dir,
        'max_steps': max_steps,  # Required for IterableDataset
        'per_device_train_batch_size': batch_size,
        'learning_rate': learning_rate,
        'warmup_steps': warmup_steps_value,
        'logging_steps': logging_steps,
        'save_steps': save_steps,
        'save_total_limit': 3,
        'prediction_loss_only': True,
        'remove_unused_columns': False,
        'dataloader_pin_memory': True,
        'fp16': device == "cuda",
        'report_to': "none",  # Disable wandb/tensorboard
    }
    
    # Only set num_train_epochs if max_steps is not set (for regular datasets)
    if max_steps is None:
        training_args_dict['num_train_epochs'] = num_epochs
    
    training_args = TrainingArguments(**training_args_dict)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("\n" + "="*80)
    logger.info("🏋️ STARTING TRAINING")
    logger.info("="*80)
    logger.info(f"   Samples: {num_train_samples}")
    logger.info(f"   Epochs: {num_epochs}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Max steps: {max_steps}")
    logger.info(f"   Learning rate: {learning_rate}")
    logger.info(f"   Max length: {max_length}")
    logger.info("="*80)
    
    trainer.train()
    
    # Save model
    logger.info("\n💾 Saving trained model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    logger.info(f"✅ Model saved to: {output_dir}")
    
    # Save training info
    training_info = {
        'model_name': model_name,
        'output_dir': output_dir,
        'num_train_samples': num_train_samples,
        'max_length': max_length,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'dataset': 'EleutherAI/pile'
    }
    
    info_path = Path(output_dir) / "training_info.json"
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    
    logger.info(f"✅ Training info saved to: {info_path}")
    logger.info("="*80)
    
    return output_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Mamba on The Pile")
    parser.add_argument("--model_name", type=str, default="state-spaces/mamba-130m-hf",
                       help="Pretrained model name")
    parser.add_argument("--output_dir", type=str, default="./models/mamba_trained_on_pile",
                       help="Output directory for trained model")
    parser.add_argument("--num_samples", type=int, default=10000,
                       help="Number of training samples")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    train_on_pile(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_train_samples=args.num_samples,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device
    )

