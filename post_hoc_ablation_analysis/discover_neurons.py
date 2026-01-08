#!/usr/bin/env python3
"""
Step 1: Neuron Discovery Script
Discovers beneficial neurons by testing all neurons individually.
Saves results to JSON file for use in validation phase.
"""

import argparse
import sys
from pathlib import Path
import torch
from ablation_3 import (
    SteeringValidator,
    StructuredTaskGenerator,
    save_discovered_neurons,
    logger
)
from mamba_model_loader import load_mamba_model_and_tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Discover beneficial neurons for steering (Stage 1)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="state-spaces/mamba-130m-hf",
        help="Model to evaluate (default: state-spaces/mamba-130m-hf)"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save discovered neurons JSON file (e.g., discovered_neurons.json)"
    )
    parser.add_argument(
        "--validation_size",
        type=int,
        default=200,
        help="Number of validation examples per task (default: 200)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ablation_3_results",
        help="Directory to save results (default: ablation_3_results)"
    )
    
    args = parser.parse_args()
    
    # Auto-correct model name: add state-spaces/ prefix and -hf suffix if needed
    model_name = args.model
    if 'mamba' in model_name.lower():
        # Add state-spaces/ prefix if not present
        if not model_name.startswith('state-spaces/'):
            model_name = f"state-spaces/{model_name}"
        # Add -hf suffix if not present
        if not model_name.endswith('-hf'):
            model_name = model_name + '-hf'
        if model_name != args.model:
            logger.info(f"Auto-correcting model name: {args.model} → {model_name}")
    
    # Load model and tokenizer
    logger.info(f"Loading model: {model_name}")
    try:
        model, tokenizer = load_mamba_model_and_tokenizer(
            model_name=model_name,
            device=args.device if torch.cuda.is_available() else "cpu",
            use_mamba_class=True,
            fallback_to_auto=True
        )
    except Exception as e:
        if "tiktoken" in str(e).lower():
            logger.error(f"Error: Model requires tiktoken package. Either:")
            logger.error(f"  1. Install tiktoken: pip install tiktoken")
            logger.error(f"  2. Use the -hf version: --model {args.model}-hf")
            sys.exit(1)
        else:
            raise
    
    device = next(model.parameters()).device
    
    # Set padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    # Generate validation tasks
    logger.info("Generating validation tasks...")
    generator = StructuredTaskGenerator(seed=42)
    validation_tasks = generator.generate_validation_set(
        size_per_task=args.validation_size // 4
    )
    
    # Run discovery
    logger.info("\n" + "="*80)
    logger.info("NEURON DISCOVERY (Stage 1)")
    logger.info("="*80)
    
    validator = SteeringValidator(model, tokenizer, device)
    discovery_results, beneficial_neurons = validator.run_comprehensive_neuron_discovery(validation_tasks)
    
    # Save discovered neurons
    # If save_path is relative, save to output_dir
    save_path = Path(args.save_path)
    if not save_path.is_absolute():
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        save_path = output_dir / save_path.name
    
    save_path.parent.mkdir(exist_ok=True, parents=True)
    save_discovered_neurons(discovery_results, beneficial_neurons, save_path)
    
    logger.info("\n" + "="*80)
    logger.info("DISCOVERY COMPLETE")
    logger.info("="*80)
    logger.info(f"✅ Discovered {len(beneficial_neurons)} beneficial neurons")
    logger.info(f"💾 Saved to: {save_path}")
    logger.info(f"\nNext step: Run validation with:")
    logger.info(f"  python steering_validation.py --model {args.model} --neurons {save_path}")


if __name__ == "__main__":
    main()

