"""
Experimental Framework for Mamba Mechanistic Interpretability

This module implements the step-by-step experimental recipe for opening Mamba's black box,
following the comprehensive framework outlined in the research methodology.
"""

import torch
import torch.nn as nn
import numpy as np
import random
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import wandb
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Configuration for mechanistic interpretability experiments."""
    # Model parameters
    model_name: str = "state-spaces/mamba-130m-hf"
    hidden_size: int = 768
    num_layers: int = 24
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_length: int = 512
    num_epochs: int = 10
    
    # Analysis parameters
    layer_idx: int = 0
    top_k: int = 10
    num_samples: int = 50
    
    # SAE parameters
    sae_latent_dim: float = 0.3  # Fraction of hidden size
    sae_l1_weight: float = 1e-3
    sae_sparsity_target: float = 0.05
    
    # Reproducibility
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging
    use_wandb: bool = False
    log_dir: str = "experiment_logs"
    save_checkpoints: bool = True
    
    def __post_init__(self):
        """Post-initialization setup."""
        os.makedirs(self.log_dir, exist_ok=True)
        if self.use_wandb:
            wandb.init(project="mamba-mechanistic-interpretability", config=asdict(self))

class DeterministicSetup:
    """Handles deterministic seeding and reproducibility."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.setup_seeds()
    
    def setup_seeds(self):
        """Set all random seeds for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        logger.info(f"Set deterministic seeds to {self.seed}")
    
    def get_seed_info(self) -> Dict[str, Any]:
        """Get information about current seed setup."""
        return {
            "seed": self.seed,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }

class ActivationHook:
    """Hook for collecting activations during forward passes."""
    
    def __init__(self, layer_idx: int, hook_type: str = "forward"):
        self.layer_idx = layer_idx
        self.hook_type = hook_type
        self.activations = []
        self.hook_handle = None
    
    def register_hook(self, model, target_layer):
        """Register the hook on the target layer."""
        if self.hook_type == "forward":
            self.hook_handle = target_layer.register_forward_hook(self._forward_hook)
        elif self.hook_type == "backward":
            self.hook_handle = target_layer.register_backward_hook(self._backward_hook)
        
        logger.info(f"Registered {self.hook_type} hook on layer {self.layer_idx}")
    
    def _forward_hook(self, module, input, output):
        """Forward hook to capture activations."""
        if isinstance(output, tuple):
            # Take the first element (usually hidden states)
            activation = output[0].detach().clone()
        else:
            activation = output.detach().clone()
        
        self.activations.append(activation)
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Backward hook to capture gradients."""
        if grad_output and grad_output[0] is not None:
            gradient = grad_output[0].detach().clone()
            self.activations.append(gradient)
    
    def remove_hook(self):
        """Remove the registered hook."""
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None
            logger.info(f"Removed hook from layer {self.layer_idx}")
    
    def get_activations(self) -> List[torch.Tensor]:
        """Get collected activations."""
        return self.activations
    
    def clear_activations(self):
        """Clear stored activations."""
        self.activations = []

class ActivationCollector:
    """Collects and manages activations from multiple layers."""
    
    def __init__(self, model, config: ExperimentConfig):
        self.model = model
        self.config = config
        self.hooks = {}
        self.activation_data = {}
        
    def register_hooks(self, layer_indices: List[int]):
        """Register hooks for specified layers."""
        from utils import get_model_layers
        
        layers = get_model_layers(self.model)
        if layers is None:
            logger.error("Could not find model layers")
            return False
        
        for layer_idx in layer_indices:
            if layer_idx < len(layers):
                hook = ActivationHook(layer_idx)
                hook.register_hook(self.model, layers[layer_idx])
                self.hooks[layer_idx] = hook
                logger.info(f"Registered hook for layer {layer_idx}")
            else:
                logger.warning(f"Layer index {layer_idx} out of range")
        
        return True
    
    def collect_activations(self, inputs: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Collect activations for given inputs."""
        # Clear previous activations
        for hook in self.hooks.values():
            hook.clear_activations()
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(inputs)
        
        # Collect activations
        activations = {}
        for layer_idx, hook in self.hooks.items():
            hook_activations = hook.get_activations()
            if hook_activations:
                # Concatenate all activations from this layer
                activations[layer_idx] = torch.cat(hook_activations, dim=0)
                logger.info(f"Collected activations for layer {layer_idx}: {activations[layer_idx].shape}")
        
        return activations
    
    def remove_all_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks.values():
            hook.remove_hook()
        self.hooks.clear()
        logger.info("Removed all activation hooks")

class ExperimentLogger:
    """Handles logging and saving of experimental results."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create timestamped directory for this experiment
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.log_dir / f"experiment_{timestamp}"
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Setup file logging
        log_file = self.experiment_dir / "experiment.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Experiment logging initialized in {self.experiment_dir}")
    
    def save_config(self, config: ExperimentConfig):
        """Save experiment configuration."""
        config_file = self.experiment_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        logger.info(f"Saved configuration to {config_file}")
    
    def save_activations(self, activations: Dict[int, torch.Tensor], filename: str = "activations.pt"):
        """Save activation data."""
        activation_file = self.experiment_dir / filename
        torch.save(activations, activation_file)
        logger.info(f"Saved activations to {activation_file}")
    
    def save_results(self, results: Dict[str, Any], filename: str = "results.json"):
        """Save experimental results."""
        results_file = self.experiment_dir / filename
        
        # Convert numpy arrays and tensors to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy().tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating, np.bool_, np.complexfloating)):
                return obj.item()  # Convert numpy scalars to Python native types
            elif isinstance(obj, np.number):
                return obj.item()  # Catch all numpy number types
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_for_json(v) for v in obj)
            else:
                return obj
        
        converted_results = convert_for_json(results)
        
        with open(results_file, 'w') as f:
            json.dump(converted_results, f, indent=2)
        logger.info(f"Saved results to {results_file}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics (to wandb if enabled, otherwise to file)."""
        if self.config.use_wandb:
            wandb.log(metrics, step=step)
        else:
            logger.info(f"Metrics: {metrics}")

class ToyDatasetGenerator:
    """Generates synthetic datasets for controlled experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.vocab_size = 1000  # Small vocabulary for toy tasks
    
    def generate_copy_task(self, num_samples: int = 1000, max_length: int = 20) -> List[str]:
        """Generate copy task examples."""
        examples = []
        for _ in range(num_samples):
            # Generate random sequence
            length = random.randint(5, max_length)
            sequence = [random.randint(1, self.vocab_size - 1) for _ in range(length)]
            
            # Create copy task: input + separator + target
            input_seq = " ".join(map(str, sequence))
            target_seq = " ".join(map(str, sequence))
            example = f"{input_seq} | {target_seq}"
            examples.append(example)
        
        return examples
    
    def generate_delayed_match(self, num_samples: int = 1000, delay: int = 5) -> List[str]:
        """Generate delayed match task examples."""
        examples = []
        for _ in range(num_samples):
            # Generate two random tokens
            token1 = random.randint(1, self.vocab_size - 1)
            token2 = random.randint(1, self.vocab_size - 1)
            
            # Create delayed match task
            filler = [random.randint(1, self.vocab_size - 1) for _ in range(delay)]
            example = f"{token1} {' '.join(map(str, filler))} {token2} {' '.join(map(str, filler))} {token1}"
            examples.append(example)
        
        return examples
    
    def generate_counting_task(self, num_samples: int = 1000) -> List[str]:
        """Generate counting task examples."""
        examples = []
        for _ in range(num_samples):
            # Generate random count
            count = random.randint(1, 10)
            tokens = [random.randint(1, self.vocab_size - 1) for _ in range(count)]
            
            # Create counting task
            example = f"{' '.join(map(str, tokens))} | count: {count}"
            examples.append(example)
        
        return examples
    
    def generate_bracket_nesting(self, num_samples: int = 1000, max_depth: int = 5) -> List[str]:
        """Generate bracket nesting task examples."""
        examples = []
        for _ in range(num_samples):
            depth = random.randint(1, max_depth)
            
            # Generate nested brackets
            brackets = ""
            for i in range(depth):
                brackets += "("
            for i in range(depth):
                brackets += ")"
            
            # Add some content
            content = " ".join([str(random.randint(1, self.vocab_size - 1)) for _ in range(depth)])
            example = f"{brackets} {content}"
            examples.append(example)
        
        return examples

def setup_experimental_environment(config: ExperimentConfig) -> Tuple[DeterministicSetup, ExperimentLogger]:
    """Set up the experimental environment with reproducibility and logging."""
    logger.info("Setting up experimental environment...")
    
    # Setup deterministic seeding
    deterministic_setup = DeterministicSetup(config.seed)
    
    # Setup logging
    experiment_logger = ExperimentLogger(config)
    
    # Save configuration
    experiment_logger.save_config(config)
    
    # Log seed information
    seed_info = deterministic_setup.get_seed_info()
    experiment_logger.log_metrics({"seed_info": seed_info})
    
    logger.info("Experimental environment setup complete")
    return deterministic_setup, experiment_logger

if __name__ == "__main__":
    # Example usage
    config = ExperimentConfig()
    deterministic_setup, logger = setup_experimental_environment(config)
    
    print("Experimental framework setup complete!")
    print(f"Experiment directory: {logger.experiment_dir}")
    print(f"Seed: {config.seed}")
    print(f"Device: {config.device}")
