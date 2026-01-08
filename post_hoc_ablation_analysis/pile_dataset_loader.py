"""
The Pile Dataset Loader for Steering Evaluation

Loads samples from The Pile dataset and creates evaluation tasks
compatible with the existing steering framework.
"""

import torch
import logging
from typing import List, Dict, Optional
from datasets import load_dataset
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PileDatasetLoader:
    """
    Load and process The Pile dataset for evaluation.
    """
    
    def __init__(self, num_samples: int = 100, seed: int = 42):
        """
        Initialize Pile dataset loader.
        
        Args:
            num_samples: Number of samples to load
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.seed = seed
        random.seed(seed)
        self.dataset = None
    
    def load_pile(self, split: str = "train", streaming: bool = True):
        """
        Load The Pile dataset from HuggingFace.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            streaming: Whether to use streaming (recommended for large datasets)
        """
        try:
            logger.info(f"📦 Loading The Pile dataset (split={split}, streaming={streaming})...")
            
            if streaming:
                # Use streaming for large datasets
                self.dataset = load_dataset(
                    "EleutherAI/pile",
                    split=split,
                    streaming=True
                )
                logger.info("✅ Loaded The Pile in streaming mode")
            else:
                # Load full dataset (may take time and memory)
                self.dataset = load_dataset(
                    "EleutherAI/pile",
                    split=split
                )
                logger.info(f"✅ Loaded The Pile dataset ({len(self.dataset)} samples)")
            
            return self.dataset
            
        except Exception as e:
            logger.error(f"❌ Error loading The Pile: {e}")
            logger.info("💡 Trying alternative: monology/pile-uncopyrighted")
            try:
                self.dataset = load_dataset(
                    "monology/pile-uncopyrighted",
                    split=split,
                    streaming=streaming
                )
                logger.info("✅ Loaded alternative Pile dataset")
                return self.dataset
            except Exception as e2:
                logger.error(f"❌ Error loading alternative: {e2}")
                raise
    
    def get_text_samples(self, num_samples: Optional[int] = None) -> List[str]:
        """
        Get text samples from The Pile.
        
        Args:
            num_samples: Number of samples to return (default: self.num_samples)
        
        Returns:
            List of text strings
        """
        if self.dataset is None:
            self.load_pile()
        
        num_samples = num_samples or self.num_samples
        samples = []
        
        logger.info(f"📖 Extracting {num_samples} text samples from The Pile...")
        
        if hasattr(self.dataset, '__iter__'):
            # Streaming dataset
            for i, example in enumerate(self.dataset):
                if i >= num_samples:
                    break
                
                text = example.get('text', '')
                if text and len(text.strip()) > 50:  # Filter very short texts
                    samples.append(text.strip())
            
            logger.info(f"✅ Extracted {len(samples)} samples")
        else:
            # Non-streaming dataset
            indices = random.sample(range(len(self.dataset)), min(num_samples, len(self.dataset)))
            for idx in indices:
                text = self.dataset[idx].get('text', '')
                if text and len(text.strip()) > 50:
                    samples.append(text.strip())
            
            logger.info(f"✅ Extracted {len(samples)} samples")
        
        return samples
    
    def create_language_modeling_tasks(
        self, 
        samples: List[str],
        context_length: int = 100,
        prediction_length: int = 20
    ) -> List[Dict]:
        """
        Create language modeling tasks from Pile samples.
        
        Args:
            samples: Text samples from The Pile
            context_length: Number of tokens to use as context
            prediction_length: Number of tokens to predict
        
        Returns:
            List of task dictionaries with 'prompt' and 'expected' fields
        """
        tasks = []
        
        logger.info(f"🔨 Creating language modeling tasks from {len(samples)} samples...")
        
        for sample in samples:
            # Simple approach: use first part as prompt, next part as expected
            words = sample.split()
            
            if len(words) < context_length + prediction_length:
                continue
            
            # Create prompt from first context_length words
            prompt_words = words[:context_length]
            prompt = ' '.join(prompt_words)
            
            # Expected continuation from next prediction_length words
            expected_words = words[context_length:context_length + prediction_length]
            expected = ' '.join(expected_words)
            
            tasks.append({
                'prompt': prompt,
                'expected': expected,
                'alternatives': [],  # For language modeling, we'll use perplexity
                'task': 'language_modeling',
                'source': 'pile'
            })
        
        logger.info(f"✅ Created {len(tasks)} language modeling tasks")
        return tasks
    
    def create_continuation_tasks(
        self,
        samples: List[str],
        num_tasks: int = 50
    ) -> List[Dict]:
        """
        Create text continuation tasks from Pile samples.
        
        Args:
            samples: Text samples from The Pile
            num_tasks: Number of tasks to create
        
        Returns:
            List of task dictionaries
        """
        tasks = []
        
        logger.info(f"🔨 Creating {num_tasks} continuation tasks...")
        
        for sample in random.sample(samples, min(num_tasks, len(samples))):
            sentences = sample.split('. ')
            
            if len(sentences) < 2:
                continue
            
            # Use first sentence(s) as prompt
            prompt = '. '.join(sentences[:-1]) + '.'
            
            # Next sentence as expected
            expected = sentences[-1].split()[0] if sentences[-1] else ""
            
            if expected and len(expected) > 2:
                tasks.append({
                    'prompt': prompt,
                    'expected': expected,
                    'alternatives': [],
                    'task': 'text_continuation',
                    'source': 'pile'
                })
        
        logger.info(f"✅ Created {len(tasks)} continuation tasks")
        return tasks
    
    def create_question_answer_tasks(
        self,
        samples: List[str],
        num_tasks: int = 30
    ) -> List[Dict]:
        """
        Create simple QA tasks from Pile samples.
        Uses first word/entity as answer.
        
        Args:
            samples: Text samples from The Pile
            num_tasks: Number of tasks to create
        
        Returns:
            List of task dictionaries
        """
        tasks = []
        
        logger.info(f"🔨 Creating {num_tasks} QA tasks...")
        
        for sample in random.sample(samples, min(num_tasks, len(samples))):
            words = sample.split()
            
            if len(words) < 10:
                continue
            
            # Simple approach: ask about first entity/word
            first_word = words[0].strip('.,!?;:')
            context = ' '.join(words[1:50])  # Use rest as context
            
            if first_word and len(first_word) > 2:
                prompt = f"Context: {context}\n\nQuestion: What is the first word mentioned?\nAnswer:"
                expected = first_word
                
                tasks.append({
                    'prompt': prompt,
                    'expected': expected,
                    'alternatives': [first_word.lower(), first_word.upper()],
                    'task': 'simple_qa',
                    'source': 'pile'
                })
        
        logger.info(f"✅ Created {len(tasks)} QA tasks")
        return tasks
    
    def get_evaluation_tasks(
        self,
        task_types: List[str] = ['continuation', 'qa'],
        num_per_type: int = 30
    ) -> Dict[str, List[Dict]]:
        """
        Get evaluation tasks from The Pile in the format expected by existing evaluators.
        
        Args:
            task_types: Types of tasks to create ('continuation', 'qa', 'lm')
            num_per_type: Number of tasks per type
        
        Returns:
            Dictionary of task categories with lists of tasks
        """
        if self.dataset is None:
            self.load_pile()
        
        samples = self.get_text_samples(num_samples=num_per_type * 3)
        
        all_tasks = {}
        
        if 'continuation' in task_types:
            all_tasks['pile_continuation'] = self.create_continuation_tasks(
                samples, num_per_type
            )
        
        if 'qa' in task_types:
            all_tasks['pile_qa'] = self.create_question_answer_tasks(
                samples, num_per_type
            )
        
        if 'lm' in task_types:
            all_tasks['pile_language_modeling'] = self.create_language_modeling_tasks(
                samples, context_length=50, prediction_length=10
            )[:num_per_type]
        
        logger.info(f"✅ Created {sum(len(v) for v in all_tasks.values())} total tasks from The Pile")
        
        return all_tasks


def get_pile_tasks_for_evaluation(num_samples: int = 50) -> Dict[str, List[Dict]]:
    """
    Convenience function to get Pile tasks for evaluation.
    
    Args:
        num_samples: Number of samples per task type
    
    Returns:
        Dictionary of task categories
    """
    loader = PileDatasetLoader(num_samples=num_samples * 2)
    return loader.get_evaluation_tasks(
        task_types=['continuation', 'qa'],
        num_per_type=num_samples
    )

