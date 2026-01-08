"""
Query Dataset Loader for Testing Trained Models

Loads query datasets for evaluation after training on The Pile.
Supports various query datasets like SQuAD, Natural Questions, etc.
"""

import torch
import logging
from typing import List, Dict, Optional
from datasets import load_dataset
import random
from pile_aligned_prompts import PileAlignedPrompts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryDatasetLoader:
    """
    Load query datasets for evaluation.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
    
    def load_squad_queries(self, num_samples: int = 50) -> List[Dict]:
        """
        Load SQuAD dataset queries.
        
        Args:
            num_samples: Number of samples to load
        
        Returns:
            List of query dictionaries
        """
        try:
            logger.info(f"📖 Loading SQuAD queries ({num_samples} samples)...")
            dataset = load_dataset("squad", split="validation")
            
            samples = random.sample(list(dataset), min(num_samples, len(dataset)))
            
            queries = []
            for sample in samples:
                context = sample['context']
                question = sample['question']
                answers = sample['answers']['text']
                expected = answers[0] if answers else ""
                
                # Original instruction-style format
                prompt_instruction = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
                
                # Pile-aligned format (natural text continuation)
                pile_prompts = PileAlignedPrompts()
                prompt_pile = pile_prompts.convert_squad_to_pile_style(context, question)
                
                queries.append({
                    'prompt': prompt_instruction,  # Keep original for comparison
                    'prompt_pile_style': prompt_pile,  # Add Pile-aligned version
                    'expected': expected,
                    'alternatives': answers[:3],  # Top 3 answers
                    'task': 'squad_qa',
                    'source': 'squad',
                    'context': context[:200]  # Store context for reference
                })
            
            logger.info(f"✅ Loaded {len(queries)} SQuAD queries")
            return queries
            
        except Exception as e:
            logger.error(f"❌ Error loading SQuAD: {e}")
            return []
    
    def load_natural_questions(self, num_samples: int = 50) -> List[Dict]:
        """
        Load Natural Questions dataset queries.
        
        Args:
            num_samples: Number of samples to load
        
        Returns:
            List of query dictionaries
        """
        try:
            logger.info(f"📖 Loading Natural Questions ({num_samples} samples)...")
            # Natural Questions is large, use streaming
            dataset = load_dataset("natural_questions", split="validation", streaming=True)
            
            queries = []
            count = 0
            
            for example in dataset:
                if count >= num_samples:
                    break
                
                question = example.get('question', {}).get('text', '')
                if not question:
                    continue
                
                # Get first answer
                annotations = example.get('annotations', [])
                if not annotations:
                    continue
                
                short_answers = annotations[0].get('short_answers', [])
                if not short_answers:
                    continue
                
                # Get context
                document = example.get('document', {})
                context_tokens = document.get('tokens', [])
                context = ' '.join([t.get('token', '') for t in context_tokens[:200]])
                
                # Get answer from context
                answer_start = short_answers[0].get('start_token', 0)
                answer_end = short_answers[0].get('end_token', 0)
                expected = ' '.join([t.get('token', '') for t in context_tokens[answer_start:answer_end]])
                
                if expected and len(expected) > 2:
                    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
                    
                    queries.append({
                        'prompt': prompt,
                        'expected': expected,
                        'alternatives': [],
                        'task': 'natural_questions',
                        'source': 'natural_questions',
                        'context': context[:200]
                    })
                    count += 1
            
            logger.info(f"✅ Loaded {len(queries)} Natural Questions queries")
            return queries
            
        except Exception as e:
            logger.error(f"❌ Error loading Natural Questions: {e}")
            return []
    
    def load_triviaqa(self, num_samples: int = 50) -> List[Dict]:
        """
        Load TriviaQA dataset queries.
        
        Args:
            num_samples: Number of samples to load
        
        Returns:
            List of query dictionaries
        """
        try:
            logger.info(f"📖 Loading TriviaQA ({num_samples} samples)...")
            # Use streaming to avoid downloading entire dataset
            dataset = load_dataset("trivia_qa", "rc", split="validation", streaming=True)
            
            queries = []
            count = 0
            
            for sample in dataset:
                if count >= num_samples:
                    break
                
                try:
                    question = sample.get('question', '')
                    if not question:
                        continue
                    
                    answer_data = sample.get('answer', {})
                    if not answer_data:
                        continue
                    
                    answer = answer_data.get('value', '')
                    if not answer:
                        continue
                    
                    # Get context from search results
                    search_results = sample.get('search_results', {})
                    search_context = search_results.get('search_context', [])
                    context = ' '.join(search_context[:2]) if search_context else ''
                    
                    # Get answer aliases
                    aliases = answer_data.get('aliases', [])[:3]
                    
                    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
                    
                    queries.append({
                        'prompt': prompt,
                        'expected': answer,
                        'alternatives': aliases,
                        'task': 'triviaqa',
                        'source': 'triviaqa',
                        'context': context[:200]
                    })
                    count += 1
                except Exception as e:
                    logger.warning(f"⚠️ Skipping TriviaQA sample due to error: {e}")
                    continue
            
            logger.info(f"✅ Loaded {len(queries)} TriviaQA queries")
            return queries
            
        except Exception as e:
            logger.error(f"❌ Error loading TriviaQA: {e}")
            return []
    
    def load_custom_queries(self, queries_file: str) -> List[Dict]:
        """
        Load custom queries from a JSON file.
        
        Args:
            queries_file: Path to JSON file with queries
        
        Returns:
            List of query dictionaries
        """
        import json
        from pathlib import Path
        
        try:
            logger.info(f"📖 Loading custom queries from {queries_file}...")
            path = Path(queries_file)
            
            if not path.exists():
                logger.error(f"❌ File not found: {queries_file}")
                return []
            
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Support multiple formats
            if isinstance(data, list):
                queries = data
            elif isinstance(data, dict) and 'queries' in data:
                queries = data['queries']
            else:
                logger.error("❌ Invalid JSON format")
                return []
            
            logger.info(f"✅ Loaded {len(queries)} custom queries")
            return queries
            
        except Exception as e:
            logger.error(f"❌ Error loading custom queries: {e}")
            return []
    
    def get_query_tasks(
        self,
        datasets: List[str] = ['squad'],
        num_per_dataset: int = 30
    ) -> Dict[str, List[Dict]]:
        """
        Get query tasks from specified datasets.
        
        Args:
            datasets: List of dataset names ('squad', 'natural_questions', 'triviaqa')
            num_per_dataset: Number of queries per dataset
        
        Returns:
            Dictionary of dataset names to query lists
        """
        all_queries = {}
        
        if 'squad' in datasets:
            squad_queries = self.load_squad_queries(num_per_dataset)
            if squad_queries:
                all_queries['squad_queries'] = squad_queries
        
        if 'natural_questions' in datasets:
            nq_queries = self.load_natural_questions(num_per_dataset)
            if nq_queries:
                all_queries['natural_questions_queries'] = nq_queries
        
        if 'triviaqa' in datasets:
            trivia_queries = self.load_triviaqa(num_per_dataset)
            if trivia_queries:
                all_queries['triviaqa_queries'] = trivia_queries
        
        logger.info(f"✅ Total queries loaded: {sum(len(v) for v in all_queries.values())}")
        
        return all_queries


def get_query_tasks_for_evaluation(
    datasets: List[str] = ['squad'],
    num_per_dataset: int = 30
) -> Dict[str, List[Dict]]:
    """
    Convenience function to get query tasks for evaluation.
    
    Args:
        datasets: List of dataset names
        num_per_dataset: Number of queries per dataset
    
    Returns:
        Dictionary of query tasks
    """
    loader = QueryDatasetLoader()
    return loader.get_query_tasks(datasets=datasets, num_per_dataset=num_per_dataset)

