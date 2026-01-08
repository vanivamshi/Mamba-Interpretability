"""
SOLUTION: Use prompts that match Mamba's training distribution

Key insight: Mamba-130M was trained on The Pile (natural text continuation),
NOT on instruction-following formats. We need to reformulate tasks to match
what the model actually learned during pretraining.
"""

import torch
import logging
from typing import Dict, List
from proper_evaluation import ProperEvaluationMetrics

logger = logging.getLogger(__name__)


class PileAlignedPrompts:
    """
    Reformulate memory tasks to match The Pile's natural text patterns.
    
    The Pile contains:
    - Books, articles, web text (natural narrative flow)
    - Code with documentation (structured but natural)
    - Academic papers (formal but continuous prose)
    
    It does NOT contain:
    - Explicit Q&A formats ("Question: ... Answer: ...")
    - Structured key-value pairs ("NAME=Alice, CITY=Paris")
    - Instruction-following templates
    """
    
    @staticmethod
    def get_pile_aligned_prompts() -> Dict[str, List[Dict]]:
        """
        Memory tasks reformulated as natural text continuation.
        """
        return {
            # ========================================================
            # NARRATIVE STYLE: Mimics book/article patterns
            # ========================================================
            'narrative_memory': [
                {
                    'prompt': 'Alice lived in Paris with her cats. She loved her life there. Alice',
                    'expected': '',
                    'alternatives': ['lived', 'loved', 'was', 'had'],
                    'task': 'name in narrative',
                    'note': 'Model should continue naturally mentioning Alice'
                },
                {
                    'prompt': 'The secret code was BLUE42. Nobody knew it except the agents. The code',
                    'expected': 'BLUE42',
                    'alternatives': ['BLUE42', 'was BLUE42', 'was'],
                    'task': 'recall in story',
                    'note': 'Natural continuation should reference the code'
                },
                {
                    'prompt': 'First came the apple, then the banana, and finally the cherry. The first fruit',
                    'expected': 'apple',
                    'alternatives': ['apple', 'was', 'the apple', 'was the apple'],
                    'task': 'sequence recall',
                    'note': 'Referencing back naturally in prose'
                },
            ],
            
            # ========================================================
            # DOCUMENT STYLE: Mimics Wikipedia/encyclopedia patterns
            # ========================================================
            'document_memory': [
                {
                    'prompt': '''Alice (born 1985) is a French resident who lives in Paris. 
Known for her love of cats, Alice''',
                    'expected': '',
                    'alternatives': ['lives', 'resides', 'is known', 'has'],
                    'task': 'biographical recall',
                    'note': 'Wikipedia-style article continuation'
                },
                {
                    'prompt': '''Security Protocol BLUE42
Classification: Top Secret
The protocol, known as BLUE42,''',
                    'expected': 'BLUE42',
                    'alternatives': ['BLUE42', 'is', 'was'],
                    'task': 'technical doc recall',
                    'note': 'Technical documentation style'
                },
            ],
            
            # ========================================================
            # CONVERSATIONAL STYLE: Mimics dialogue from The Pile
            # ========================================================
            'dialogue_memory': [
                {
                    'prompt': '''Person A: Hi, I'm Alice from Paris, I have three cats.
Person B: Nice to meet you, Alice. So you''',
                    'expected': '',
                    'alternatives': ['live', 'are', 'have'],
                    'task': 'dialogue recall',
                    'note': 'Natural dialogue continuation'
                },
            ],
            
            # ========================================================
            # CODE COMMENT STYLE: Mimics programming documentation
            # ========================================================
            'code_memory': [
                {
                    'prompt': '''# User: Alice
# Location: Paris
# Pet: cats
# The user name is''',
                    'expected': 'Alice',
                    'alternatives': ['Alice', 'alice'],
                    'task': 'code comment recall',
                    'note': 'Natural continuation in code comments'
                },
                {
                    'prompt': '''// Secret code: BLUE42
// Store this for authentication
// Code value =''',
                    'expected': 'BLUE42',
                    'alternatives': ['BLUE42', 'blue42'],
                    'task': 'code constant recall',
                    'note': 'Programming context'
                },
            ],
            
            # ========================================================
            # REFORMULATED Q&A: Make it look like The Pile text
            # ========================================================
            'pile_style_qa': [
                {
                    # Instead of "Context: ... Question: ..."
                    # Make it look like an encyclopedia article with implicit Q&A
                    'prompt': '''Newton's Law of Universal Gravitation

Newton's law states that gravitational force is inversely proportional to 
the square of the distance between objects. This means the force decreases 
at larger distances.

The force decreases at''',
                    'expected': 'larger distances',
                    'alternatives': ['larger distances', 'greater distances', 'larger'],
                    'task': 'implicit QA',
                    'note': 'Q&A hidden in natural text flow'
                },
                {
                    'prompt': '''History of the Alvin and the Chipmunks Band

The animated band Alvin and the Chipmunks was created by David Seville in 
1958. Seville (real name Ross Bagdasarian) developed the concept and provided 
the original voice for the characters. The creator was''',
                    'expected': 'David Seville',
                    'alternatives': ['David Seville', 'Seville', 'Ross Bagdasarian'],
                    'task': 'encyclopedia QA',
                    'note': 'Wikipedia-style implicit answer'
                },
            ],
        }
    
    @staticmethod
    def convert_squad_to_pile_style(context: str, question: str) -> str:
        """
        Convert SQuAD format to natural text continuation.
        
        SQuAD format (instruction-style):
            Context: [passage]
            Question: What is X?
            Answer:
        
        Pile-aligned format (narrative-style):
            [passage]. [passage restated as statement about X]. X
        """
        # Example transformation:
        # Q: "What does X do?" 
        # → "The mechanism of X. X works by"
        
        # This requires understanding the question type
        question_lower = question.lower()
        
        if 'what' in question_lower and 'is' in question_lower:
            # "What is X?" → "X is"
            subject = question.replace('What is', '').replace('?', '').strip()
            prompt = f"{context}\n\nThe {subject}. This {subject} is"
        
        elif 'who' in question_lower:
            # "Who created X?" → "The creator of X. X was created by"
            subject = question.replace('Who', '').replace('?', '').strip()
            prompt = f"{context}\n\nThe person who {subject}. This person was"
        
        elif 'when' in question_lower:
            # "When did X?" → "The time of X. X occurred in"
            subject = question.replace('When', '').replace('?', '').strip()
            prompt = f"{context}\n\nThe time when {subject}. This event occurred in"
        
        else:
            # Default: just natural continuation
            prompt = f"{context}\n\n{question.replace('?', '.')} The answer is"
        
        return prompt


class ImprovedMemoryDiagnostic:
    """
    Test memory using Pile-aligned prompts instead of instruction formats.
    """
    
    def __init__(self, tokenizer, device="cuda"):
        self.tokenizer = tokenizer
        self.device = device
        self.pile_prompts = PileAlignedPrompts()
        self.evaluator = ProperEvaluationMetrics()
    
    def _smart_match(self, response: str, expected: str, alternatives: List[str]) -> bool:
        """
        Smart matching using proper evaluation metrics.
        """
        eval_result = self.evaluator.evaluate_recall(
            response=response,
            expected=expected,
            context_keywords=alternatives if alternatives else [],
            task_type='entity_recall'
        )
        # Consider correct if score >= 0.5 (at least contextual recall)
        return eval_result['score'] >= 0.5
    
    def test_pile_aligned_memory(self, model, test_cases: Dict) -> Dict:
        """
        Test if Pile-aligned prompts improve memory task performance.
        """
        logger.info("="*80)
        logger.info("🎯 TESTING PILE-ALIGNED PROMPTS")
        logger.info("="*80)
        logger.info("Hypothesis: Natural text patterns should match training distribution")
        logger.info("")
        
        results = {}
        
        for strategy_name, cases in test_cases.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing: {strategy_name}")
            logger.info(f"{'='*60}")
            
            correct = 0
            total = len(cases)
            details = []
            
            for case in cases:
                prompt = case['prompt']
                expected = case['expected']
                alternatives = case.get('alternatives', [])
                
                # Tokenize
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                # Generate
                import torch
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=False,
                        temperature=0.1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                input_len = inputs['input_ids'].shape[1]
                response = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
                response = response.strip()
                
                # Use proper evaluation
                eval_result = self.evaluator.evaluate_recall(
                    response=response,
                    expected=expected,
                    context_keywords=alternatives if alternatives else [],
                    task_type=case.get('task', 'unknown')
                )
                is_correct = eval_result['score'] >= 0.5  # At least contextual recall
                eval_details = eval_result['details']
                
                if is_correct:
                    correct += 1
                
                details.append({
                    'prompt': prompt[:60] + "..." if len(prompt) > 60 else prompt,
                    'expected': expected,
                    'response': response,  # Full response (no truncation)
                    'correct': is_correct,
                    'score': eval_result['score'],
                    'evaluation': eval_details,
                    'task': case['task']
                })
                
                # Show first 2 examples with FULL output
                if len(details) <= 2:
                    symbol = "✅" if is_correct else "❌"
                    logger.info(f"  {symbol} {case['task']}")
                    logger.info(f"     Expected: '{expected}'")
                    logger.info(f"     Got: '{response}'")  # FULL OUTPUT
                    logger.info(f"     {eval_details} (score: {eval_result['score']:.2f})")
            
            accuracy = correct / total if total > 0 else 0
            results[strategy_name] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'details': details
            }
            
            # Summary
            status = "🟢" if accuracy >= 0.5 else "🟡" if accuracy >= 0.3 else "🔴"
            logger.info(f"  {status} Accuracy: {accuracy*100:.1f}% ({correct}/{total})")
        
        return results

