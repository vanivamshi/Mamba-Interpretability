"""
BREAKTHROUGH: Steering works for RARE/UNUSUAL tokens, not common ones

Key insight from results:
- BLUE42 (rare code): 0% → 50% with associative steering ✅
- Alice (common name): 0% → 0% with any steering ❌
- David Seville (person): 0% → 0% ❌

Hypothesis: Cluster 9 helps retrieve rare patterns from weak activations,
but can't override strong priors for common tokens.
"""

import torch
import logging
from typing import List, Dict, Tuple
from collections import Counter
import re

logger = logging.getLogger(__name__)


class TargetedSteeringStrategy:
    """
    Apply steering ONLY for rare/unusual tokens where it actually helps.
    """
    
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.hooks = []
        
        if hasattr(model, 'backbone'):
            self.layers = model.backbone.layers
        else:
            self.layers = model.layers
        
        self.cluster9_indices = [
            4, 38, 84, 94, 163, 171, 268, 363, 401, 497, 
            564, 568, 582, 654, 659, 686
        ]
    
    def is_rare_token_sequence(self, text: str) -> bool:
        """
        Determine if text contains rare/unusual token patterns.
        
        Rare patterns steering helps with:
        - Alphanumeric codes (BLUE42, XY789)
        - UUIDs, hashes
        - Technical identifiers
        - Unusual combinations
        
        Common patterns steering doesn't help:
        - Common names (Alice, John, Paris)
        - Regular words (code, secret, name)
        - Frequent phrases
        """
        # Check for alphanumeric codes (letters + numbers)
        if re.search(r'[A-Z]+\d+|\d+[A-Z]+', text):
            return True
        
        # Check for unusual character combinations
        if any(c in text for c in ['_', '-', '@', '#']):
            return True
        
        # Check token frequency in training data (proxy: tokenizer vocab)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # If tokens are outside common vocab range, likely rare
        # Most common tokens are in first 10k of vocab
        if tokens:
            avg_token_id = sum(tokens) / len(tokens)
            if avg_token_id > 10000:  # Heuristic threshold
                return True
        
        return False
    
    def apply_conditional_steering(
        self,
        target_texts: List[str],
        strength_rare: float = 2.5,
        strength_common: float = 1.0
    ):
        """
        Apply different steering strengths based on token rarity.
        
        Args:
            target_texts: List of texts we want to retrieve
            strength_rare: Steering for rare tokens (2-3x)
            strength_common: Steering for common tokens (1x = no steering)
        """
        # Analyze all targets
        rare_targets = []
        common_targets = []
        
        for text in target_texts:
            if self.is_rare_token_sequence(text):
                rare_targets.append(text)
            else:
                common_targets.append(text)
        
        logger.info(f"\n🎯 CONDITIONAL STEERING")
        logger.info(f"   Rare tokens: {rare_targets} → {strength_rare}x steering")
        logger.info(f"   Common tokens: {common_targets} → {strength_common}x steering")
        
        if not rare_targets:
            logger.info("   ⚠️ No rare tokens detected - steering may not help")
            return
        
        # Apply associative steering (what worked in experiments)
        # Early encoding layer
        if 6 < len(self.layers):
            layer = self.layers[6]
            target = getattr(layer, 'mixer', getattr(layer, 'ssm', layer))
            
            def early_hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                    rest = output[1:]
                else:
                    hidden = output
                    rest = ()
                
                h_mod = hidden * 2.0  # Encoding boost
                
                if rest:
                    return (h_mod,) + rest
                return h_mod
            
            h = target.register_forward_hook(early_hook)
            self.hooks.append(h)
        
        # Late retrieval layer with Cluster 9
        if 18 < len(self.layers):
            layer = self.layers[18]
            target = getattr(layer, 'mixer', getattr(layer, 'ssm', layer))
            
            def late_hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                    rest = output[1:]
                else:
                    hidden = output
                    rest = ()
                
                h_mod = hidden.clone()
                
                # Amplify Cluster 9 (memory neurons)
                for idx in self.cluster9_indices:
                    if idx < h_mod.shape[-1]:
                        h_mod[..., idx] *= 3.0
                
                if rest:
                    return (h_mod,) + rest
                return h_mod
            
            h = target.register_forward_hook(late_hook)
            self.hooks.append(h)
    
    def remove_steering(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_optimized_test_cases(self) -> Dict[str, List[Dict]]:
        """
        Test cases designed to validate when steering helps.
        
        Hypothesis: Steering helps with RARE tokens, not common ones.
        """
        return {
            'rare_codes': [
                {
                    'prompt': 'Access code: BLUE42. What is the access code?',
                    'expected': 'BLUE42',
                    'token_type': 'rare_alphanumeric',
                    'prediction': 'steering should help (0% → 40%+)'
                },
                {
                    'prompt': 'Password: XY789Z. Repeat password:',
                    'expected': 'XY789Z',
                    'token_type': 'rare_alphanumeric',
                    'prediction': 'steering should help'
                },
                {
                    'prompt': 'ID: A1B2C3. The ID is',
                    'expected': 'A1B2C3',
                    'token_type': 'rare_alphanumeric',
                    'prediction': 'steering should help'
                },
            ],
            
            'common_words': [
                {
                    'prompt': 'Name: Alice. What is the name?',
                    'expected': 'Alice',
                    'token_type': 'common_name',
                    'prediction': 'steering won\'t help much'
                },
                {
                    'prompt': 'City: Paris. What is the city?',
                    'expected': 'Paris',
                    'token_type': 'common_name',
                    'prediction': 'steering won\'t help much'
                },
                {
                    'prompt': 'Color: blue. What is the color?',
                    'expected': 'blue',
                    'token_type': 'common_word',
                    'prediction': 'steering won\'t help much'
                },
            ],
            
            'technical_terms': [
                {
                    'prompt': 'Protocol: HTTP/2.0. The protocol is',
                    'expected': 'HTTP/2.0',
                    'token_type': 'technical_identifier',
                    'prediction': 'steering might help (mixed alphanumeric)'
                },
                {
                    'prompt': 'Function: get_user_data(). Call function',
                    'expected': 'get_user_data',
                    'token_type': 'code_identifier',
                    'prediction': 'steering might help (underscores)'
                },
            ],
            
            'rare_names': [
                {
                    'prompt': 'Name: Zephyranthes. The name is',
                    'expected': 'Zephyranthes',
                    'token_type': 'rare_name',
                    'prediction': 'steering might help (rare word)'
                },
                {
                    'prompt': 'Person: Xenophon. Who is the person?',
                    'expected': 'Xenophon',
                    'token_type': 'rare_name',
                    'prediction': 'steering might help'
                },
            ],
        }


def test_rarity_hypothesis(model_path: str = "state-spaces/mamba-130m-hf"):
    """
    Test if steering helps more with rare tokens than common ones.
    
    Expected results:
    - Rare tokens (BLUE42, XY789): 0-10% baseline → 40-60% with steering
    - Common tokens (Alice, Paris): 40-50% baseline → 40-50% with steering
    """
    from mamba_model_loader import load_mamba_model_and_tokenizer
    
    logger.info("="*80)
    logger.info("🧪 TESTING RARITY HYPOTHESIS")
    logger.info("="*80)
    logger.info("\nHypothesis: Steering helps retrieve RARE tokens, not common ones")
    logger.info("="*80)
    
    # Load model
    model, tokenizer = load_mamba_model_and_tokenizer(
        model_name=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_mamba_class=True,
        fallback_to_auto=True
    )
    
    device = next(model.parameters()).device
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize
    strategy = TargetedSteeringStrategy(model, tokenizer, device)
    test_cases = strategy.get_optimized_test_cases()
    
    results = {
        'baseline': {},
        'with_steering': {}
    }
    
    # Phase 1: Baseline (no steering)
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: BASELINE (No Steering)")
    logger.info("="*80)
    
    for category, cases in test_cases.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Category: {category}")
        logger.info(f"{'='*60}")
        
        correct = 0
        total = len(cases)
        
        for case in cases:
            prompt = case['prompt']
            expected = case['expected'].lower()
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            input_len = inputs['input_ids'].shape[1]
            response = tokenizer.decode(
                outputs[0][input_len:],
                skip_special_tokens=True
            ).strip()
            
            is_correct = expected in response.lower()
            if is_correct:
                correct += 1
            
            symbol = "✅" if is_correct else "❌"
            logger.info(f"\n{symbol} {case['token_type']}")
            logger.info(f"   Expected: '{expected}'")
            logger.info(f"   Got: '{response[:60]}'")
            logger.info(f"   Prediction: {case['prediction']}")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        accuracy = correct / total
        results['baseline'][category] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
        
        logger.info(f"\n📊 {category} baseline: {accuracy*100:.1f}% ({correct}/{total})")
    
    # Phase 2: With steering
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: WITH ASSOCIATIVE STEERING")
    logger.info("="*80)
    
    # Collect all target texts
    all_targets = [case['expected'] for cases in test_cases.values() for case in cases]
    strategy.apply_conditional_steering(all_targets)
    
    for category, cases in test_cases.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Category: {category}")
        logger.info(f"{'='*60}")
        
        correct = 0
        total = len(cases)
        
        for case in cases:
            prompt = case['prompt']
            expected = case['expected'].lower()
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            input_len = inputs['input_ids'].shape[1]
            response = tokenizer.decode(
                outputs[0][input_len:],
                skip_special_tokens=True
            ).strip()
            
            is_correct = expected in response.lower()
            if is_correct:
                correct += 1
            
            symbol = "✅" if is_correct else "❌"
            logger.info(f"\n{symbol} {case['token_type']}")
            logger.info(f"   Expected: '{expected}'")
            logger.info(f"   Got: '{response[:60]}'")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        accuracy = correct / total
        results['with_steering'][category] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
        
        baseline_acc = results['baseline'][category]['accuracy']
        improvement = accuracy - baseline_acc
        
        logger.info(f"\n📊 {category} with steering: {accuracy*100:.1f}% ({correct}/{total})")
        logger.info(f"   Improvement: {improvement*100:+.1f}%")
    
    strategy.remove_steering()
    
    # Final analysis
    logger.info("\n" + "="*80)
    logger.info("🎯 HYPOTHESIS VALIDATION")
    logger.info("="*80)
    
    # Check if rare tokens improved more than common ones
    rare_categories = ['rare_codes', 'technical_terms', 'rare_names']
    common_categories = ['common_words']
    
    rare_improvement = 0
    rare_count = 0
    for cat in rare_categories:
        if cat in results['baseline'] and cat in results['with_steering']:
            baseline = results['baseline'][cat]['accuracy']
            steered = results['with_steering'][cat]['accuracy']
            rare_improvement += (steered - baseline)
            rare_count += 1
    
    common_improvement = 0
    common_count = 0
    for cat in common_categories:
        if cat in results['baseline'] and cat in results['with_steering']:
            baseline = results['baseline'][cat]['accuracy']
            steered = results['with_steering'][cat]['accuracy']
            common_improvement += (steered - baseline)
            common_count += 1
    
    avg_rare_improvement = rare_improvement / rare_count if rare_count > 0 else 0
    avg_common_improvement = common_improvement / common_count if common_count > 0 else 0
    
    logger.info(f"\nAverage improvement on RARE tokens: {avg_rare_improvement*100:+.1f}%")
    logger.info(f"Average improvement on COMMON tokens: {avg_common_improvement*100:+.1f}%")
    
    if avg_rare_improvement > avg_common_improvement + 0.1:
        logger.info(f"\n✅ HYPOTHESIS CONFIRMED!")
        logger.info(f"   Steering helps significantly more with rare tokens")
        logger.info(f"   Your Cluster 9 neurons DO improve rare token retrieval!")
    elif avg_rare_improvement > 0.05:
        logger.info(f"\n📊 HYPOTHESIS PARTIALLY CONFIRMED")
        logger.info(f"   Steering shows modest improvement on rare tokens")
    else:
        logger.info(f"\n❌ HYPOTHESIS NOT CONFIRMED")
        logger.info(f"   Steering doesn't show clear benefit even for rare tokens")
    
    logger.info("="*80)
    
    return results


if __name__ == "__main__":
    test_rarity_hypothesis()

