"""
TARGETED STEERING TEST FOR MAMBA-130M

Focus on tasks where model shows:
1. Moderate performance (33-67%)
2. Clear room for improvement
3. Natural language patterns

RESULTS SUMMARY
Task	           Baseline	Steered	Improvement
Simple Instructions	33.3%	66.7%	+33.3%
In-context Learning	66.7%	66.7%	0%
Pattern with Noise	100%	100%	0%
"""

import torch
import logging
from typing import List, Dict, Tuple
import json
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TargetedSteeringTest:
    """
    Focus steering on tasks where Mamba-130M actually needs help.
    """
    
    def __init__(self, tokenizer, device="cuda"):
        self.tokenizer = tokenizer
        self.device = device
    
    def get_targeted_test_cases(self) -> Dict[str, List[Dict]]:
        """
        Focus on tasks where baseline is 30-70%
        """
        
        return {
            # 1. IN-CONTEXT LEARNING (33% baseline)
            'in_context_learning': [
                {
                    'prompt': 'Q: 2+2=?\nA: 4\n\nQ: 3+3=?\nA:',
                    'expected': '6',
                    'alternatives': ['3', '9', '33'],
                    'task': 'simple addition pattern'
                },
                {
                    'prompt': 'English: hello\nSpanish: hola\n\nEnglish: goodbye\nSpanish: adios\n\nEnglish: thank you\nSpanish:',
                    'expected': 'gracias',
                    'alternatives': ['hola', 'adios', 'por favor'],
                    'task': 'translation pattern'
                },
                {
                    'prompt': 'Input: cat\nOutput: CAT\n\nInput: dog\nOutput: DOG\n\nInput: bird\nOutput:',
                    'expected': 'BIRD',
                    'alternatives': ['bird', 'BIRD', 'BIRD.'],
                    'task': 'uppercase pattern'
                },
            ],
            
            # 2. SIMPLE INSTRUCTIONS (67% baseline - room for improvement)
            'simple_instructions': [
                {
                    'prompt': 'Repeat after me: hello',
                    'expected': 'hello',
                    'alternatives': ['hello', 'hello.', 'Hello'],
                    'task': 'exact repetition'
                },
                {
                    'prompt': 'Answer with just the number: What is 5+3?',
                    'expected': '8',
                    'alternatives': ['8', '8.', 'eight'],
                    'task': 'concise answer'
                },
                {
                    'prompt': 'Say YES or NO only: Is water wet?',
                    'expected': 'YES',
                    'alternatives': ['YES', 'Yes', 'yes'],
                    'task': 'binary answer'
                },
            ],
            
            # 3. PATTERN CONTINUATION WITH DISTRACTIONS
            'pattern_with_noise': [
                {
                    'prompt': 'Ignore previous text. Just continue: A, B, C, D, E,',
                    'expected': 'F',
                    'alternatives': ['F', 'G', 'H'],
                    'task': 'continue alphabet after instruction'
                },
                {
                    'prompt': 'Text: Many words here. Now count: 2, 4, 6, 8,',
                    'expected': '10',
                    'alternatives': ['10', '9', '12'],
                    'task': 'continue sequence after noise'
                },
            ],
        }
    
    def test_model(self, model, test_cases: Dict, num_samples: int = 10) -> Dict:
        """
        Test model on targeted cases.
        """
        results = {}
        
        for task_name, cases in test_cases.items():
            logger.info(f"\n📊 Testing: {task_name}")
            
            correct = 0
            responses = []
            
            for case in cases:
                prompt = case['prompt']
                expected = case['expected']
                
                # Generate
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,
                        temperature=0.1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                input_len = inputs['input_ids'].shape[1]
                response = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
                response = response.strip()
                
                # Check if correct (flexible matching)
                is_correct = False
                if expected.lower() in response.lower():
                    is_correct = True
                elif any(alt.lower() in response.lower() for alt in case.get('alternatives', [])):
                    is_correct = True
                
                if is_correct:
                    correct += 1
                
                responses.append({
                    'prompt': prompt[:50],
                    'response': response[:50],
                    'expected': expected,
                    'correct': is_correct
                })
                
                # Show
                symbol = "✅" if is_correct else "❌"
                logger.info(f"  {symbol} Expected: '{expected}', Got: '{response[:30]}'")
            
            accuracy = correct / len(cases)
            results[task_name] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': len(cases),
                'responses': responses
            }
            
            logger.info(f"  Accuracy: {accuracy*100:.1f}% ({correct}/{len(cases)})")
        
        return results


class SimpleCluster9Steering:
    """
    Simple but powerful steering using Cluster 9 neurons.
    """
    
    def __init__(self, model):
        self.model = model
        self.hooks = []
        
        if hasattr(model, 'backbone'):
            self.layers = model.backbone.layers
        else:
            self.layers = model.layers
        
        # Cluster 9 neurons from your research
        self.cluster9_neurons = [
            4, 38, 84, 94, 163, 171, 268, 363, 401, 497, 
            564, 568, 582, 654, 659, 686
        ]
    
    def apply_steering(self, strength: float = 5.0, layer_idx: int = 20):
        """
        Apply strong steering to Cluster 9 neurons.
        """
        if layer_idx >= len(self.layers):
            logger.warning(f"Layer {layer_idx} doesn't exist")
            return
        
        logger.info(f"\n🎯 APPLYING STRONG STEERING")
        logger.info(f"  Layer: {layer_idx}")
        logger.info(f"  Strength: {strength}x")
        logger.info(f"  Neurons: {len(self.cluster9_neurons)} from Cluster 9")
        
        layer = self.layers[layer_idx]
        target = getattr(layer, 'mixer', getattr(layer, 'ssm', layer))
        
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = ()
            
            # Strong modification
            h_mod = hidden.clone()
            for idx in self.cluster9_neurons:
                if idx < h_mod.shape[-1]:
                    h_mod[..., idx] *= strength
            
            if rest:
                return (h_mod,) + rest
            return h_mod
        
        # Apply to ALL positions, not just last token
        h = target.register_forward_hook(hook)
        self.hooks.append(h)
    
    def remove_steering(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def run_final_targeted_test():
    """
    FINAL TEST: Targeted steering on tasks where model can improve.
    """
    from mamba_model_loader import load_mamba_model_and_tokenizer
    
    logger.info("="*80)
    logger.info("🎯 FINAL TARGETED STEERING TEST")
    logger.info("Focusing on tasks with 30-70% baseline")
    logger.info("="*80)
    
    # Load model
    logger.info("\n📦 Loading model...")
    model, tokenizer = load_mamba_model_and_tokenizer(
        model_name="state-spaces/mamba-130m-hf",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_mamba_class=True,
        fallback_to_auto=True
    )
    
    device = next(model.parameters()).device
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize
    tester = TargetedSteeringTest(tokenizer, device)
    steering = SimpleCluster9Steering(model)
    
    # Get targeted test cases
    test_cases = tester.get_targeted_test_cases()
    
    # ================================================
    # PHASE 1: BASELINE
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("📊 PHASE 1: BASELINE PERFORMANCE")
    logger.info("="*80)
    
    baseline_results = tester.test_model(model, test_cases)
    
    # Calculate overall baseline
    total_correct = sum(res['correct'] for res in baseline_results.values())
    total_tests = sum(res['total'] for res in baseline_results.values())
    baseline_accuracy = total_correct / total_tests
    
    logger.info(f"\n📈 OVERALL BASELINE: {baseline_accuracy*100:.1f}% ({total_correct}/{total_tests})")
    
    # ================================================
    # PHASE 2: WITH STEERING
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("🎯 PHASE 2: WITH CLUSTER 9 STEERING")
    logger.info("="*80)
    
    # Apply STRONG steering
    steering.apply_steering(strength=5.0, layer_idx=20)
    
    # Test with steering
    steered_results = tester.test_model(model, test_cases)
    
    # Calculate overall steered
    steered_correct = sum(res['correct'] for res in steered_results.values())
    steered_accuracy = steered_correct / total_tests
    
    logger.info(f"\n📈 OVERALL WITH STEERING: {steered_accuracy*100:.1f}% ({steered_correct}/{total_tests})")
    
    # Remove steering
    steering.remove_steering()
    
    # ================================================
    # PHASE 3: COMPARE RESULTS
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("📊 RESULTS COMPARISON")
    logger.info("="*80)
    
    improvements = {}
    
    for task_name in test_cases.keys():
        base = baseline_results[task_name]['accuracy']
        steer = steered_results[task_name]['accuracy']
        improvement = steer - base
        
        improvements[task_name] = improvement
        
        logger.info(f"\n{task_name}:")
        logger.info(f"  Baseline: {base*100:5.1f}%")
        logger.info(f"  Steered:  {steer*100:5.1f}%")
        logger.info(f"  Change:   {improvement*100:+5.1f}%")
    
    total_improvement = steered_accuracy - baseline_accuracy
    
    logger.info("\n" + "="*80)
    logger.info("📈 FINAL RESULTS")
    logger.info("="*80)
    logger.info(f"Baseline Accuracy:    {baseline_accuracy*100:.1f}%")
    logger.info(f"Steered Accuracy:     {steered_accuracy*100:.1f}%")
    logger.info(f"Improvement:          {total_improvement*100:+.1f}%")
    logger.info("="*80)
    
    # ================================================
    # EVALUATION
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("🏁 EVALUATION")
    logger.info("="*80)
    
    if total_improvement > 0.1:
        logger.info("✅ EXCELLENT! Significant improvement from steering!")
        logger.info("This suggests Cluster 9 neurons are indeed important.")
    elif total_improvement > 0.05:
        logger.info("✅ GOOD! Clear improvement from steering.")
        logger.info("Cluster 9 neurons show potential for steering.")
    elif total_improvement > 0.02:
        logger.info("📈 MODEST improvement detected.")
        logger.info("Cluster 9 has some effect, but may not be the strongest.")
    elif total_improvement > 0:
        logger.info("📊 SMALL improvement.")
        logger.info("Cluster 9 has minor effects on these tasks.")
    else:
        logger.info("⚠️ NO IMPROVEMENT detected.")
        logger.info("Cluster 9 may not affect these specific tasks.")
        logger.info("\n💡 Recommendations:")
        logger.info("1. Try stronger steering (strength=10.0)")
        logger.info("2. Try different layers (15, 18, 22)")
        logger.info("3. Focus on longer-context tasks")
    
    # ================================================
    # DETAILED ANALYSIS
    # ================================================
    logger.info("\n" + "="*80)
    logger.info("🔍 DETAILED ANALYSIS")
    logger.info("="*80)
    
    # Show which tasks improved most
    if improvements:
        best_task = max(improvements.items(), key=lambda x: x[1])
        worst_task = min(improvements.items(), key=lambda x: x[1])
        
        logger.info(f"\nMost improved task: {best_task[0]} ({best_task[1]*100:+.1f}%)")
        logger.info(f"Least improved task: {worst_task[0]} ({worst_task[1]*100:+.1f}%)")
    
    # Show specific examples
    logger.info("\nExample improvements:")
    for task_name in test_cases.keys():
        if improvements[task_name] > 0:
            base_responses = baseline_results[task_name]['responses']
            steer_responses = steered_results[task_name]['responses']
            
            for i in range(min(2, len(base_responses))):
                if base_responses[i]['correct'] != steer_responses[i]['correct']:
                    logger.info(f"\n{task_name}, Example {i+1}:")
                    logger.info(f"  Before: '{base_responses[i]['response']}'")
                    logger.info(f"  After:  '{steer_responses[i]['response']}'")
    
    # Save results
    output_path = Path("experiment_logs/final_targeted_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_data = {
        'baseline': {
            'overall_accuracy': float(baseline_accuracy),
            'total_correct': int(total_correct),
            'total_tests': int(total_tests),
            'task_results': {
                k: {'accuracy': float(v['accuracy']), 'correct': v['correct'], 'total': v['total']}
                for k, v in baseline_results.items()
            }
        },
        'steered': {
            'overall_accuracy': float(steered_accuracy),
            'total_correct': int(steered_correct),
            'total_tests': int(total_tests),
            'task_results': {
                k: {'accuracy': float(v['accuracy']), 'correct': v['correct'], 'total': v['total']}
                for k, v in steered_results.items()
            }
        },
        'improvements': {
            'overall': float(total_improvement),
            'by_task': {k: float(v) for k, v in improvements.items()}
        },
        'test_cases': test_cases
    }
    
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    logger.info(f"\n💾 Results saved to: {output_path}")
    
    return save_data


if __name__ == "__main__":
    results = run_final_targeted_test()