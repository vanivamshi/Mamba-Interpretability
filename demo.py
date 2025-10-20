#!/usr/bin/env python3
"""
Demo script for Mamba Mechanistic Interpretability Framework

This script demonstrates the key features of the framework with a simple example.
"""

import torch
import numpy as np
from experimental_framework import ExperimentConfig, MambaMechanisticAnalyzer
from transformers import AutoTokenizer, AutoModelForCausalLM, MambaForCausalLM

def demo_basic_analysis():
    """Demonstrate basic analysis workflow."""
    print("🚀 Mamba Mechanistic Interpretability Framework Demo")
    print("=" * 60)
    
    # Step 1: Setup configuration
    config = ExperimentConfig(
        model_name="state-spaces/mamba-130m-hf",
        layer_idx=0,
        num_samples=20,  # Small number for demo
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"📊 Configuration:")
    print(f"   Model: {config.model_name}")
    print(f"   Layer: {config.layer_idx}")
    print(f"   Samples: {config.num_samples}")
    print(f"   Device: {config.device}")
    print()
    
    # Step 2: Initialize analyzer
    analyzer = MambaMechanisticAnalyzer(config)
    
    try:
        # Step 3: Setup environment
        print("🔧 Setting up experimental environment...")
        analyzer.setup()
        print("✅ Setup complete!")
        print()
        
        # Step 4: Prepare sample texts
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming industries.",
            "Machine learning models require training data.",
            "Natural language processing has advanced significantly.",
            "Deep learning architectures continue to evolve."
        ] * 4  # Repeat to get 20 samples
        
        print(f"📝 Using {len(sample_texts)} sample texts")
        print()
        
        # Step 5: Collect activations
        print("🔍 Collecting activations...")
        activations = analyzer.collect_activations(sample_texts)
        print(f"✅ Collected activations for {len(activations)} layers")
        print()
        
        # Step 6: Discover interpretable features with SAE
        print("🧠 Discovering interpretable features with SAE...")
        sae_results = analyzer.discover_interpretable_features(layer_idx=0)
        print("✅ SAE analysis complete!")
        print()
        
        # Step 7: Run hypothesis probes
        print("🔬 Running hypothesis probes...")
        probe_results = analyzer.run_hypothesis_probes(layer_idx=0)
        print("✅ Hypothesis probes complete!")
        print()
        
        # Step 8: Select candidate circuits
        print("🎯 Selecting candidate circuits...")
        circuits = analyzer.select_candidate_circuits(layer_idx=0)
        print(f"✅ Selected {len(circuits)} candidate circuits")
        print()
        
        # Step 9: Test circuit causality
        print("⚡ Testing circuit causality...")
        patching_results = analyzer.test_circuit_causality(layer_idx=0)
        print("✅ Circuit causality testing complete!")
        print()
        
        # Step 10: Generate report
        print("📊 Generating comprehensive report...")
        report = analyzer.generate_comprehensive_report()
        print("✅ Report generated!")
        print()
        
        # Display results summary
        print("🎉 Analysis Complete!")
        print("=" * 60)
        print(f"📁 Results saved in: {analyzer.experiment_logger.experiment_dir}")
        print(f"🔍 Significant findings: {len(report['significant_findings'])}")
        
        if report['significant_findings']:
            print("\nSignificant Findings:")
            for finding in report['significant_findings']:
                print(f"  • {finding}")
        
        print(f"\n📋 Generated Files:")
        for file_path in analyzer.experiment_logger.experiment_dir.glob("*.json"):
            print(f"  • {file_path.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

def demo_sae_only():
    """Demonstrate SAE analysis only."""
    print("\n🧠 SAE Analysis Demo")
    print("=" * 40)
    
    try:
        # Load model
        model_name = "state-spaces/mamba-130m-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Use MambaForCausalLM for proper SSM architecture initialization
        try:
            model = MambaForCausalLM.from_pretrained(model_name)
            print("✅ Successfully loaded model using MambaForCausalLM")
        except Exception as e:
            print(f"⚠️ Failed to load with MambaForCausalLM: {e}")
            print("Falling back to AutoModelForCausalLM...")
            model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        
        # Generate sample activations
        text = "The quick brown fox jumps over the lazy dog."
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Extract activations (simplified)
        with torch.no_grad():
            outputs = model(**inputs)
            # Use logits as proxy for activations
            activations = outputs.logits.view(-1, outputs.logits.shape[-1])
        
        # Generate task labels
        task_labels = torch.randint(0, 3, (activations.shape[0],))
        
        # Run SAE analysis
        from sparse_autoencoder import run_sae_analysis
        
        print("Running SAE analysis...")
        sae_results = run_sae_analysis(
            activations=activations,
            task_labels=task_labels,
            config={
                'latent_dim_ratio': 0.3,
                'sparsity_weight': 1e-3,
                'num_epochs': 20,  # Reduced for demo
                'batch_size': 64,
                'learning_rate': 1e-3
            }
        )
        
        print("✅ SAE analysis complete!")
        print(f"   Training epochs: {len(sae_results['training_history']['train_losses'])}")
        
        if sae_results['correlation_results']:
            max_corr = max(sae_results['correlation_results']['max_correlation'].values())
            print(f"   Max correlation: {max_corr:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ SAE demo failed: {e}")
        return False

def demo_activation_patching():
    """Demonstrate activation patching."""
    print("\n⚡ Activation Patching Demo")
    print("=" * 40)
    
    try:
        # Load model
        model_name = "state-spaces/mamba-130m-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Use MambaForCausalLM for proper SSM architecture initialization
        try:
            model = MambaForCausalLM.from_pretrained(model_name)
            print("✅ Successfully loaded model using MambaForCausalLM")
        except Exception as e:
            print(f"⚠️ Failed to load with MambaForCausalLM: {e}")
            print("Falling back to AutoModelForCausalLM...")
            model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        
        # Prepare test inputs
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming industries."
        ]
        
        test_inputs = []
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=32)
            test_inputs.append(inputs["input_ids"].to(device))
        
        test_input = torch.cat(test_inputs, dim=0)
        
        # Define candidate circuits (random for demo)
        candidate_circuits = [
            [100, 200, 300],  # Circuit 1
            [150, 250, 350],  # Circuit 2
            [50, 150, 250]    # Circuit 3
        ]
        
        # Run activation patching analysis
        from activation_patching import run_activation_patching_analysis
        
        print("Running activation patching analysis...")
        patching_results = run_activation_patching_analysis(
            model=model,
            inputs=test_input,
            candidate_circuits=candidate_circuits,
            layer_idx=0,
            reference_inputs=test_input
        )
        
        print("✅ Activation patching complete!")
        print(f"   Circuits tested: {len(candidate_circuits)}")
        
        if 'significant_circuits' in patching_results:
            print(f"   Significant circuits: {len(patching_results['significant_circuits'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Activation patching demo failed: {e}")
        return False

def main():
    """Run all demos."""
    print("🎯 Mamba Mechanistic Interpretability Framework - Demo Suite")
    print("=" * 70)
    
    demos = [
        ("Basic Analysis", demo_basic_analysis),
        ("SAE Only", demo_sae_only),
        ("Activation Patching", demo_activation_patching)
    ]
    
    results = []
    
    for name, demo_func in demos:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            success = demo_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 Demo Results Summary:")
    print("=" * 70)
    
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name:<25} {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nOverall: {passed}/{total} demos passed")
    
    if passed == total:
        print("🎉 All demos passed! Framework is working correctly.")
    else:
        print("⚠️  Some demos failed. Check error messages above.")

if __name__ == "__main__":
    main()
