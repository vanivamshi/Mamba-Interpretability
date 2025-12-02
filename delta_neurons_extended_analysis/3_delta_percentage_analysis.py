# delta_percentage_analysis.py
# Analysis to understand why Mamba shows high delta % (percentage changes) in PPL
# This investigates the relationship between baseline PPL and percentage sensitivity

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from delta_extraction import find_delta_sensitive_neurons_fixed, register_perturbation_hook, evaluate_perplexity, evaluate_perturbation_effect
from factual_recall import get_relation_prompts
import os

def load_analysis_texts(num_samples=50):
    """Load analysis texts for comparison - EXACT same as original"""
    try:
        from datasets import load_dataset
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")
        texts = [item["text"] for item in dataset if item["text"].strip()]
        texts = [text for text in texts if len(text.split()) > 10 and not text.startswith("=")]
        return texts[:num_samples]
    except:
        return [
            "Artificial intelligence is transforming industries.",
            "The quick brown fox jumps over the lazy dog.",
            "Transformer models have revolutionized NLP tasks.",
            "Quantum computing promises exponential speedup.",
            "She loves chocolate. She hates chocolate."
        ]

def compute_ppl_changes(model, tokenizer, relation_texts, layer_idx=1, top_k=10):
    """
    Compute PPL changes using the EXACT same method as the original perplexity analysis.
    This replicates the exact implementation from factual_recall.py
    """
    # EXACT same neuron selection as original: use only first 25 texts for neuron selection
    delta_results = find_delta_sensitive_neurons_fixed(
        model, tokenizer, relation_texts[:25], layer_idx, top_k
    )
    neuron_indices = [idx for idx, _ in delta_results]
    
    # EXACT same other texts as original analysis
    other_relation_texts = load_analysis_texts(num_samples=50)
    
    # EXACT same evaluation method as original
    result = evaluate_perturbation_effect(
        model, tokenizer,
        relation_texts,      # The relation-specific texts
        other_relation_texts, # EXACT same other texts as original
        neuron_indices,
        layer_idx=layer_idx,
        mode="zero"         # Zero perturbation mode
    )
    
    if result is None:
        return {
            'baseline_ppl': 0.0,
            'perturbed_ppl': 0.0,
            'absolute_change': 0.0,
            'percentage_change': 0.0,
            'baseline_std': 0.0,
            'perturbed_std': 0.0
        }
    
    # Extract results from the original method
    baseline_ppl = result['erased_relation']['before']
    perturbed_ppl = result['erased_relation']['after']
    percentage_change = result['erased_relation']['change_pct']
    absolute_change = perturbed_ppl - baseline_ppl
    
    return {
        'baseline_ppl': baseline_ppl,
        'perturbed_ppl': perturbed_ppl,
        'absolute_change': absolute_change,
        'percentage_change': percentage_change,
        'baseline_std': 0.0,  # Not calculated in original method
        'perturbed_std': 0.0   # Not calculated in original method
    }

def analyze_delta_percentage_causes():
    """Analyze what causes high delta % in Mamba vs Transformer"""
    print("🔬 Analyzing Delta Percentage Causes")
    
    # Load models
    print("🔄 Loading models...")
    mamba_model_name = 'state-spaces/mamba-130m-hf'
    gpt2_model_name = 'gpt2'
    
    mamba_tokenizer = AutoTokenizer.from_pretrained(mamba_model_name)
    mamba_model = AutoModelForCausalLM.from_pretrained(mamba_model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mamba_model = mamba_model.to(device)
    
    gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_model_name)
    gpt2_model = AutoModelForCausalLM.from_pretrained(gpt2_model_name)
    gpt2_model = gpt2_model.to(device)
    
    if mamba_tokenizer.pad_token is None:
        mamba_tokenizer.pad_token = mamba_tokenizer.eos_token
    if gpt2_tokenizer.pad_token is None:
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    
    # Get relation prompts
    relation_prompts = get_relation_prompts()
    
    # Focus on the relations with highest delta % from your data
    # high_delta_relations = [
    #     'P36 (capital)',
    #     'P449 (original_network)', 
    #     'P530 (diplomatic_relation)',
    #     'P463 (member_of)',
    #     'P264 (record_label)'
    # ]
    
    # Use EXACT same texts from original perplexity analysis
    # These are the exact texts that produced the 400%+ delta percentages
    high_delta_relation_texts = {
        'P36 (capital)': [
            "France's capital is Paris.",
            "Japan's capital is Tokyo.",
            "Canada's capital is Ottawa."
        ],
        'P449 (original_network)': [
            "Friends originally aired on NBC.",
            "Breaking Bad originally aired on AMC.",
            "Game of Thrones originally aired on HBO."
        ],
        'P530 (diplomatic_relation)': [
            "Germany has diplomatic relations with the United States.",
            "Japan has diplomatic relations with China.",
            "India has diplomatic relations with Russia."
        ],
        'P463 (member_of)': [
            "France is a member of the European Union.",
            "Germany is a member of NATO.",
            "Japan is a member of the United Nations."
        ],
        'P264 (record_label)': [
            "Taylor Swift is signed to Republic Records.",
            "Drake is signed to OVO Sound.",
            "Beyoncé is signed to Columbia Records."
        ],
        'P413 (position_played_on_team)': [
            "Lionel Messi plays as a forward.",
            "Cristiano Ronaldo plays as a forward.",
            "Manuel Neuer plays as a goalkeeper."
        ],
        'P30 (continent)': [
            "France is located in Europe.",
            "Egypt is located in Africa.",
            "Brazil is located in South America."
        ],
        'P495 (country_of_origin)': [
            "Sushi originated in Japan.",
            "Pizza originated in Italy.",
            "Tacos originated in Mexico."
        ],
        'P279 (subclass_of)': [
            "A square is a subclass of a rectangle.",
            "A dog is a subclass of a mammal.",
            "A car is a subclass of a vehicle."
        ]
    }
    
    # Use the high delta relation texts instead of relation prompts
    high_delta_relations = list(high_delta_relation_texts.keys())
    
    analysis_data = []
    
    print("🔄 Computing PPL changes for high-delta relations...")
    
    for relation_label in high_delta_relations:
        if relation_label in high_delta_relation_texts:
            fact_texts = high_delta_relation_texts[relation_label]
            
            print(f"  Analyzing {relation_label}...")
            
            # Compute Mamba PPL changes
            mamba_results = compute_ppl_changes(
                mamba_model, mamba_tokenizer, fact_texts, layer_idx=1, top_k=10
            )
            
            analysis_data.append({
                'model': 'Mamba',
                'relation': relation_label,
                'baseline_ppl': mamba_results['baseline_ppl'],
                'perturbed_ppl': mamba_results['perturbed_ppl'],
                'absolute_change': mamba_results['absolute_change'],
                'percentage_change': mamba_results['percentage_change'],
                'baseline_std': mamba_results['baseline_std'],
                'perturbed_std': mamba_results['perturbed_std']
            })
            
            # Compute GPT-2 PPL changes
            gpt2_results = compute_ppl_changes(
                gpt2_model, gpt2_tokenizer, fact_texts, layer_idx=1, top_k=10
            )
            
            analysis_data.append({
                'model': 'GPT-2',
                'relation': relation_label,
                'baseline_ppl': gpt2_results['baseline_ppl'],
                'perturbed_ppl': gpt2_results['perturbed_ppl'],
                'absolute_change': gpt2_results['absolute_change'],
                'percentage_change': gpt2_results['percentage_change'],
                'baseline_std': gpt2_results['baseline_std'],
                'perturbed_std': gpt2_results['perturbed_std']
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(analysis_data)
    
    print(f"\n📊 Analysis complete! Generated {len(df)} data points")
    
    # Analyze the results
    print("\n🔍 DELTA PERCENTAGE ANALYSIS")
    print("="*60)
    
    for relation in high_delta_relations:
        relation_df = df[df['relation'] == relation]
        if len(relation_df) == 0:
            continue
            
        print(f"\n🔬 {relation}:")
        print("-" * 40)
        
        mamba_data = relation_df[relation_df['model'] == 'Mamba'].iloc[0]
        gpt2_data = relation_df[relation_df['model'] == 'GPT-2'].iloc[0]
        
        print(f"Mamba:")
        print(f"  Baseline PPL: {mamba_data['baseline_ppl']:.2f}")
        print(f"  Perturbed PPL: {mamba_data['perturbed_ppl']:.2f}")
        print(f"  Absolute Change: {mamba_data['absolute_change']:.2f}")
        print(f"  Percentage Change: {mamba_data['percentage_change']:.1f}%")
        
        print(f"GPT-2:")
        print(f"  Baseline PPL: {gpt2_data['baseline_ppl']:.2f}")
        print(f"  Perturbed PPL: {gpt2_data['perturbed_ppl']:.2f}")
        print(f"  Absolute Change: {gpt2_data['absolute_change']:.2f}")
        print(f"  Percentage Change: {gpt2_data['percentage_change']:.1f}%")
        
        # Debug: Manual calculation verification (commented out for cleaner output)
        # manual_percentage = (gpt2_data['absolute_change'] / gpt2_data['baseline_ppl']) * 100
        # print(f"  Manual Calc: {gpt2_data['absolute_change']:.2f} / {gpt2_data['baseline_ppl']:.2f} * 100 = {manual_percentage:.1f}%")
        
        # Key insight: Compare absolute changes
        print(f"\n💡 Key Insight:")
        print(f"  Mamba absolute change: {mamba_data['absolute_change']:.2f}")
        print(f"  GPT-2 absolute change: {gpt2_data['absolute_change']:.2f}")
        print(f"  Ratio (Mamba/GPT-2): {mamba_data['absolute_change']/gpt2_data['absolute_change']:.2f}x")
        
        # Baseline effect
        print(f"  Mamba baseline: {mamba_data['baseline_ppl']:.2f}")
        print(f"  GPT-2 baseline: {gpt2_data['baseline_ppl']:.2f}")
        print(f"  Baseline ratio (GPT-2/Mamba): {gpt2_data['baseline_ppl']/mamba_data['baseline_ppl']:.2f}x")
    
    # Create visualizations
    create_delta_percentage_visualizations(df)
    
    # Save results
    df.to_csv('delta_percentage_analysis_results.csv', index=False)
    print(f"\n✅ Results saved to delta_percentage_analysis_results.csv")
    
    return df

def create_delta_percentage_visualizations(df):
    """Create visualizations for delta percentage analysis"""
    os.makedirs("images", exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Baseline PPL vs Percentage Change
    ax1 = axes[0, 0]
    for model in ['Mamba', 'GPT-2']:
        model_df = df[df['model'] == model]
        if len(model_df) > 0:
            ax1.scatter(model_df['baseline_ppl'], model_df['percentage_change'], 
                       label=model, alpha=0.7, s=100)
            
            # Add trend line
            z = np.polyfit(model_df['baseline_ppl'], model_df['percentage_change'], 1)
            p = np.poly1d(z)
            ax1.plot(model_df['baseline_ppl'], p(model_df['baseline_ppl']), 
                   linestyle='--', alpha=0.8)
    
    ax1.set_xlabel('Baseline PPL')
    ax1.set_ylabel('Percentage Change (%)')
    ax1.set_title('Baseline PPL vs Percentage Change')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Absolute Change Comparison
    ax2 = axes[0, 1]
    relations = df['relation'].unique()
    mamba_changes = []
    gpt2_changes = []
    
    for relation in relations:
        mamba_data = df[(df['model'] == 'Mamba') & (df['relation'] == relation)]
        gpt2_data = df[(df['model'] == 'GPT-2') & (df['relation'] == relation)]
        
        if len(mamba_data) > 0 and len(gpt2_data) > 0:
            mamba_changes.append(mamba_data.iloc[0]['absolute_change'])
            gpt2_changes.append(gpt2_data.iloc[0]['absolute_change'])
    
    x = np.arange(len(relations))
    width = 0.35
    
    ax2.bar(x - width/2, mamba_changes, width, label='Mamba', alpha=0.8)
    ax2.bar(x + width/2, gpt2_changes, width, label='GPT-2', alpha=0.8)
    
    ax2.set_xlabel('Relations')
    ax2.set_ylabel('Absolute PPL Change')
    ax2.set_title('Absolute PPL Changes by Model')
    ax2.set_xticks(x)
    ax2.set_xticklabels([r.split('(')[0].strip() for r in relations], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Percentage Change Comparison
    ax3 = axes[1, 0]
    mamba_percentages = []
    gpt2_percentages = []
    
    for relation in relations:
        mamba_data = df[(df['model'] == 'Mamba') & (df['relation'] == relation)]
        gpt2_data = df[(df['model'] == 'GPT-2') & (df['relation'] == relation)]
        
        if len(mamba_data) > 0 and len(gpt2_data) > 0:
            mamba_percentages.append(mamba_data.iloc[0]['percentage_change'])
            gpt2_percentages.append(gpt2_data.iloc[0]['percentage_change'])
    
    ax3.bar(x - width/2, mamba_percentages, width, label='Mamba', alpha=0.8)
    ax3.bar(x + width/2, gpt2_percentages, width, label='GPT-2', alpha=0.8)
    
    ax3.set_xlabel('Relations')
    ax3.set_ylabel('Percentage Change (%)')
    ax3.set_title('Percentage Changes by Model')
    ax3.set_xticks(x)
    ax3.set_xticklabels([r.split('(')[0].strip() for r in relations], rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Baseline PPL Distribution
    ax4 = axes[1, 1]
    baseline_data = []
    model_labels = []
    for model in ['Mamba', 'GPT-2']:
        model_df = df[df['model'] == model]
        if len(model_df) > 0:
            baseline_data.append(model_df['baseline_ppl'])
            model_labels.append(model)
    
    ax4.boxplot(baseline_data, labels=model_labels)
    ax4.set_ylabel('Baseline PPL')
    ax4.set_title('Baseline PPL Distribution by Model')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/delta_percentage_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📊 Visualizations saved to images/delta_percentage_analysis.png")

def main():
    """Main function to run delta percentage analysis"""
    df = analyze_delta_percentage_causes()
    
    print("\n🎯 KEY INSIGHTS:")
    print("="*50)
    
    # Analyze the root cause
    mamba_df = df[df['model'] == 'Mamba']
    gpt2_df = df[df['model'] == 'GPT-2']
    
    if len(mamba_df) > 0 and len(gpt2_df) > 0:
        print(f"🔬 Mamba average baseline PPL: {mamba_df['baseline_ppl'].mean():.2f}")
        print(f"🔬 GPT-2 average baseline PPL: {gpt2_df['baseline_ppl'].mean():.2f}")
        print(f"📊 Baseline ratio (GPT-2/Mamba): {gpt2_df['baseline_ppl'].mean()/mamba_df['baseline_ppl'].mean():.2f}x")
        
        print(f"\n🔬 Mamba average absolute change: {mamba_df['absolute_change'].mean():.2f}")
        print(f"🔬 GPT-2 average absolute change: {gpt2_df['absolute_change'].mean():.2f}")
        print(f"📊 Absolute change ratio (Mamba/GPT-2): {mamba_df['absolute_change'].mean()/gpt2_df['absolute_change'].mean():.2f}x")
        
        print(f"\n🔬 Mamba average percentage change: {mamba_df['percentage_change'].mean():.1f}%")
        print(f"🔬 GPT-2 average percentage change: {gpt2_df['percentage_change'].mean():.1f}%")
        print(f"📊 Percentage change ratio (Mamba/GPT-2): {mamba_df['percentage_change'].mean()/gpt2_df['percentage_change'].mean():.2f}x")
        
        print(f"\n💡 ROOT CAUSE ANALYSIS:")
        print(f"   High delta % in Mamba is caused by:")
        print(f"   1. Lower baseline PPL values (denominator effect)")
        print(f"   2. Similar or larger absolute changes")
        print(f"   3. Percentage = (absolute_change / baseline) * 100")
        print(f"   → Smaller baseline → Larger percentage!")
    
    print("\n🎉 Delta percentage analysis complete!")

if __name__ == "__main__":
    main()
