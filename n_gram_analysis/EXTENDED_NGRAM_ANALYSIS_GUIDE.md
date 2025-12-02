# Extended N-gram Analysis: Step-by-Step Guide

This guide explains how to run the extended n-gram analysis based on the paper "In-Context Language Learning: Architectures and Algorithms" to study patterns in Mamba models.

## Overview

The extended analysis implements three key steps from the paper:

1. **Move from neuron triggers → head behaviors**: Analyze Mamba's equivalent of attention heads (input-state-output maps)
2. **Check correspondence between neurons and n-gram heads**: Identify which neurons activate for the same n-grams as n-gram heads
3. **Try hard-wiring n-gram bias**: Inject unigram/bigram frequency predictors into early layers

## Prerequisites

Make sure you have the required dependencies:
```bash
pip install torch numpy matplotlib seaborn transformers
```

## Step-by-Step Execution

### Step 1: Run the Extended Analysis

```bash
python3 4_ngram_neurons_extended.py
```

This will:
- Analyze each model in the `models_to_analyze` dictionary
- For each layer, perform the three analysis steps
- Generate plots and save results

### Step 2: Understanding the Analysis Steps

#### Step 2.1: Head Behavior Analysis (`MambaHeadAnalyzer`)

The `MambaHeadAnalyzer` class:

1. **Extracts state-space parameters** from Mamba layers:
   ```python
   def extract_state_space_params(self, layer_idx: int) -> Dict:
       # Extracts A, B, C, D matrices from SSM
   ```

2. **Analyzes n-gram dependencies**:
   ```python
   def analyze_ngram_dependencies(self, layer_idx: int, texts: List[str], n_max: int = 3):
       # Determines how state updates depend on last 1, 2, or 3 tokens
   ```

3. **Identifies n-gram heads**:
   ```python
   def identify_ngram_heads(self, layer_idx: int, texts: List[str]) -> Dict[int, List[int]]:
       # Finds components that behave like n-gram detectors
   ```

#### Step 2.2: Neuron-Head Correspondence (`NeuronHeadCorrespondenceAnalyzer`)

This class:

1. **Collects neuron n-gram triggers** (enhanced version of original function):
   ```python
   def collect_neuron_ngram_triggers(self, layer_idx: int, texts: List[str], n_max: int = 3):
       # Maps neurons to the n-grams that trigger them
   ```

2. **Analyzes correspondence**:
   ```python
   def analyze_correspondence(self, layer_idx: int, texts: List[str], ngram_heads: Dict[int, List[int]]):
       # Finds overlap between n-gram heads and n-gram neurons
   ```

#### Step 2.3: Hard-wired N-gram Bias Intervention (`HardwiredNgramBiasIntervention`)

This class:

1. **Creates n-gram predictors**:
   ```python
   def create_ngram_predictor(self, vocab_size: int, ngram_size: int = 1) -> nn.Module:
       # Simple frequency predictor for n-grams
   ```

2. **Injects bias into layers**:
   ```python
   def inject_ngram_bias(self, layer_idx: int, ngram_size: int = 1) -> nn.Module:
       # Combines original layer with n-gram bias
   ```

3. **Tests intervention effects**:
   ```python
   def test_intervention_effect(self, layer_idx: int, texts: List[str], ngram_size: int = 1):
       # Compares baseline vs intervention neuron clustering
   ```

### Step 3: Interpreting Results

#### Generated Plots

1. **`ngram_head_distribution.png`**: Shows distribution of n-gram heads across layers
2. **`head_neuron_correspondence.png`**: Heatmap of correspondence between heads and neurons
3. **`intervention_results.png`**: Effect of hard-wired n-gram bias on neuron clustering

#### Key Metrics to Look For

1. **N-gram Head Distribution**:
   - Do unigram heads appear earlier in smaller models?
   - Do bigger models show later clustering of unigram heads?

2. **Head-Neuron Correspondence**:
   - High overlap ratio indicates neurons and heads detect same n-grams
   - Low overlap suggests different mechanisms

3. **Intervention Effects**:
   - Does injecting n-gram bias reduce late clustering?
   - Does it improve n-gram coverage?

### Step 4: Customizing the Analysis

#### Modifying Models to Analyze

Edit the `models_to_analyze` dictionary:
```python
models_to_analyze = {
    "Mamba-130M": "state-spaces/mamba-130m-hf",
    "Mamba-370M": "state-spaces/mamba-370m-hf", 
    "Mamba-790M": "state-spaces/mamba-790m-hf",
    "Mamba-1.4B": "state-spaces/mamba-1.4b-hf",
    "GPT-2": "gpt2",
}
```

#### Adjusting Analysis Parameters

1. **N-gram size limit**:
   ```python
   ngram_deps = self.analyze_ngram_dependencies(layer_idx, texts, n_max=3)  # Change n_max
   ```

2. **Activation thresholds**:
   ```python
   GPT2_ACTIVATION_THRESHOLD = 0.7
   OTHER_MODELS_ACTIVATION_THRESHOLD = 0.6
   ```

3. **Text sample size**:
   ```python
   texts = load_analysis_texts(200)  # Change number of texts
   ```

### Step 5: Advanced Analysis

#### Comparing Different Model Sizes

To study the "later clustering" phenomenon:

1. Run analysis on multiple Mamba model sizes
2. Compare unigram head distributions across layers
3. Look for patterns where larger models show later unigram clustering

#### Intervention Experiments

To test the paper's architectural bias hypothesis:

1. Inject different types of n-gram bias (unigram, bigram, trigram)
2. Test on different layers (early vs late)
3. Measure impact on neuron clustering patterns

#### Custom N-gram Circuits

To identify specific n-gram circuits:

1. Focus on high-variance components in `identify_ngram_heads`
2. Analyze activation patterns for specific n-grams
3. Trace information flow through the state-space model

## Expected Results

Based on the paper, you should observe:

1. **N-gram head specialization**: Different heads specialize in different n-gram sizes
2. **Layer-wise organization**: Earlier layers handle simpler patterns (unigrams), later layers handle complex patterns
3. **Model size effects**: Larger models may show different clustering patterns
4. **Intervention benefits**: Hard-wired n-gram bias should improve performance and reduce late clustering

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce text sample size or use CPU
2. **Model loading errors**: Check model names and internet connection
3. **Hook registration errors**: Ensure model is in eval mode

### Performance Tips

1. Use smaller text samples for initial testing
2. Analyze fewer layers for quick results
3. Use CPU for smaller models if GPU memory is limited

## Next Steps

After running the analysis:

1. Compare results with the original paper's findings
2. Investigate specific n-gram circuits in detail
3. Experiment with different intervention strategies
4. Analyze the relationship between model architecture and n-gram processing

This extended analysis provides a comprehensive framework for studying n-gram processing in Mamba models, directly paralleling the methodology from "In-Context Language Learning: Architectures and Algorithms".
