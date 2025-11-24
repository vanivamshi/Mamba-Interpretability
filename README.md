1. Project Overview

1.1. This project explores interpretability in Mamba models. Unlike transformer-based models, Mamba does not have explicit neurons or attention weights. To enable interpretability, we:

1.2. Model layer attention weights into equivalent neuron weights, neuron activations, and neuron importance.

1.3. Use these derived neuron representations to analyze Mamba's behavior across different types of texts.

1.4. Cluster related neurons into neuron groups for better interpretability.

2. Current Progress

2.1. Implemented multi-layer analysis for 6 neuron types across all 24 layers.

2.2. Provided comprehensive visualizations (heatmaps and layer-wise plots).

2.3. Analyzed universality, delta-sensitive, projection-dominant, speciality, dead, and causal neurons.

3. Quick Start

3.1. Run complete analysis: `python analysis/main_all_layers.py`

3.2. Results saved to: `plots/all_layers_TIMESTAMP/`
