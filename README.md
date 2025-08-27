1. Project Overview

1.1. This project explores interpretability in Mamba models. Unlike transformer-based models, Mamba does not have explicit neurons or attention weights. To enable interpretability, we:

1.2. Model layer attention weights into equivalent neuron weights, neuron activations, and neuron importance.

1.3. Use these derived neuron representations to analyze Mamba’s behavior across different types of texts.

1.4. Cluster related neurons into neuron groups for better interpretability.

2.Current Progress

2.1. Implemented initial code to derive neuron groups.

2.2. Provided baseline analysis for different neuron groups.

2.3. Early-stage experiments for studying text-specific neuron activations.
