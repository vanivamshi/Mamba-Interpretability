## Overview

This project develops a **mechanistic interpretability framework for State Space Models (SSMs)** that explains their internal computation from first principles and connects recurrence-based dynamics to transformer-like behaviors.

We analyze SSMs at the level of **activation subspaces, parameter clusters, and causal circuits**, introducing tools to decompose sequence computation into interpretable phases and functional modules. The framework enables **post-hoc steering**, **causal validation**, and **architecture-level improvements** grounded in mechanistic evidence rather than heuristics.


## Project Does

* **Identifies interpretable activation subspaces** in SSMs using sparse autoencoders, attribution statistics, and stability metrics, avoiding neuron-level assumptions that do not translate cleanly from transformers.
* **Discovers causal circuits and bottlenecks** governing temporal gating, compression, and information routing using targeted interventions and KL-based causal analysis.
* **Introduces Stochastic Parameter Decomposition (SPD)** to cluster parameters by causal influence, stability, and functional role rather than architectural labels.
* **Decomposes SSM computation into modular phases**, revealing distinct regimes of feature extraction, bottlenecked reorganization, specialization, and output projection.
* **Demonstrates steerability** by selectively ablating and amplifying mechanistically identified components, improving performance without retraining.
* **Translates interpretability insights into architectural modifications**, yielding an SSM variant with improved long-context reasoning, stability, and parameter efficiency.


## Key Contributions

* A **parameter-centric view of SSMs** that separates temporal control, state evolution, input conditioning, and output projection into functional modules.
* Evidence that **SSMs rely on sparse, high-impact activation subspaces** rather than distributed computation, explaining both their efficiency and brittleness.
* Causal validation showing that small, stable parameter sets can exert **disproportionate control over sequence behavior**.
* A mechanistically motivated path from **interpretation → steering → architecture design**.

## Why This Matters

SSMs are increasingly competitive with attention-based models but lack clear interpretability tools. This work:

* Bridges the gap between **recurrence and transformer-style representations**
* Enables **mechanistic debugging and control** of long-context models
* Shows how interpretability can directly inform **model design**, not just analysis

