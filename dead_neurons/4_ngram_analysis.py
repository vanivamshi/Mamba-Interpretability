# Run as: python3 4_ngram_analysis.py
"""
N-gram analysis of neuron activations
Reproduction of Figures 2–4 from:
"Neurons in Large Language Models: Dead, N-gram, Positional" (arXiv:2309.04827)

Now extended to multiple models (Mamba & GPT-2), matching paper style:
- Figure 2: per-model stacked histograms (saved separately)
- Figure 3a: overlay curves (# token-detecting neurons per layer, normalized depth)
- Figure 3b: overlay curves (per-layer & cumulative token coverage)
- Figure 4: overlay curves (new tokens overall vs new vs prev layer)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import json
import datetime
from utils import get_model_layers
from neuron_characterization import find_dead_neurons


# -------------------------
# Global threshold settings
# -------------------------
GPT2_ACTIVATION_THRESHOLD = 0.7
OTHER_MODELS_ACTIVATION_THRESHOLD = 0.6

# -------------------------
# Model zoo (same as 2_analyze_dead_neurons.py)
# -------------------------
models_to_analyze = {
    #"Mamba-130M": "state-spaces/mamba-130m-hf",
    #"Mamba-370M": "state-spaces/mamba-370m-hf",
    #"Mamba-790M": "state-spaces/mamba-790m-hf",
    #"Mamba-1.4B": "state-spaces/mamba-1.4b-hf",
    #"Mamba-2.8B": "state-spaces/mamba-2.8b-hf",
    "GPT-2": "gpt2",
}


# -------------------------
# N-gram trigger collection
# -------------------------
def collect_ngram_triggers(model, tokenizer, texts, layer_idx, n_max=3, model_label=None):
    """Collect n-gram triggers for neurons up to n_max."""
    # Determine activation threshold based on model
    if model_label == "GPT-2":
        activation_threshold = GPT2_ACTIVATION_THRESHOLD
    else:
        activation_threshold = OTHER_MODELS_ACTIVATION_THRESHOLD
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    layers = get_model_layers(model)
    neuron_triggers = defaultdict(lambda: defaultdict(set))
    current_input_ids = None
    current_attention_mask = None

    def hook_fn(module, inp, out):
        nonlocal current_input_ids, current_attention_mask
        if isinstance(out, tuple):
            out = out[0]
        acts = out.detach()
        if current_input_ids is None:
            return
        for b in range(acts.shape[0]):
            seq_len = acts.shape[1]
            mask = current_attention_mask[b] if current_attention_mask is not None else torch.ones(seq_len)
            for t in range(seq_len):
                if mask[t] == 0:
                    continue
                tok_id = int(current_input_ids[b, t].cpu().item())
                token_acts = acts[b, t].cpu().numpy()
                active_neurons = np.where(token_acts > activation_threshold)[0]
                for n in active_neurons:
                    neuron_triggers[int(n)][1].add((tok_id,))
                for nsize in range(2, n_max + 1):
                    if t + nsize <= seq_len and all(mask[t+i] == 1 for i in range(nsize)):
                        ngram = tuple(int(x) for x in current_input_ids[b, t:t+nsize].cpu().numpy())
                        for n in active_neurons:
                            neuron_triggers[int(n)][nsize].add(ngram)

    handle = layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True).to(device)
            current_input_ids = inputs["input_ids"]
            current_attention_mask = inputs.get("attention_mask", None)
            _ = model(**inputs)
    handle.remove()
    return neuron_triggers


# -------------------------
# Plotting Functions
# -------------------------
def plot_paper_figure2(all_layer_triggers, all_dead_neurons, model_label, save_dir="plots"):
    """Figure 2: Stacked histogram of neurons categorized by # of unigrams that trigger them."""
    os.makedirs(save_dir, exist_ok=True)
    num_layers = len(all_layer_triggers)

    bins = [(1,5),(6,10),(11,20),(21,50),(51,200),(201,1000),(1001,2000),(2001,5000)]
    bin_labels = ["1–5","6–10","11–20","21–50","51–200","201–1k","1k–2k","2k–5k"]
    colors = ["#355f2d","#5b9c51","#9bd68d","#d6e59f","#f9e4b7","#f9c4a0","#e78a61","#c0504d"]

    data = {label:[] for label in bin_labels}
    for l, neuron_triggers in all_layer_triggers.items():
        dead = all_dead_neurons[l]
        alive = [n for n in neuron_triggers if n not in dead]
        counts = [len(neuron_triggers[n][1]) for n in alive if 1 in neuron_triggers[n]]
        for (low,high), label in zip(bins,bin_labels):
            c = sum(1 for x in counts if low <= x <= high)
            data[label].append(c)

    depths = [l/num_layers for l in range(num_layers)]  # all layers

    fig, ax = plt.subplots(figsize=(10,6))
    bottom = np.zeros(len(depths))
    for label,color in zip(bin_labels,colors):
        vals = data[label][:len(depths)]
        ax.bar(depths, vals, bottom=bottom, label=label, color=color, width=0.05)
        bottom += np.array(vals)

    ax.set_xlabel("Layer (relative depth)")
    ax.set_ylabel("Number of Neurons")
    ax.set_title(f"Neurons Categorised by Unigram Triggers in {model_label}")
    ax.legend(title="Unigrams triggering neuron", bbox_to_anchor=(1.05,1), loc="upper left")
    fpath = os.path.join(save_dir,f"{model_label}_fig2_paper.png")
    plt.savefig(fpath,dpi=300,bbox_inches="tight")
    plt.close()
    print(f"✅ Saved Figure 2 for {model_label}: {fpath}")


def plot_overlay_figures(all_results, save_dir="plots"):
    """Overlay Figures 3a, 3b, 4 across models."""
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8,6))
    for model_label,res in all_results.items():
        num_layers = len(res["layer_triggers"])
        depths = [l/num_layers for l in range(num_layers)]
        per_layer_neurons = []
        for l,trig in res["layer_triggers"].items():
            per_layer_neurons.append(sum(1 for n in trig if 1 in trig[n] and len(trig[n][1])>0))
        plt.plot(depths, per_layer_neurons, marker="o", label=model_label)
    plt.xlabel("Layer (Relative Depth)")
    plt.ylabel("Number of Neurons")
    plt.title("Token-detecting neurons")
    plt.grid(True,alpha=0.3)
    plt.legend()
    f3a=os.path.join(save_dir,"fig3a_overlay.png")
    plt.savefig(f3a,dpi=300,bbox_inches="tight")
    plt.close()
    print(f"✅ Saved Figure 3a overlay: {f3a}")

    plt.figure(figsize=(8,6))
    for model_label,res in all_results.items():
        num_layers = len(res["layer_triggers"])
        depths = [l/num_layers for l in range(num_layers)]
        per_layer_tokens=[]; cumulative_tokens=[]; seen=set()
        for l,trig in res["layer_triggers"].items():
            tokens=set()
            for n in trig:
                if 1 in trig[n]:
                    tokens|={u[0] for u in trig[n][1] if len(u)==1}
            per_layer_tokens.append(len(tokens))
            seen|=tokens; cumulative_tokens.append(len(seen))
        # plt.plot(depths, per_layer_tokens, label=f"{model_label} per-layer")
        plt.plot(depths, cumulative_tokens, linestyle="--", label=f"{model_label} cumulative")
    plt.xlabel("Layer (Relative Depth)")
    plt.ylabel("Number of Tokens Covered")
    plt.title("Token coverage")
    plt.grid(True,alpha=0.3)
    plt.legend()
    f3b=os.path.join(save_dir,"fig3b_overlay.png")
    plt.savefig(f3b,dpi=300,bbox_inches="tight")
    plt.close()
    print(f"✅ Saved Figure 3b overlay: {f3b}")

    plt.figure(figsize=(8,6))
    for model_label,res in all_results.items():
        num_layers = len(res["layer_triggers"])
        depths = [l/num_layers for l in range(num_layers)]
        new_overall=[]; seen=set()
        # new_vs_prev=[]; prev=set()  # Commented out: removed new vs prev calculation
        for l,trig in res["layer_triggers"].items():
            tokens=set()
            for n in trig:
                if 1 in trig[n]:
                    tokens|={u[0] for u in trig[n][1] if len(u)==1}
            new_overall.append(len(tokens-seen))
            # new_vs_prev.append(len(tokens-prev))  # Commented out: removed new vs prev calculation
            seen|=tokens
            # prev=tokens  # Commented out: removed prev tracking
        plt.plot(depths, new_overall, marker="o", label=f"{model_label} new overall")
        # plt.plot(depths, new_vs_prev, marker="s", linestyle="--", label=f"{model_label} new vs prev")  # Commented out: removed new vs prev plot
    plt.xlabel("Layer (Relative Depth)")
    plt.ylabel("Number of New Tokens")
    plt.title("Number of Covered New Tokens")
    plt.grid(True,alpha=0.3)
    plt.legend()
    f4=os.path.join(save_dir,"fig4_overlay.png"); plt.savefig(f4,dpi=300,bbox_inches="tight"); plt.close()
    print(f"✅ Saved Figure 4 overlay: {f4}")


def save_analysis_results_to_log(all_results, save_dir="logs"):
    """Save analysis results to JSON log file for offline plotting."""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_dir, f"ngram_analysis_{timestamp}.json")
    
    # Convert sets to lists for JSON serialization
    serializable_results = {}
    for model_label, res in all_results.items():
        serializable_results[model_label] = {
            "layer_triggers": {},
            "dead_neurons": {}
        }
        
        # Convert layer triggers (convert sets to lists)
        for layer_idx, triggers in res["layer_triggers"].items():
            serializable_results[model_label]["layer_triggers"][str(layer_idx)] = {}
            for neuron_idx, ngram_triggers in triggers.items():
                serializable_results[model_label]["layer_triggers"][str(layer_idx)][str(neuron_idx)] = {}
                for ngram_size, ngrams in ngram_triggers.items():
                    serializable_results[model_label]["layer_triggers"][str(layer_idx)][str(neuron_idx)][str(ngram_size)] = [
                        list(ngram) for ngram in ngrams
                    ]
        
        # Convert dead neurons (convert sets to lists)
        for layer_idx, dead_set in res["dead_neurons"].items():
            serializable_results[model_label]["dead_neurons"][str(layer_idx)] = list(dead_set) if dead_set else []
    
    log_data = {
        "timestamp": timestamp,
        "models_analyzed": list(all_results.keys()),
        "results": serializable_results
    }
    
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    print(f"✅ Analysis results logged to '{log_file}' for offline plotting.")


# -------------------------
# Main
# -------------------------
if __name__=="__main__":
    try:
        from main import setup_model_and_tokenizer, load_analysis_texts
    except ImportError:
        print("main.py not found; please place this script with main.py")
        exit(1)

    texts=load_analysis_texts(200)  # subset for runtime
    all_results={}

    for model_label,model_name in models_to_analyze.items():
        print(f"\n🔍 Analyzing {model_label}")
        model,tokenizer=setup_model_and_tokenizer(model_name)
        num_layers=getattr(model.config,"num_hidden_layers",6)
        layer_triggers={}; dead_neurons={}
        for l in range(num_layers):
            print(f"  Layer {l}/{num_layers-1}")
            # Use model-specific threshold for dead neuron detection
            from neuron_characterization import find_dead_neurons_custom_threshold
            dead_threshold = GPT2_ACTIVATION_THRESHOLD if model_label == "GPT-2" else OTHER_MODELS_ACTIVATION_THRESHOLD
            dead,_=find_dead_neurons_custom_threshold(model,tokenizer,texts[:50],l, threshold=dead_threshold)
            # Function now automatically determines threshold based on model_label
            trig=collect_ngram_triggers(model,tokenizer,texts[:100],l, model_label=model_label)
            layer_triggers[l]=trig; dead_neurons[l]=dead
        all_results[model_label]={"layer_triggers":layer_triggers,"dead_neurons":dead_neurons}
        plot_paper_figure2(layer_triggers,dead_neurons,model_label)

    plot_overlay_figures(all_results)
    
    # Save results to log file for offline plotting
    save_analysis_results_to_log(all_results)
    
    print("\n🎉 Multi-model N-gram analysis complete! Paper-style Figures 2–4 saved in 'plots/'")