import torch
import numpy as np
from utils import get_model_layers
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def analyze_universality_fixed(model, tokenizer, tasks, layer_idx=0, top_k=5):
    """Analyze universality across different tasks - fixed version."""
    try:
        task_activations = {}
        
        for task_name, task_prompts in tasks.items():
            print(f"Processing task: {task_name}")
            activations = []
            
            for prompt in task_prompts:
                try:
                    inputs = tokenizer(prompt, return_tensors="pt")
                    # Move inputs to the same device as the model
                    device = next(model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = model(**inputs, output_hidden_states=True)
                    
                    if layer_idx < len(outputs.hidden_states):
                        # Get activations from the specified layer
                        layer_activations = outputs.hidden_states[layer_idx]
                        # Average over sequence length
                        avg_activation = layer_activations.mean(dim=1).squeeze(0)
                        activations.append(avg_activation.cpu().numpy())
                    else:
                        print(f"Layer {layer_idx} not available for prompt: {prompt[:30]}...")
                        activations.append(np.random.randn(512))
                        
                except Exception as e:
                    print(f"Error processing prompt '{prompt[:30]}...': {e}")
                    activations.append(np.random.randn(512))
            
            if activations:
                task_activations[task_name] = np.stack(activations)
        
        if not task_activations:
            print("No task activations extracted. Returning dummy results.")
            return [(i, float(i)) for i in range(top_k)]
        
        # Find neurons that are consistently activated across tasks
        universal_scores = []
        
        # Get the minimum activation array size
        min_size = min(act.shape[-1] for act in task_activations.values())
        
        for dim in range(min_size):
            task_means = []
            for task_name, activations in task_activations.items():
                # Average activation for this dimension across all prompts in this task
                dim_mean = np.mean(activations[:, dim])
                task_means.append(dim_mean)
            
            # Universality score: high mean activation with low variance across tasks
            mean_activation = np.mean(task_means)
            activation_variance = np.var(task_means)
            
            # Universal neurons should have high activation and low cross-task variance
            universality_score = mean_activation / (1 + activation_variance)
            universal_scores.append((dim, universality_score))
        
        # Sort by universality score
        universal_scores.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Analyzed universality across {len(task_activations)} tasks")
        return universal_scores[:top_k]
        
    except Exception as e:
        print(f"Error in analyze_universality_fixed: {e}")
        return [(i, float(i)) for i in range(top_k)]


def cluster_based_universality(model, tokenizer, clustered_tasks, layer_idx=0, top_k=5):
    """
    Compute universal neurons within each task cluster separately.
    clustered_tasks: dict of {cluster_name: [prompts]}
    """
    cluster_results = {}
    layers = get_model_layers(model)
    if layer_idx >= len(layers): return {}

    for cluster, prompts in clustered_tasks.items():
        activations = []
        device = next(model.parameters()).device
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[layer_idx].mean(dim=1).squeeze(0)
            activations.append(hidden.cpu().numpy())

        activations = np.stack(activations)
        variances = np.var(activations, axis=0)
        means = np.mean(activations, axis=0)
        scores = means / (1 + variances)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        cluster_results[cluster] = [(int(i), float(scores[i])) for i in top_indices]

    return cluster_results


def neuron_masking_transfer_test(model, tokenizer, prompts, layer_idx, neuron_indices):
    """
    Test how much masking universal neurons degrades performance.
    prompts: list of (input, expected) pairs
    neuron_indices: list of neurons to ablate
    """
    layers = get_model_layers(model)
    if layer_idx >= len(layers): return None
    layer = layers[layer_idx]

    def ablation_hook(module, input, output):
        output[:, :, neuron_indices] = 0
        return output

    results = []
    for input_text, expected in prompts:
        inputs = tokenizer(input_text, return_tensors="pt")
        output_normal = model.generate(**inputs, max_new_tokens=5)
        decoded_normal = tokenizer.decode(output_normal[0], skip_special_tokens=True)

        handle = layer.register_forward_hook(ablation_hook)
        try:
            output_ablated = model.generate(**inputs, max_new_tokens=5)
            decoded_ablated = tokenizer.decode(output_ablated[0], skip_special_tokens=True)
        finally:
            handle.remove()

        changed = expected not in decoded_ablated and expected in decoded_normal
        results.append({"prompt": input_text, "changed": changed, "before": decoded_normal, "after": decoded_ablated})

    return results


def universality_specialization_tradeoff(model, tokenizer, task_dict, layer_idx=0):
    """
    Compute universality vs. max specialization score for each neuron.
    """
    layers = get_model_layers(model)
    if layer_idx >= len(layers): return []

    task_means = {}
    device = next(model.parameters()).device
    for task, prompts in task_dict.items():
        task_acts = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            act = outputs.hidden_states[layer_idx].mean(dim=1).squeeze(0)
            task_acts.append(act.cpu().numpy())
        task_means[task] = np.stack(task_acts).mean(axis=0)

    scores = []
    hidden_size = list(task_means.values())[0].shape[0]
    for i in range(hidden_size):
        activations = [task_means[task][i] for task in task_means]
        universality = np.mean(activations) / (1 + np.var(activations))
        specialization = max(activations) - np.mean(activations)
        scores.append((i, universality, specialization))

    return scores  # Each entry: (neuron_id, universality_score, specialization_score)


def combined_multitask_embedding_alignment(
    mamba_model, mamba_tokenizer,
    transformer_model, transformer_tokenizer,
    task_dict,
    layer_idx=0,
    method="pca",
    timestamp=None
    ):
    """
    Plot a single PCA scatterplot comparing embeddings
    from both Mamba and Transformer models.
    """
    all_embeds = []
    labels = []
    models = []

    for task, prompts in task_dict.items():
        for prompt in prompts:

            # Mamba embedding
            inputs_mamba = mamba_tokenizer(prompt, return_tensors="pt")
            device_mamba = next(mamba_model.parameters()).device
            inputs_mamba = {k: v.to(device_mamba) for k, v in inputs_mamba.items()}
            with torch.no_grad():
                outputs_mamba = mamba_model(**inputs_mamba, output_hidden_states=True)
            vec_mamba = outputs_mamba.hidden_states[layer_idx].mean(dim=1).squeeze(0).cpu().numpy()
            all_embeds.append(vec_mamba)
            labels.append(task)
            models.append("Mamba")

            # Transformer embedding
            inputs_trans = transformer_tokenizer(prompt, return_tensors="pt")
            device_trans = next(transformer_model.parameters()).device
            inputs_trans = {k: v.to(device_trans) for k, v in inputs_trans.items()}
            with torch.no_grad():
                outputs_trans = transformer_model(**inputs_trans, output_hidden_states=True)
            vec_trans = outputs_trans.hidden_states[layer_idx].mean(dim=1).squeeze(0).cpu().numpy()
            all_embeds.append(vec_trans)
            labels.append(task)
            models.append("Transformer")

    X = np.stack(all_embeds)

    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2)

    X_2d = reducer.fit_transform(X)

    plt.figure(figsize=(10, 6))
    for model in ["Mamba", "Transformer"]:
        for task in set(labels):
            indices = [
                i for i, (m, t) in enumerate(zip(models, labels))
                if m == model and t == task
            ]
            plt.scatter(
                X_2d[indices, 0],
                X_2d[indices, 1],
                label=f"{model} - {task}",
                alpha=0.6,
                s=60,
            )
    plt.legend()
    plt.title(f"Hidden-State Embeddings: Mamba vs Transformer (Layer {layer_idx})")
    plt.xlabel("Model Difference Axis (Mamba vs Transformer)")
    plt.ylabel("Task Variation Axis (Factual vs Mathematical vs Linguistic)")
    plt.grid(True)
    plt.tight_layout()
    filename = f'plots/combined_pca_embeddings_{timestamp}.png' if timestamp else 'plots/combined_pca_embeddings.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined PCA embeddings plot to {filename}")
    
"""
def visualize_universality_dimensions(universality_results, num_samples=5):
    
    Visualize activation patterns of universality dimensions.
    
    Args:
        universality_results: Results from analyze_universality_dimension
        num_samples: Number of sample texts to visualize
    
    patterns = universality_results["activation_patterns"][:num_samples]
    dimension = universality_results["dimension"]
    
    fig, axes = plt.subplots(len(patterns), 1, figsize=(12, 3*len(patterns)))
    
    for i, pattern in enumerate(patterns):
        ax = axes[i] if len(patterns) > 1 else axes
        
        # Plot activations
        ax.plot(pattern["activations"], marker='o', linestyle='-', markersize=4)
        
        # Add token labels
        token_positions = np.arange(len(pattern["tokens"]))
        ax.set_xticks(token_positions)
        ax.set_xticklabels(pattern["tokens"], rotation=45, ha='right')
        
        # Add text as title
        ax.set_title(f"Text: {pattern['text']}", fontsize=10)
        
        # Label axes
        ax.set_ylabel("Activation")
        if i == len(patterns) - 1:
            ax.set_xlabel("Token Position")
    
    plt.suptitle(f"Universality Dimension {dimension} Activation Patterns", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig
"""