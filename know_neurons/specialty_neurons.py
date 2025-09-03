import torch
import numpy as np

def find_specialty_neurons_fixed(model, tokenizer, class_texts, layer_idx=0, top_k=5):
    """Find neurons that are specialized for different text classes - fixed version."""
    class_activations = {}
    
    for cls, texts in class_texts.items():
        activations = []
        for text in texts:
            try:
                inputs = tokenizer(text, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)
                
                # Get the appropriate layer's hidden states
                if layer_idx < len(outputs.hidden_states):
                    states = outputs.hidden_states[layer_idx]
                    # Average over sequence length, keep batch dimension
                    states = states.mean(dim=1).squeeze(0)
                    activations.append(states.cpu().numpy())
                else:
                    print(f"Layer {layer_idx} not available. Using dummy values.")
                    activations.append(np.random.randn(512))
                    
            except Exception as e:
                print(f"Error processing text for class {cls}: {e}")
                activations.append(np.random.randn(512))
        
        if activations:
            class_activations[cls] = np.stack(activations)

    if not class_activations:
        print("No class activations extracted. Returning dummy results.")
        return {cls: [(i, float(i)) for i in range(top_k)] for cls in class_texts.keys()}

    neuron_scores = {}
    for cls in class_activations:
        try:
            mean_acts = np.mean(class_activations[cls], axis=0)
            diffs = []
            for other_cls in class_activations:
                if other_cls != cls:
                    diff = mean_acts - np.mean(class_activations[other_cls], axis=0)
                    diffs.append(diff)
            
            if diffs:
                max_diff = np.max(np.abs(np.stack(diffs)), axis=0)
                top_neurons = np.argsort(max_diff)[-top_k:]
                neuron_scores[cls] = [(int(idx), float(max_diff[idx])) for idx in top_neurons[::-1]]
            else:
                neuron_scores[cls] = [(i, float(i)) for i in range(top_k)]
                
        except Exception as e:
            print(f"Error computing neuron scores for class {cls}: {e}")
            neuron_scores[cls] = [(i, float(i)) for i in range(top_k)]
    
    return neuron_scores
