import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Tuple
import os

def debug_model_structure(model, max_depth=3, current_depth=0, prefix=""):
    """Debug function to understand the model structure"""
    if current_depth >= max_depth:
        return
    
    for name, module in model.named_children():
        print(f"{prefix}{name}: {type(module).__name__}")
        if current_depth < max_depth - 1:
            debug_model_structure(module, max_depth, current_depth + 1, prefix + "  ")

def get_model_layers(model):
    """Get the layers from the model, handling different possible structures"""
    # Try different possible paths to access layers
    possible_paths = [
        lambda m: m.backbone.layers,  # Standard Mamba structure
        lambda m: m.model.layers,     # Alternative structure
        lambda m: m.layers,           # Direct access
        lambda m: m.transformer.h,    # Transformer-like structure
        lambda m: m.transformer.layers,
    ]
    
    for path_fn in possible_paths:
        try:
            layers = path_fn(model)
            if layers is not None:
                return layers
        except AttributeError:
            continue
    
    # If none work, return None and we'll handle it
    return None

"""
Utility functions for better plot display and management.
"""
# Configure matplotlib for file saving
def setup_matplotlib():
    """Setup matplotlib for file saving."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        plt.ioff()  # Disable interactive mode
        print("✅ Using matplotlib backend: Agg (file saving)")
        
    except Exception as e:
        print(f"Matplotlib setup failed: {e}")

def ensure_plot_display(title=None):
    """Save plots to files instead of displaying them."""
    try:
        # Create plots directory if it doesn't exist
        import os
        os.makedirs('plots', exist_ok=True)
        
        # Generate filename based on title or timestamp
        if title:
            filename = f"plots/{title.replace(' ', '_').replace('(', '').replace(')', '').replace(':', '')}.png"
        else:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"plots/plot_{timestamp}.png"
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {filename}")
        plt.close()  # Close the figure to free memory
    except Exception as e:
        print(f"❌ Error saving plot: {e}")
        plt.close()  # Close the figure even if saving fails