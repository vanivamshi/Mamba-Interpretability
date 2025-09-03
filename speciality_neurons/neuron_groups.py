# This file is kept as a placeholder - functions have been moved to separate files
# Import functions from their respective modules for backward compatibility

from .specialty_neurons import find_specialty_neurons_fixed
from .universality_analysis import analyze_universality_fixed
from .projection_neurons import find_projection_dominant_neurons_fixed

__all__ = [
    'find_specialty_neurons_fixed',
    'analyze_universality_fixed', 
    'find_projection_dominant_neurons_fixed'
]
