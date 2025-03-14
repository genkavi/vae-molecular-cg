"""
Module containing different molecular coarse-graining model implementations.
"""

from .vae_model import VAECoarseGraining
from .pca_model import PCACoarseGraining

__all__ = [
    'VAECoarseGraining',
    'PCACoarseGraining'
]
