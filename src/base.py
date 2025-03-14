"""
Base classes for molecular coarse-graining methods.
"""
from abc import ABC, abstractmethod
import numpy as np
import mdtraj as md


class CoarseGrainingMethod(ABC):
    """
    Abstract base class for all coarse-graining methods.
    
    This class defines the interface that all coarse-graining methods must implement.
    """
    
    def __init__(self, encoding_dim=1):
        """
        Initialize the coarse-graining method.
        
        Args:
            encoding_dim (int): Dimension of the coarse-grained representation.
        """
        self.encoding_dim = encoding_dim
        self.models = {}
        self.residue_indices = {}
    
    @abstractmethod
    def fit(self, trajectory):
        """
        Fit the coarse-graining method to a trajectory.
        
        Args:
            trajectory: MDTraj trajectory object
            
        Returns:
            cg_representation: Coarse-grained representation of the trajectory
        """
        pass
    
    @abstractmethod
    def transform(self, trajectory):
        """
        Apply the coarse-graining to a trajectory.
        
        Args:
            trajectory: MDTraj trajectory object
            
        Returns:
            cg_representation: Coarse-grained representation of the trajectory
        """
        pass
    
    @abstractmethod
    def inverse_transform(self, cg_representation, reference_traj):
        """
        Reconstruct a trajectory from its coarse-grained representation.
        
        Args:
            cg_representation: Coarse-grained representation
            reference_traj: Reference trajectory for topology and other properties
            
        Returns:
            reconstructed_traj: Reconstructed MDTraj trajectory
        """
        pass
    
    def fit_transform(self, trajectory):
        """
        Convenience method to fit and transform in one step.
        
        Args:
            trajectory: MDTraj trajectory object
            
        Returns:
            cg_representation: Coarse-grained representation of the trajectory
        """
        return self.fit(trajectory)
    
    def save(self, output_dir):
        """
        Save the coarse-graining model to disk.
        
        Args:
            output_dir: Directory to save the model
        """
        pass
    
    @classmethod
    def load(cls, model_dir):
        """
        Load a coarse-graining model from disk.
        
        Args:
            model_dir: Directory containing the saved model
            
        Returns:
            method: Instance of the coarse-graining method
        """
        pass
