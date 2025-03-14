"""
PCA-based coarse-graining method.
"""
import os
import numpy as np
import mdtraj as md
from sklearn.decomposition import PCA
import pickle

from ..base import CoarseGrainingMethod


class PCACoarseGraining(CoarseGrainingMethod):
    """
    Coarse-graining method based on Principal Component Analysis (PCA).
    
    This method reduces the dimensionality of each residue's coordinates
    using PCA, resulting in a compact representation that captures the
    most significant modes of variation.
    """
    
    def __init__(self, encoding_dim=1):
        """
        Initialize the PCA-based coarse-graining method.
        
        Args:
            encoding_dim (int): Number of principal components to use.
        """
        super().__init__(encoding_dim)
        self.pca_models = {}
        self.residue_means = {}
    
    def fit(self, trajectory):
        """
        Fit PCA models to each residue in the trajectory.
        
        Args:
            trajectory: MDTraj trajectory object
            
        Returns:
            cg_representation: Coarse-grained representation of the trajectory
        """
        n_frames = trajectory.n_frames
        n_residues = trajectory.topology.n_residues
        
        # Initialize our coarse-grained representation
        cg_representation = np.zeros((n_frames, n_residues, self.encoding_dim))
        
        # For each residue, we'll compute a representative value
        for res_idx in range(n_residues):
            # Get atom indices for this residue
            atom_indices = [atom.index for atom in trajectory.topology.residue(res_idx).atoms]
            
            if not atom_indices:
                continue
                
            # Store atom indices for reconstruction
            self.residue_indices[res_idx] = atom_indices
                
            # Extract coordinates for these atoms across all frames
            residue_coords = trajectory.xyz[:, atom_indices, :]
            
            # Reshape to 2D array: (n_frames, n_atoms*3)
            reshaped_coords = residue_coords.reshape(n_frames, -1)
            
            # Store mean for reconstruction
            self.residue_means[res_idx] = np.mean(reshaped_coords, axis=0)
            
            # Apply PCA to reduce dimensionality
            pca = PCA(n_components=self.encoding_dim)
            reduced_coords = pca.fit_transform(reshaped_coords)
            
            # Store the reduced representation
            cg_representation[:, res_idx, :] = reduced_coords
            
            # Store the PCA model
            self.pca_models[res_idx] = pca
        
        return cg_representation
    
    def transform(self, trajectory):
        """
        Apply the fitted PCA models to a new trajectory.
        
        Args:
            trajectory: MDTraj trajectory object
            
        Returns:
            cg_representation: Coarse-grained representation of the trajectory
        """
        n_frames = trajectory.n_frames
        n_residues = trajectory.topology.n_residues
        
        # Initialize our coarse-grained representation
        cg_representation = np.zeros((n_frames, n_residues, self.encoding_dim))
        
        # For each residue, we'll apply the pre-fitted PCA model
        for res_idx in range(n_residues):
            if res_idx not in self.pca_models:
                continue
                
            # Get atom indices for this residue
            atom_indices = self.residue_indices.get(res_idx)
            if not atom_indices:
                continue
                
            # Extract coordinates for these atoms across all frames
            residue_coords = trajectory.xyz[:, atom_indices, :]
            
            # Reshape to 2D array: (n_frames, n_atoms*3)
            reshaped_coords = residue_coords.reshape(n_frames, -1)
            
            # Apply PCA transformation
            reduced_coords = self.pca_models[res_idx].transform(reshaped_coords)
            
            # Store the reduced representation
            cg_representation[:, res_idx, :] = reduced_coords
        
        return cg_representation
    
    def inverse_transform(self, cg_representation, reference_traj):
        """
        Reconstruct a trajectory from its coarse-grained representation.
        
        Args:
            cg_representation: Coarse-grained representation
            reference_traj: Reference trajectory for topology and other properties
            
        Returns:
            reconstructed_traj: Reconstructed MDTraj trajectory
        """
        n_frames, n_residues, encoding_dim = cg_representation.shape
        reconstructed_xyz = np.zeros((n_frames, reference_traj.n_atoms, 3))
        
        # Copy the original trajectory for atoms we don't modify
        reconstructed_xyz[:] = reference_traj.xyz[:]
        
        for res_idx in range(n_residues):
            if res_idx not in self.pca_models or res_idx not in self.residue_indices:
                continue
                
            atom_indices = self.residue_indices[res_idx]
            
            # Get the number of atoms in this residue
            n_atoms = len(atom_indices)
            
            # Extract original coordinates for validation
            original_coords = reference_traj.xyz[:, atom_indices, :]
            
            # For each frame, reconstruct the atomic coordinates
            for frame_idx in range(n_frames):
                # Use the CG value to reconstruct the full representation
                cg_value = cg_representation[frame_idx, res_idx].reshape(1, -1)
                
                # Use the PCA model to transform back to original dimensions
                reconstructed_flat = self.pca_models[res_idx].inverse_transform(cg_value)
                
                # Add the mean back (this is crucial for correct scaling)
                reconstructed_flat += self.residue_means[res_idx]
                
                # Reshape to (n_atoms, 3)
                reconstructed_coords = reconstructed_flat.reshape(n_atoms, 3)
                
                # Verify scale matches original data
                orig_scale = np.sqrt(np.mean(np.sum(original_coords[frame_idx]**2, axis=1)))
                recon_scale = np.sqrt(np.mean(np.sum(reconstructed_coords**2, axis=1)))
                
                if recon_scale > 0:  # Avoid division by zero
                    # Apply scaling correction if needed
                    scale_factor = orig_scale / recon_scale
                    reconstructed_coords *= scale_factor
                
                # Store in the full reconstruction
                reconstructed_xyz[frame_idx, atom_indices, :] = reconstructed_coords
        
        # Create a new trajectory with the reconstructed coordinates
        reconstructed_traj = md.Trajectory(
            xyz=reconstructed_xyz,
            topology=reference_traj.topology,
            time=reference_traj.time,
            unitcell_lengths=reference_traj.unitcell_lengths,
            unitcell_angles=reference_traj.unitcell_angles
        )
        
        return reconstructed_traj
    
    def save(self, output_dir):
        """
        Save the PCA models and related data to disk.
        
        Args:
            output_dir: Directory to save the model
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metadata
        metadata = {
            'encoding_dim': self.encoding_dim,
            'method_type': 'PCA'
        }
        with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save PCA models
        with open(os.path.join(output_dir, 'pca_models.pkl'), 'wb') as f:
            pickle.dump(self.pca_models, f)
        
        # Save residue means
        with open(os.path.join(output_dir, 'residue_means.pkl'), 'wb') as f:
            pickle.dump(self.residue_means, f)
        
        # Save residue indices
        with open(os.path.join(output_dir, 'residue_indices.pkl'), 'wb') as f:
            pickle.dump(self.residue_indices, f)
    
    @classmethod
    def load(cls, model_dir):
        """
        Load a PCA coarse-graining model from disk.
        
        Args:
            model_dir: Directory containing the saved model
            
        Returns:
            method: Instance of PCACoarseGraining
        """
        # Load metadata
        with open(os.path.join(model_dir, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        # Create new instance
        instance = cls(encoding_dim=metadata['encoding_dim'])
        
        # Load PCA models
        with open(os.path.join(model_dir, 'pca_models.pkl'), 'rb') as f:
            instance.pca_models = pickle.load(f)
        
        # Load residue means
        with open(os.path.join(model_dir, 'residue_means.pkl'), 'rb') as f:
            instance.residue_means = pickle.load(f)
        
        # Load residue indices
        with open(os.path.join(model_dir, 'residue_indices.pkl'), 'rb') as f:
            instance.residue_indices = pickle.load(f)
        
        return instance
