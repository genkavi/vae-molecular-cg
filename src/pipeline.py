"""
Pipeline for molecular coarse-graining.
"""
import os
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt

from .models import VAECoarseGraining, PCACoarseGraining
from .evaluation import evaluate_reconstruction


def coarse_graining_pipeline(trajectory_file, topology_file=None, method='vae', output_dir=None, 
                           encoding_dim=1, evaluate=True, **method_kwargs):
    """
    Pipeline for molecular coarse-graining using different methods.
    
    Args:
        trajectory_file: Path to trajectory file (.xtc, .dcd, .trr, etc.)
        topology_file: Path to topology file (.pdb, .gro, etc.). Not needed if trajectory includes topology.
        method: Coarse-graining method to use ('vae' or 'pca')
        output_dir: Directory for output files
        encoding_dim: Dimension of the latent space
        evaluate: Whether to evaluate reconstruction quality
        **method_kwargs: Additional keyword arguments for the specific method
        
    Returns:
        cg_representation: Coarse-grained representation
        reconstructed_traj: Reconstructed trajectory
        evaluation_results: Dictionary with evaluation results
    """
    # Load trajectory
    print(f"Loading trajectory from {trajectory_file}")
    if topology_file:
        print(f"Using topology from {topology_file}")
        traj = md.load(trajectory_file, top=topology_file)
    else:
        # For file formats that include topology (like PDB)
        traj = md.load(trajectory_file)
    print(f"Loaded trajectory with {traj.n_frames} frames, {traj.n_atoms} atoms, {traj.topology.n_residues} residues")
    
    # Set output directory if not provided
    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(trajectory_file))[0]
        output_dir = f"{base_name}_{method}_cg"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize coarse-graining method
    if method.lower() == 'vae':
        print(f"Using VAE coarse-graining with encoding_dim={encoding_dim}")
        cg_method = VAECoarseGraining(encoding_dim=encoding_dim, **method_kwargs)
    elif method.lower() == 'pca':
        print(f"Using PCA coarse-graining with encoding_dim={encoding_dim}")
        cg_method = PCACoarseGraining(encoding_dim=encoding_dim, **method_kwargs)
    else:
        raise ValueError(f"Unsupported method: {method}. Choose 'vae' or 'pca'.")
    
    # Perform coarse-graining
    print("Starting coarse-graining...")
    cg_representation = cg_method.fit(traj)
    
    # Save coarse-grained representation
    cg_file = os.path.join(output_dir, "cg_representation.npy")
    np.save(cg_file, cg_representation)
    print(f"Saved coarse-grained representation to {cg_file}")
    
    # Save method
    model_dir = os.path.join(output_dir, "model")
    cg_method.save(model_dir)
    print(f"Saved coarse-graining model to {model_dir}")
    
    # Visualize latent space
    visualize_latent_space(cg_representation, 
                          os.path.join(output_dir, "latent_space.png"))
    print(f"Saved latent space visualization")
    
    # Reconstruct trajectory
    print("Reconstructing trajectory from coarse-grained representation")
    reconstructed_traj = cg_method.inverse_transform(cg_representation, traj)
    
    # Save reconstructed trajectory
    reconstructed_file = os.path.join(output_dir, "reconstructed.pdb")
    reconstructed_traj.save(reconstructed_file)
    print(f"Saved reconstructed trajectory to {reconstructed_file}")
    
    # Evaluate reconstruction if requested
    evaluation_results = {}
    if evaluate:
        print("Evaluating reconstruction quality")
        mean_rmsd, rmsd_per_frame = evaluate_reconstruction(
            traj, reconstructed_traj, plot=True, 
            output_file=os.path.join(output_dir, "rmsd_plot.png")
        )
        
        evaluation_results = {
            'mean_rmsd': mean_rmsd,
            'rmsd_per_frame': rmsd_per_frame
        }
        
        print(f"Mean reconstruction RMSD: {mean_rmsd:.6f} nm")
        
        # Save evaluation results
        np.save(os.path.join(output_dir, "rmsd_per_frame.npy"), rmsd_per_frame)
        
        # Save summary to text file
        with open(os.path.join(output_dir, "evaluation_summary.txt"), 'w') as f:
            f.write(f"Trajectory: {trajectory_file}\n")
            f.write(f"Coarse-graining method: {method}\n")
            f.write(f"Encoding dimension: {encoding_dim}\n")
            for key, value in method_kwargs.items():
                f.write(f"{key}: {value}\n")
            f.write(f"Mean RMSD: {mean_rmsd:.6f} nm\n")
            f.write(f"Min RMSD: {np.min(rmsd_per_frame):.6f} nm\n")
            f.write(f"Max RMSD: {np.max(rmsd_per_frame):.6f} nm\n")
    
    return cg_representation, reconstructed_traj, evaluation_results


def visualize_latent_space(cg_representation, output_file=None):
    """
    Visualize the distribution of values in the latent space.
    
    Args:
        cg_representation: Coarse-grained representation 
        output_file: Path to save the plot (if None, display instead)
    """
    n_frames, n_residues, encoding_dim = cg_representation.shape
    
    plt.figure(figsize=(12, 8))
    
    if encoding_dim == 1:
        # For 1D latent space, plot distribution for each residue
        for res_idx in range(min(n_residues, 10)):  # Limit to first 10 residues
            plt.hist(cg_representation[:, res_idx, 0], alpha=0.5, bins=20, 
                     label=f'Residue {res_idx+1}')
        
        plt.xlabel('Latent Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of Latent Space Values for First 10 Residues')
        plt.legend()
        
    elif encoding_dim == 2:
        # For 2D latent space, create scatter plots
        plt.figure(figsize=(15, 12))
        
        # We'll create a grid of scatter plots for the first 16 residues
        rows = min(4, n_residues)
        cols = min(4, n_residues)
        
        for i in range(rows * cols):
            if i >= n_residues:
                break
                
            plt.subplot(rows, cols, i+1)
            plt.scatter(cg_representation[:, i, 0], cg_representation[:, i, 1], 
                        alpha=0.5, s=10)
            plt.title(f'Residue {i+1}')
            plt.xlabel('Latent Dim 1')
            plt.ylabel('Latent Dim 2')
            
        plt.tight_layout()
    
    else:
        # For higher dimensions, we can use PCA to visualize
        from sklearn.decomposition import PCA
        
        # Reshape to combine all residues
        reshaped = cg_representation.reshape(-1, encoding_dim)
        
        # Apply PCA
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(reshaped)
        
        # Plot
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.5, s=10)
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title(f'PCA Projection of {encoding_dim}D Latent Space')
    
    if output_file:
        plt.savefig(output_file)
        plt.close()
    else:
        plt.show()


# Aliases for backward compatibility
vae_coarse_graining_pipeline = lambda trajectory_file, **kwargs: coarse_graining_pipeline(
    trajectory_file, method='vae', **kwargs
)

pca_coarse_graining_pipeline = lambda trajectory_file, **kwargs: coarse_graining_pipeline(
    trajectory_file, method='pca', **kwargs
)
