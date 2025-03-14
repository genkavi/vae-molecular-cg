"""
Evaluation functions for molecular coarse-graining methods.
"""
import numpy as np
import matplotlib.pyplot as plt


def evaluate_reconstruction(original_traj, reconstructed_traj, plot=True, output_file=None):
    """
    Evaluate the quality of reconstruction by computing mass-weighted RMSD between 
    original and reconstructed trajectories.
    
    Args:
        original_traj: Original MDTraj trajectory
        reconstructed_traj: Reconstructed MDTraj trajectory
        plot: Whether to plot RMSD over time
        output_file: Path to save the plot
        
    Returns:
        mean_rmsd: Mean mass-weighted RMSD over all frames
        rmsd_per_frame: Mass-weighted RMSD for each frame
    """
    # Calculate mass-weighted RMSD for each frame
    n_frames = original_traj.n_frames
    rmsd_per_frame = np.zeros(n_frames)
    
    # Get atomic masses from the topology
    masses = np.array([atom.element.mass for atom in original_traj.topology.atoms])
    total_mass = np.sum(masses)
    mass_weights = masses / total_mass  # Normalize to get weights
    
    for i in range(n_frames):
        # Calculate squared displacement for each atom
        diff = original_traj.xyz[i] - reconstructed_traj.xyz[i]
        squared_diff = np.sum(diff**2, axis=1)
        
        # Apply mass weighting to the squared differences
        weighted_squared_diff = squared_diff * mass_weights
        
        # Calculate mass-weighted RMSD
        rmsd_per_frame[i] = np.sqrt(np.sum(weighted_squared_diff))
    
    mean_rmsd = np.mean(rmsd_per_frame)
    
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(rmsd_per_frame)
        plt.xlabel('Frame')
        plt.ylabel('Mass-weighted RMSD (nm)')
        plt.title(f'Mass-weighted Reconstruction RMSD (Mean: {mean_rmsd:.4f} nm)')
        plt.grid(True)
        
        if output_file:
            plt.savefig(output_file)
            plt.close()
        else:
            plt.show()
    
    return mean_rmsd, rmsd_per_frame


def compare_methods(trajectory, methods, method_names=None, encoding_dims=1):
    """
    Compare different coarse-graining methods on the same trajectory.
    
    Args:
        trajectory: MDTraj trajectory object
        methods: List of CoarseGrainingMethod objects
        method_names: List of names for each method (for the plot)
        encoding_dims: Encoding dimension(s) to use
        
    Returns:
        results: Dictionary with results for each method
    """
    if method_names is None:
        method_names = [f"Method {i+1}" for i in range(len(methods))]
    
    if not isinstance(encoding_dims, list):
        encoding_dims = [encoding_dims] * len(methods)
    
    results = {}
    rmsd_data = {}
    
    for i, (method, name, dim) in enumerate(zip(methods, method_names, encoding_dims)):
        print(f"Testing {name} with encoding_dim={dim}...")
        
        # Set encoding dimension
        method.encoding_dim = dim
        
        # Fit and transform
        cg_representation = method.fit(trajectory)
        
        # Reconstruct
        reconstructed_traj = method.inverse_transform(cg_representation, trajectory)
        
        # Evaluate
        mean_rmsd, rmsd_per_frame = evaluate_reconstruction(
            trajectory, reconstructed_traj, plot=False
        )
        
        # Store results
        results[name] = {
            'mean_rmsd': mean_rmsd,
            'rmsd_per_frame': rmsd_per_frame,
            'cg_representation': cg_representation
        }
        
        rmsd_data[name] = rmsd_per_frame
        
        print(f"  Mean RMSD: {mean_rmsd:.6f} nm")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    for name, rmsd in rmsd_data.items():
        plt.plot(rmsd, label=f"{name} (mean: {np.mean(rmsd):.4f} nm)")
    
    plt.xlabel('Frame')
    plt.ylabel('Mass-weighted RMSD (nm)')
    plt.title('Comparison of Reconstruction RMSD for Different Methods')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return results


def analyze_latent_space(cg_representation, residue_indices=None, method_name=None):
    """
    Analyze the latent space of a coarse-grained representation.
    
    Args:
        cg_representation: Coarse-grained representation
        residue_indices: List of residue indices to analyze (if None, first 5 residues)
        method_name: Name of the method (for the plot title)
        
    Returns:
        stats: Dictionary with statistics for each residue
    """
    n_frames, n_residues, encoding_dim = cg_representation.shape
    
    if residue_indices is None:
        residue_indices = range(min(5, n_residues))
    
    title_prefix = f"{method_name} " if method_name else ""
    
    # Compute statistics
    stats = {}
    
    for res_idx in residue_indices:
        if res_idx >= n_residues:
            continue
            
        latent_values = cg_representation[:, res_idx, :]
        
        # Compute statistics
        stats[res_idx] = {
            'mean': np.mean(latent_values, axis=0),
            'std': np.std(latent_values, axis=0),
            'min': np.min(latent_values, axis=0),
            'max': np.max(latent_values, axis=0),
            'range': np.max(latent_values, axis=0) - np.min(latent_values, axis=0)
        }
    
    # Visualize
    if encoding_dim == 1:
        # Plot histograms for 1D latent space
        plt.figure(figsize=(12, 6))
        
        for res_idx in residue_indices:
            if res_idx >= n_residues:
                continue
                
            plt.hist(cg_representation[:, res_idx, 0], alpha=0.5, bins=20, 
                     label=f'Residue {res_idx+1}')
        
        plt.xlabel('Latent Value')
        plt.ylabel('Frequency')
        plt.title(f'{title_prefix}Distribution of Latent Space Values')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    elif encoding_dim == 2:
        # Create scatter plots for 2D latent space
        plt.figure(figsize=(15, 10))
        
        for i, res_idx in enumerate(residue_indices):
            if res_idx >= n_residues or i >= 6:  # Limit to 6 plots
                continue
                
            plt.subplot(2, 3, i+1)
            plt.scatter(cg_representation[:, res_idx, 0], cg_representation[:, res_idx, 1], 
                        alpha=0.5, s=10)
            plt.title(f'Residue {res_idx+1}')
            plt.xlabel('Latent Dim 1')
            plt.ylabel('Latent Dim 2')
            plt.grid(True)
            
        plt.tight_layout()
        plt.suptitle(f'{title_prefix}2D Latent Space Visualization', y=1.02)
        plt.show()
    
    # Print statistics
    print(f"{title_prefix}Latent Space Statistics:")
    for res_idx, res_stats in stats.items():
        print(f"  Residue {res_idx+1}:")
        for stat_name, stat_value in res_stats.items():
            if encoding_dim == 1:
                print(f"    {stat_name}: {stat_value[0]:.4f}")
            else:
                print(f"    {stat_name}: {stat_value}")
    
    return stats
