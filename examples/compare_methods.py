#!/usr/bin/env python
"""
Example script for comparing different coarse-graining methods on the same trajectory.
"""

import argparse
import os
import sys
import mdtraj as md

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import VAECoarseGraining, PCACoarseGraining
from src.evaluation import compare_methods

def main():
    parser = argparse.ArgumentParser(description='Compare Coarse-Graining Methods')
    
    parser.add_argument('trajectory', type=str, 
                        help='Path to trajectory file (.xtc, .dcd, etc.)')
    parser.add_argument('--topology', '-t', type=str, default=None,
                        help='Path to topology file (.pdb, .gro, etc.)')
    parser.add_argument('--encoding-dim', '-d', type=int, default=1,
                        help='Dimension of the latent space (default: 1)')
    parser.add_argument('--vae-beta', type=float, default=0.01,
                        help='Beta parameter for VAE (default: 0.01)')
    parser.add_argument('--vae-epochs', type=int, default=100,
                        help='Number of epochs for VAE training (default: 100)')
    
    args = parser.parse_args()
    
    # Load trajectory
    print(f"Loading trajectory from {args.trajectory}")
    if args.topology:
        print(f"Using topology from {args.topology}")
        traj = md.load(args.trajectory, top=args.topology)
    else:
        traj = md.load(args.trajectory)
    print(f"Loaded trajectory with {traj.n_frames} frames, {traj.n_atoms} atoms")
    
    # Create methods
    vae_method = VAECoarseGraining(
        encoding_dim=args.encoding_dim,
        beta=args.vae_beta,
        epochs=args.vae_epochs
    )
    
    pca_method = PCACoarseGraining(
        encoding_dim=args.encoding_dim
    )
    
    # Compare methods
    results = compare_methods(
        trajectory=traj,
        methods=[vae_method, pca_method],
        method_names=['VAE', 'PCA'],
        encoding_dims=args.encoding_dim
    )
    
    # Print summary of results
    print("\nSummary of Results:")
    for method_name, method_results in results.items():
        mean_rmsd = method_results['mean_rmsd']
        print(f"{method_name}: Mean RMSD = {mean_rmsd:.6f} nm")

if __name__ == '__main__':
    main()
