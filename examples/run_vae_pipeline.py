#!/usr/bin/env python
"""
Example script for running the VAE coarse-graining pipeline on a trajectory file.
"""

import argparse
import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import vae_coarse_graining_pipeline

def main():
    parser = argparse.ArgumentParser(description='VAE Coarse-Graining Pipeline')
    
    parser.add_argument('trajectory', type=str, 
                        help='Path to trajectory file (.xtc, .dcd, etc.)')
    parser.add_argument('--topology', '-t', type=str, default=None,
                        help='Path to topology file (.pdb, .gro, etc.)')
    parser.add_argument('--output-dir', '-o', type=str, default=None, 
                        help='Directory for output files (default: auto-generated)')
    parser.add_argument('--encoding-dim', '-d', type=int, default=1,
                        help='Dimension of the latent space (default: 1)')
    parser.add_argument('--beta', '-b', type=float, default=0.01,
                        help='Weight of the KL divergence term (default: 0.01)')
    parser.add_argument('--epochs', '-e', type=int, default=200,
                        help='Number of training epochs (default: 200)')
    parser.add_argument('--batch-size', '-bs', type=int, default=64,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--no-evaluate', action='store_false', dest='evaluate',
                        help='Skip reconstruction evaluation')
    
    args = parser.parse_args()
    
    # Call the pipeline
    vae_coarse_graining_pipeline(
        trajectory_file=args.trajectory,
        topology_file=args.topology,
        output_dir=args.output_dir,
        encoding_dim=args.encoding_dim,
        beta=args.beta,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        evaluate=args.evaluate
    )

if __name__ == '__main__':
    main()
