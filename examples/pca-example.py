#!/usr/bin/env python
"""
Example script for running the PCA coarse-graining pipeline on a trajectory file.
"""

import argparse
import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import pca_coarse_graining_pipeline

def main():
    parser = argparse.ArgumentParser(description='PCA Coarse-Graining Pipeline')
    
    parser.add_argument('trajectory', type=str, 
                        help='Path to trajectory file (any format supported by MDTraj)')
    parser.add_argument('--output-dir', '-o', type=str, default=None, 
                        help='Directory for output files (default: auto-generated)')
    parser.add_argument('--encoding-dim', '-d', type=int, default=1,
                        help='Dimension of the latent space (default: 1)')
    parser.add_argument('--no-evaluate', action='store_false', dest='evaluate',
                        help='Skip reconstruction evaluation')
    
    args = parser.parse_args()
    
    # Call the pipeline
    pca_coarse_graining_pipeline(
        trajectory_file=args.trajectory,
        output_dir=args.output_dir,
        encoding_dim=args.encoding_dim,
        evaluate=args.evaluate
    )

if __name__ == '__main__':
    main()
