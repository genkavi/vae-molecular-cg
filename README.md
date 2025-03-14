# VAE Molecular Coarse-Graining

A Python package for coarse-graining molecular dynamics trajectories using variational autoencoders (VAEs), principal component analysis (PCA), and other methods.

## Features

- Multiple coarse-graining methods:
  - Variational Autoencoder (VAE) based coarse-graining
  - Principal Component Analysis (PCA) based coarse-graining
- Customizable encoding dimensions
- Residue-level coarse-graining
- Trajectory reconstruction
- Evaluation tools
- Easy comparison between methods

## Installation

```bash
git clone https://github.com/yourusername/vae-molecular-cg.git
cd vae-molecular-cg
pip install -e .
```

## Quick Start

### VAE Coarse-Graining

```bash
python examples/run_vae_pipeline.py your_trajectory.xtc --topology your_topology.pdb --encoding-dim 1 --beta 0.01
```

### PCA Coarse-Graining

```bash
python examples/run_pca_pipeline.py your_trajectory.xtc --topology your_topology.pdb --encoding-dim 1
```

### Compare Methods

```bash
python examples/compare_methods.py your_trajectory.xtc --topology your_topology.pdb --encoding-dim 2
```

## Usage Guide

### Command Line Arguments

#### VAE Pipeline

```
usage: run_vae_pipeline.py [-h] [--output-dir OUTPUT_DIR] [--encoding-dim ENCODING_DIM]
                         [--beta BETA] [--epochs EPOCHS] [--batch-size BATCH_SIZE]
                         [--learning-rate LEARNING_RATE] [--no-evaluate]
                         trajectory

VAE Coarse-Graining Pipeline

positional arguments:
  trajectory            Path to trajectory file (any format supported by MDTraj)

optional arguments:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Directory for output files (default: auto-generated)
  --encoding-dim ENCODING_DIM, -d ENCODING_DIM
                        Dimension of the latent space (default: 1)
  --beta BETA, -b BETA  Weight of the KL divergence term (default: 0.01)
  --epochs EPOCHS, -e EPOCHS
                        Number of training epochs (default: 200)
  --batch-size BATCH_SIZE, -bs BATCH_SIZE
                        Batch size for training (default: 64)
  --learning-rate LEARNING_RATE, -lr LEARNING_RATE
                        Learning rate (default: 0.001)
  --no-evaluate         Skip reconstruction evaluation
```

#### PCA Pipeline

```
usage: run_pca_pipeline.py [-h] [--output-dir OUTPUT_DIR] [--encoding-dim ENCODING_DIM]
                         [--no-evaluate]
                         trajectory

PCA Coarse-Graining Pipeline

positional arguments:
  trajectory            Path to trajectory file (any format supported by MDTraj)

optional arguments:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Directory for output files (default: auto-generated)
  --encoding-dim ENCODING_DIM, -d ENCODING_DIM
                        Dimension of the latent space (default: 1)
  --no-evaluate         Skip reconstruction evaluation
```

### Python API

You can also use the package directly in your Python code:

```python
import mdtraj as md
from vae_molecular_cg.models import VAECoarseGraining, PCACoarseGraining
from vae_molecular_cg.evaluation import compare_methods

# Load trajectory
traj = md.load('trajectory.pdb')

# Create coarse-graining methods
vae_cg = VAECoarseGraining(encoding_dim=2, beta=0.01, epochs=100)
pca_cg = PCACoarseGraining(encoding_dim=2)

# Compare methods
results = compare_methods(
    trajectory=traj,
    methods=[vae_cg, pca_cg],
    method_names=['VAE', 'PCA']
)

# Use a specific method
cg_representation = vae_cg.fit(traj)
reconstructed_traj = vae_cg.inverse_transform(cg_representation, traj)

# Save the model for later use
vae_cg.save('vae_model')

# Load a saved model
loaded_vae = VAECoarseGraining.load('vae_model')
```

## Output Files

The coarse-graining pipeline generates the following files:

- `cg_representation.npy`: Coarse-grained representation (numpy array)
- `reconstructed.pdb`: Reconstructed trajectory
- `model/`: Directory containing saved model parameters
- `latent_space.png`: Visualization of the latent space
- `rmsd_plot.png`: RMSD plot (if evaluation is enabled)
- `rmsd_per_frame.npy`: RMSD values for each frame
- `evaluation_summary.txt`: Summary of evaluation results

## How It Works

### VAE-based Coarse-Graining

The VAE method trains a variational autoencoder on each residue to learn a low-dimensional representation of the atomic coordinates. The encoder maps the high-dimensional atomic coordinates to a low-dimensional latent space, and the decoder reconstructs the original coordinates from this representation.

### PCA-based Coarse-Graining

The PCA method uses principal component analysis to reduce the dimensionality of each residue's coordinates. This is a linear dimensionality reduction technique that finds the directions of maximum variance in the data.

### Reconstruction

Both methods can reconstruct the original atomic coordinates from the coarse-grained representation, allowing for evaluation of the quality of the coarse-graining.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
