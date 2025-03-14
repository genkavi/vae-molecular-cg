"""
VAE-based coarse-graining method.
"""
import os
import numpy as np
import mdtraj as md
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle

from ..base import CoarseGrainingMethod


class VAE(nn.Module):
    """
    Variational Autoencoder for molecular coarse-graining.
    
    This VAE maps atomic coordinates to a lower-dimensional latent space
    and can reconstruct the original coordinates from this representation.
    """
    def __init__(self, input_dim, encoding_dim=1, hidden_dim=None):
        super(VAE, self).__init__()
        
        # Set hidden dimension if not provided
        if hidden_dim is None:
            hidden_dim = max(input_dim // 2, 8)
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, encoding_dim)
        self.fc_logvar = nn.Linear(hidden_dim, encoding_dim)
        
        # Decoder
        self.fc3 = nn.Linear(encoding_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
        # Store dimensions for sampling
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        
    def encode(self, x):
        """
        Encode input coordinates to latent space parameters.
        
        Args:
            x: Input coordinates tensor
            
        Returns:
            tuple: (mean, log-variance) of the latent space distribution
        """
        h1 = F.relu(self.fc1(x))
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Perform the reparameterization trick for VAE sampling.
        
        Args:
            mu: Mean of the latent distribution
            logvar: Log-variance of the latent distribution
            
        Returns:
            tensor: Sampled point from the latent space
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """
        Decode from latent space to reconstructed coordinates.
        
        Args:
            z: Latent space vector
            
        Returns:
            tensor: Reconstructed coordinates
        """
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)
    
    def forward(self, x):
        """
        Forward pass through the VAE.
        
        Args:
            x: Input coordinates tensor
            
        Returns:
            tuple: (reconstructed_x, mean, log-variance)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def get_latent(self, x):
        """
        Get the latent space representation without sampling.
        
        Args:
            x: Input coordinates tensor
            
        Returns:
            tensor: Mean of the latent distribution
        """
        mu, _ = self.encode(x)
        return mu


def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    Loss function for VAE with beta weighting for KL divergence term.
    
    Args:
        recon_x: Reconstructed data
        x: Original data
        mu: Mean of the latent distribution
        logvar: Log-variance of the latent distribution
        beta: Weight for the KL divergence term
        
    Returns:
        tuple: (total_loss, reconstruction_loss, kl_divergence)
    """
    # Reconstruction loss (using MSE for coordinate data)
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss with beta weighting for KL term
    return MSE + beta * KLD, MSE, KLD


class VAECoarseGraining(CoarseGrainingMethod):
    """
    Coarse-graining method based on Variational Autoencoders (VAE).
    
    This method uses a VAE to learn a low-dimensional embedding for each
    residue, allowing for a more expressive and potentially more accurate
    representation than linear methods like PCA.
    """
    
    def __init__(self, encoding_dim=1, hidden_dim=None, beta=0.01, 
                 epochs=200, batch_size=64, lr=0.001):
        """
        Initialize the VAE-based coarse-graining method.
        
        Args:
            encoding_dim (int): Dimension of the latent space.
            hidden_dim (int): Dimension of the hidden layer. If None, it's set automatically.
            beta (float): Weight of the KL divergence term (lower = stronger reconstruction).
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            lr (float): Learning rate.
        """
        super().__init__(encoding_dim)
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.vae_models = {}
        self.residue_indices = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def fit(self, trajectory):
        """
        Fit VAE models to each residue in the trajectory.
        
        Args:
            trajectory: MDTraj trajectory object
            
        Returns:
            cg_representation: Coarse-grained representation of the trajectory
        """
        n_frames = trajectory.n_frames
        n_residues = trajectory.topology.n_residues
        
        print(f"Using device: {self.device}")
        
        # Initialize our coarse-grained representation
        cg_representation = np.zeros((n_frames, n_residues, self.encoding_dim))
        
        # For each residue, train a VAE and extract the representation
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
            
            # Convert to PyTorch tensor
            coords_tensor = torch.tensor(reshaped_coords, dtype=torch.float32).to(self.device)
            
            # Normalize the data to improve training stability
            coords_mean = coords_tensor.mean(dim=0, keepdim=True)
            coords_std = coords_tensor.std(dim=0, keepdim=True) + 1e-6  # Avoid division by zero
            coords_normalized = (coords_tensor - coords_mean) / coords_std
            
            # Create dataset and dataloader
            dataset = TensorDataset(coords_normalized, coords_normalized)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            # Create VAE model for this residue
            input_dim = reshaped_coords.shape[1]
            hidden_dim = self.hidden_dim if self.hidden_dim is not None else max(input_dim, 16)
            model = VAE(input_dim, encoding_dim=self.encoding_dim, hidden_dim=hidden_dim).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            
            # Track loss for monitoring
            train_losses = []
            recon_losses = []
            kl_losses = []
            
            # Training loop
            for epoch in range(self.epochs):
                model.train()
                epoch_loss = 0
                epoch_recon_loss = 0
                epoch_kl_loss = 0
                
                for batch_idx, (data, _) in enumerate(dataloader):
                    data = data.to(self.device)
                    
                    # Forward pass
                    recon_batch, mu, logvar = model(data)
                    
                    # Calculate loss
                    loss, recon_loss, kl_loss = vae_loss_function(
                        recon_batch, data, mu, logvar, beta=self.beta
                    )
                    
                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Accumulate losses
                    epoch_loss += loss.item()
                    epoch_recon_loss += recon_loss.item()
                    epoch_kl_loss += kl_loss.item()
                
                # Record average losses
                avg_loss = epoch_loss / len(dataloader.dataset)
                avg_recon_loss = epoch_recon_loss / len(dataloader.dataset)
                avg_kl_loss = epoch_kl_loss / len(dataloader.dataset)
                
                train_losses.append(avg_loss)
                recon_losses.append(avg_recon_loss)
                kl_losses.append(avg_kl_loss)
                
                # Print progress
                if (epoch + 1) % 20 == 0 or epoch == 0:
                    print(f"Residue {res_idx+1}/{n_residues}, Epoch [{epoch+1}/{self.epochs}], "
                          f"Loss: {avg_loss:.6f}, Recon: {avg_recon_loss:.6f}, KL: {avg_kl_loss:.6f}")
            
            # Extract coarse-grained representation
            model.eval()
            with torch.no_grad():
                mu, _ = model.encode(coords_normalized)
                cg_representation[:, res_idx, :] = mu.cpu().numpy()
            
            # Store model and normalization parameters for reconstruction
            self.vae_models[res_idx] = {
                'model': model,
                'mean': coords_mean.cpu(),
                'std': coords_std.cpu()
            }
        
        return cg_representation
    
    def transform(self, trajectory):
        """
        Apply the fitted VAE models to a new trajectory.
        
        Args:
            trajectory: MDTraj trajectory object
            
        Returns:
            cg_representation: Coarse-grained representation of the trajectory
        """
        n_frames = trajectory.n_frames
        n_residues = trajectory.topology.n_residues
        
        # Initialize our coarse-grained representation
        cg_representation = np.zeros((n_frames, n_residues, self.encoding_dim))
        
        for res_idx in range(n_residues):
            if res_idx not in self.vae_models:
                continue
                
            # Get atom indices for this residue
            atom_indices = self.residue_indices.get(res_idx)
            if not atom_indices:
                continue
                
            # Extract coordinates for these atoms across all frames
            residue_coords = trajectory.xyz[:, atom_indices, :]
            
            # Reshape to 2D array: (n_frames, n_atoms*3)
            reshaped_coords = residue_coords.reshape(n_frames, -1)
            
            # Convert to PyTorch tensor
            coords_tensor = torch.tensor(reshaped_coords, dtype=torch.float32).to(self.device)
            
            # Get normalization parameters
            coords_mean = self.vae_models[res_idx]['mean'].to(self.device)
            coords_std = self.vae_models[res_idx]['std'].to(self.device)
            
            # Normalize
            coords_normalized = (coords_tensor - coords_mean) / coords_std
            
            # Apply VAE encoding
            model = self.vae_models[res_idx]['model']
            model.eval()
            with torch.no_grad():
                mu, _ = model.encode(coords_normalized)
                cg_representation[:, res_idx, :] = mu.cpu().numpy()
        
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
            if res_idx not in self.vae_models or res_idx not in self.residue_indices:
                continue
                
            atom_indices = self.residue_indices[res_idx]
            model_dict = self.vae_models[res_idx]
            model = model_dict['model']
            coords_mean = model_dict['mean'].to(self.device)
            coords_std = model_dict['std'].to(self.device)
            
            # Extract all coarse-grained values for this residue
            cg_values = cg_representation[:, res_idx, :]
            cg_tensor = torch.tensor(cg_values, dtype=torch.float32).to(self.device)
            
            # Reconstruct all frames at once
            model.eval()
            with torch.no_grad():
                # Decode from latent space
                reconstructed_normalized = model.decode(cg_tensor)
                
                # Denormalize
                reconstructed_flat = reconstructed_normalized * coords_std + coords_mean
                reconstructed_flat = reconstructed_flat.cpu().numpy()
            
            # Reshape and store in the full reconstruction
            n_atoms = len(atom_indices)
            for frame_idx in range(n_frames):
                reconstructed_coords = reconstructed_flat[frame_idx].reshape(n_atoms, 3)
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
        Save the VAE models and related data to disk.
        
        Args:
            output_dir: Directory to save the model
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metadata
        metadata = {
            'encoding_dim': self.encoding_dim,
            'hidden_dim': self.hidden_dim,
            'beta': self.beta,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'method_type': 'VAE'
        }
        with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save residue indices
        with open(os.path.join(output_dir, 'residue_indices.pkl'), 'wb') as f:
            pickle.dump(self.residue_indices, f)
        
        # Save each model separately
        models_dir = os.path.join(output_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        for res_idx, model_dict in self.vae_models.items():
            model_path = os.path.join(models_dir, f'model_res_{res_idx}.pt')
            torch.save({
                'model_state_dict': model_dict['model'].state_dict(),
                'mean': model_dict['mean'],
                'std': model_dict['std'],
                'input_dim': model_dict['model'].input_dim,
                'encoding_dim': model_dict['model'].encoding_dim
            }, model_path)
    
    @classmethod
    def load(cls, model_dir):
        """
        Load a VAE coarse-graining model from disk.
        
        Args:
            model_dir: Directory containing the saved model
            
        Returns:
            method: Instance of VAECoarseGraining
        """
        # Load metadata
        with open(os.path.join(model_dir, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        # Create new instance
        instance = cls(
            encoding_dim=metadata['encoding_dim'],
            hidden_dim=metadata['hidden_dim'],
            beta=metadata['beta'],
            epochs=metadata['epochs'],
            batch_size=metadata['batch_size'],
            lr=metadata['lr']
        )
        
        # Load residue indices
        with open(os.path.join(model_dir, 'residue_indices.pkl'), 'rb') as f:
            instance.residue_indices = pickle.load(f)
        
        # Load models
        models_dir = os.path.join(model_dir, 'models')
        device = instance.device
        
        for model_file in os.listdir(models_dir):
            if not model_file.startswith('model_res_') or not model_file.endswith('.pt'):
                continue
                
            # Extract residue index from filename
            res_idx = int(model_file.split('_')[2].split('.')[0])
            
            # Load checkpoint
            checkpoint = torch.load(os.path.join(models_dir, model_file), map_location=device)
            
            # Create model
            model = VAE(
                input_dim=checkpoint['input_dim'],
                encoding_dim=checkpoint['encoding_dim']
            ).to(device)
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Store in the models dictionary
            instance.vae_models[res_idx] = {
                'model': model,
                'mean': checkpoint['mean'].to(device),
                'std': checkpoint['std'].to(device)
            }
        
        return instance
