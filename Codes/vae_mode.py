# vae_model.py
"""
Variational Autoencoder for geological uncertainty modeling
Learns latent representation and generates new scenarios
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from config import VAEParameters, MODEL_DIR, DEVICE
from data_generator import MiningDataGenerator

class GeologicalDataset(Dataset):
    """PyTorch Dataset for geological features"""
    
    def __init__(self, data_dict):
        """
        Combine features: [grade, rock_type, alteration, structural, distance]
        """
        features = np.column_stack([
            data_dict['base_grades'],
            data_dict['base_rock_types'],
            data_dict['geological_features']['alteration_intensity'],
            data_dict['geological_features']['structural_density'],
            data_dict['geological_features']['distance_to_intrusion']
        ])
        self.data = torch.FloatTensor(features)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class GeologicalVAE(nn.Module):
    """
    Variational Autoencoder for geological scenario generation
    Architecture: Encoder -> Latent Space -> Decoder
    """
    
    def __init__(self, params: VAEParameters):
        super(GeologicalVAE, self).__init__()
        self.params = params
        
        # Encoder network
        encoder_layers = []
        prev_dim = params.input_dim
        
        for h_dim in params.hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(params.hidden_dims[-1], params.latent_dim)
        self.fc_logvar = nn.Linear(params.hidden_dims[-1], params.latent_dim)
        
        # Decoder network
        decoder_layers = []
        prev_dim = params.latent_dim
        
        for h_dim in reversed(params.hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        
        decoder_layers.append(nn.Linear(params.hidden_dims[0], params.input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Encode input to latent distribution parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling from latent space"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to reconstruction"""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass through VAE"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar, beta=0.5, lambda_geo=0.1):
    """
    VAE loss function with:
    - Reconstruction loss (MSE)
    - KL divergence (regularization)
    - Geological continuity constraint
    """
    # Reconstruction loss
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Geological continuity constraint (smooth grade transitions)
    if len(recon_x) > 1:
        # Grade column (index 0)
        grade_diff = torch.mean((recon_x[1:, 0] - recon_x[:-1, 0]).pow(2))
        geo_loss = lambda_geo * grade_diff
    else:
        geo_loss = torch.tensor(0.0)
    
    total_loss = recon_loss + beta * kl_loss + geo_loss
    
    return total_loss, recon_loss, kl_loss, geo_loss

def train_vae(model, train_loader, params: VAEParameters, device=DEVICE):
    """Train VAE model"""
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    model.train()
    model.to(device)
    
    losses = {
        'total': [],
        'recon': [],
        'kl': [],
        'geo': []
    }
    
    print("=" * 80)
    print(f"TRAINING VAE ON {device}")
    print("=" * 80)
    
    for epoch in range(params.epochs):
        total_loss = 0
        total_recon = 0
        total_kl = 0
        total_geo = 0
        
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, logvar = model(data)
            
            # Compute losses
            loss, recon_loss, kl_loss, geo_loss = vae_loss_function(
                recon_batch, data, mu, logvar, beta=params.beta
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_geo += geo_loss.item()
        
        # Average losses
        n_samples = len(train_loader.dataset)
        avg_loss = total_loss / n_samples
        avg_recon = total_recon / n_samples
        avg_kl = total_kl / n_samples
        avg_geo = total_geo / n_samples
        
        losses['total'].append(avg_loss)
        losses['recon'].append(avg_recon)
        losses['kl'].append(avg_kl)
        losses['geo'].append(avg_geo)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{params.epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Recon: {avg_recon:.4f} | "
                  f"KL: {avg_kl:.4f} | "
                  f"Geo: {avg_geo:.4f}")
    
    print("=" * 80)
    print("VAE TRAINING COMPLETE")
    print("=" * 80)
    
    return model, losses

def generate_scenarios(model, n_scenarios, device=DEVICE):
    """Generate new geological scenarios from trained VAE"""
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        # Sample from standard normal distribution
        z = torch.randn(n_scenarios, model.params.latent_dim).to(device)
        
        # Decode to generate scenarios
        scenarios = model.decode(z)
    
    return scenarios.cpu().numpy()

def save_vae_model(model, losses=None, filename='vae_model.pt'):
    """Save trained VAE model and training history"""
    filepath = os.path.join(MODEL_DIR, filename)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'params': model.params,
        'losses': losses
    }
    
    torch.save(checkpoint, filepath)
    print(f"VAE model saved to {filepath}")

def load_vae_model(params: VAEParameters, filename='vae_model.pt', device=DEVICE):
    """Load trained VAE model"""
    filepath = os.path.join(MODEL_DIR, filename)
    
    checkpoint = torch.load(filepath, map_location=device)
    
    model = GeologicalVAE(params)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"VAE model loaded from {filepath}")
    
    return model, checkpoint.get('losses', None)

# ============================================================================
# MAIN: Train VAE
# ============================================================================

if __name__ == '__main__':
    from config import DEFAULT_VAE_PARAMS
    
    print("=" * 80)
    print("VAE GEOLOGICAL MODELING")
    print("=" * 80)
    
    # Load dataset
    print("\nLoading mining dataset...")
    dataset = MiningDataGenerator.load_dataset()
    
    # Create PyTorch dataset and dataloader
    print("Preparing training data...")
    geo_dataset = GeologicalDataset(dataset)
    train_loader = DataLoader(
        geo_dataset, 
        batch_size=DEFAULT_VAE_PARAMS.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    print(f"Training samples: {len(geo_dataset)}")
    print(f"Batch size: {DEFAULT_VAE_PARAMS.batch_size}")
    print(f"Batches per epoch: {len(train_loader)}")
    
    # Create and train VAE
    print("\nInitializing VAE...")
    vae_model = GeologicalVAE(DEFAULT_VAE_PARAMS)
    
    print(f"Model parameters: {sum(p.numel() for p in vae_model.parameters()):,}")
    print(f"Latent dimension: {DEFAULT_VAE_PARAMS.latent_dim}")
    
    # Train
    trained_model, training_losses = train_vae(
        vae_model, 
        train_loader, 
        DEFAULT_VAE_PARAMS,
        device=DEVICE
    )
    
    # Save model
    print("\nSaving trained model...")
    save_vae_model(trained_model, training_losses)
    
    # Test generation
    print("\nTesting scenario generation...")
    test_scenarios = generate_scenarios(trained_model, n_scenarios=10, device=DEVICE)
    print(f"Generated {len(test_scenarios)} new scenarios")
    print(f"Scenario shape: {test_scenarios.shape}")
    
    print("\n" + "=" * 80)
    print("VAE TRAINING COMPLETE!")
    print("=" * 80)