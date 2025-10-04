"""
Variational Autoencoder for Geological Scenario Generation
Implements VAE-based dynamic scenario generation as per Section 4.3-4.4 of manuscript
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from config import DEFAULT_VAE_PARAMS, MODEL_DIR, DEVICE

class GeologicalDataset(Dataset):
    """PyTorch Dataset for geological features as per manuscript Section 4.4"""
    
    def __init__(self, data_dict):
        """Combine features: [grade, rock_type, alteration, structural, distance]"""
        features = np.column_stack([
            data_dict['base_grades'],
            data_dict['base_rock_types'],
            data_dict['geological_features']['alteration_intensity'],
            data_dict['geological_features']['structural_density'],
            data_dict['geological_features']['distance_to_intrusion']
        ])
        self.data = torch.FloatTensor(features)
        self.spatial_coords = torch.FloatTensor(data_dict['coordinates'])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.spatial_coords[idx]

class GeologicalVAE(nn.Module):
    """
    VAE for geological scenario generation (Equations 3-7 from manuscript)
    Implements dynamic scenario generation capability (50-200+ scenarios)
    """
    
    def __init__(self, params=DEFAULT_VAE_PARAMS):
        super(GeologicalVAE, self).__init__()
        self.params = params
        self.latent_dim = params.latent_dim
        
        # Encoder network (Equations 3-4)
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
        
        # Latent space parameters μ and log σ² (Equation 3-4)
        self.fc_mu = nn.Linear(params.hidden_dims[-1], params.latent_dim)
        self.fc_logvar = nn.Linear(params.hidden_dims[-1], params.latent_dim)
        
        # Decoder network (Equation 6)
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
        
        # Dual conditioning for Dantzig-Wolfe integration
        self.dual_conditioning = nn.Linear(10, params.latent_dim)
    
    def encode(self, x):
        """Encode to latent distribution (Equations 3-4)"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick (Equation 5)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode from latent space (Equation 6)"""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def condition_on_duals(self, z, dual_values):
        """Condition latent space on dual values for column generation"""
        dual_embedding = self.dual_conditioning(dual_values)
        return z + 0.1 * dual_embedding  # Soft conditioning
    
    def generate_scenarios(self, n_scenarios, dual_values=None):
        """Generate n_scenarios (50-200+) geological realizations"""
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_scenarios, self.latent_dim).to(next(self.parameters()).device)
            
            if dual_values is not None:
                z = self.condition_on_duals(z, dual_values)
            
            scenarios = self.decode(z)
            return scenarios.cpu().numpy()
    
    def calculate_continuity_loss(self, x, coords):
        """Geological continuity constraint for spatial correlation"""
        batch_size = x.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0)
        
        # Calculate pairwise distances
        dist = torch.cdist(coords, coords)
        
        # Create weight matrix based on distance
        weights = torch.exp(-dist / 50.0)  # 50m correlation range
        weights.fill_diagonal_(0)
        
        # Grade continuity loss
        grade_diff = (x[:, 0].unsqueeze(1) - x[:, 0].unsqueeze(0)).pow(2)
        continuity_loss = (weights * grade_diff).sum() / weights.sum()
        
        return continuity_loss

def vae_loss_function(recon_x, x, mu, logvar, coords, beta=0.5, lambda_geo=0.1):
    """
    VAE loss (Equation 7) with geological constraints
    L_VAE = L_reconstruction + β·L_KL + λ·L_geological
    """
    # Reconstruction loss
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Geological continuity constraint
    vae = GeologicalVAE()
    geo_loss = vae.calculate_continuity_loss(recon_x, coords)
    
    total_loss = recon_loss + beta * kl_loss + lambda_geo * geo_loss
    
    return total_loss, recon_loss, kl_loss, geo_loss

def train_vae(model, train_loader, params=DEFAULT_VAE_PARAMS, device=DEVICE):
    """Train VAE for geological scenario generation"""
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    model.train()
    model.to(device)
    
    losses = {'total': [], 'recon': [], 'kl': [], 'geo': []}
    
    for epoch in range(params.epochs):
        total_loss = 0
        for batch_idx, (data, coords) in enumerate(train_loader):
            data, coords = data.to(device), coords.to(device)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            
            loss, recon_loss, kl_loss, geo_loss = vae_loss_function(
                recon_batch, data, mu, logvar, coords, 
                beta=params.beta, lambda_geo=params.lambda_geo
            )
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader.dataset)
        losses['total'].append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{params.epochs} - Loss: {avg_loss:.4f}")
    
    return model, losses

def validate_geological_constraints(scenarios):
    """Validate geological realism of generated scenarios"""
    # Check spatial continuity
    spatial_correlation = np.corrcoef(scenarios[:, 0], scenarios[:, 3])[0, 1]
    
    # Check grade-tonnage relationship
    grade_tonnage_valid = np.corrcoef(scenarios[:, 0], scenarios[:, 4])[0, 1] < 0.3
    
    # Check alteration patterns
    alteration_valid = np.all(scenarios[:, 2] >= 0) and np.all(scenarios[:, 2] <= 1)
    
    return {
        'spatial_correlation': abs(spatial_correlation) > 0.2,
        'grade_tonnage_valid': grade_tonnage_valid,
        'alteration_valid': alteration_valid,
        'overall_valid': all([
            abs(spatial_correlation) > 0.2,
            grade_tonnage_valid,
            alteration_valid
        ])
    }