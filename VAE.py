import torch

import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 128)

        self.fc_mu = nn.Linear(128, latent_dim)

        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))

        mu = self.mc_mu(h)

        logvar = self.fc_logvar(h)

        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()

        self.fc1 = nn.Linear(latent_dim, 128)

        self.fc_out = nn.Linear(128, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))

        return self.fc_out(h)

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.encoder = Encoder(input_dim, latent_dim)

        self.decoder = Decoder(latent_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
    
        esp = torch.randn_like(std)
    
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)

        z = self.reparameterize(mu, logvar)

        recon_x = self.decoder(z)

        return recon_x, mu, logvar

    def vae_loss(self, recon_x, x, mu, logvar, beta=1.0):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')

        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + beta * kl_loss