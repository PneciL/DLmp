import torch

import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=64, stride=4, padding=30)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=32, stride=4, padding=15)

        self.fc_mu = nn.Linear(32 * 4134, latent_dim)

        self.fc_logvar = nn.Linear(32 * 4134, latent_dim)

    def forward(self, x):
        # [b, 1, 66150]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = torch.flatten(x, start_dim=1)

        return self.fc_mu(x), self.fc_logvar(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.fc1 = nn.Linear(latent_dim, 32 * 4134)

        self.deconv1 = nn.ConvTranspose1d(32, 16, kernel_size=32, stride=4, padding=14, output_padding=1)
        self.deconv2 = nn.ConvTranspose1d(16, 1, kernel_size=64, stride=4, padding=30, output_padding=2)

    def forward(self, z):
        h = self.fc1(z)
        h = h.view(-1, 32, 4134)
        h = F.relu(self.deconv1(h))

        return torch.tanh(self.deconv2(h))

class CVAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.encoder = Encoder(latent_dim)

        self.decoder = Decoder(latent_dim)

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
    
        eps = torch.randn_like(std)
    
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)

        z = self.reparameterize(mu, logvar)

        return self.decoder(z), mu, logvar, self.classifier(z)