import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock2d(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        # [1,128,128] -> [32,64,64] -> [64,32,32] -> [128,16,16] -> [256,8,8]
        self.net = nn.Sequential(
            nn.Conv2d(1,   32,  4, stride=2, padding=1), nn.LeakyReLU(0.2), ResBlock2d(32),
            nn.Conv2d(32,  64,  4, stride=2, padding=1), nn.LeakyReLU(0.2), ResBlock2d(64),
            nn.Conv2d(64,  128, 4, stride=2, padding=1), nn.LeakyReLU(0.2), ResBlock2d(128),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.LeakyReLU(0.2), nn.BatchNorm2d(256),
        )
        self.fc_mu     = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)

    def forward(self, x):
        h = self.net(x).flatten(1)
        logvar = torch.clamp(self.fc_logvar(h), -10, 10)
        return self.fc_mu(h), logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)
        # [256,8,8] -> [128,16,16] -> [64,32,32] -> [32,64,64] -> [1,128,128]
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.LeakyReLU(0.2), ResBlock2d(128),
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1), nn.LeakyReLU(0.2), ResBlock2d(64),
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1), nn.LeakyReLU(0.2), ResBlock2d(32),
            nn.ConvTranspose2d(32,  1,   4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 256, 8, 8)
        return self.net(h)

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
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar, self.classifier(z)
