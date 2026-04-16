import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as par

class ResBlock1d(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            par.weight_norm(nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)),
            nn.LeakyReLU(0.2),
            par.weight_norm(nn.Conv1d(channels, channels, kernel_size=1))
        )

    def forward(self, x):

        return x + self.block(x)

class ResStack1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResBlock1d(channels, dilation=1),
            ResBlock1d(channels, dilation=3),
            ResBlock1d(channels, dilation=9)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.blocks = nn.Sequential(
            par.weight_norm(nn.Conv1d(1, 16, 15, stride=4, padding=7)),
            ResStack1d(16),
            nn.LeakyReLU(0.2),
            par.weight_norm(nn.Conv1d(16, 32, 7, stride=4, padding=3)),
            ResStack1d(32),
            nn.LeakyReLU(0.2),
            par.weight_norm(nn.Conv1d(32, 64, 5, stride=4, padding=2)),
            ResStack1d(64),
            nn.LeakyReLU(0.2),
            par.weight_norm(nn.Conv1d(64, 128, 3, stride=4, padding=1)),
            ResStack1d(128),
            nn.LeakyReLU(0.2),
            par.weight_norm(nn.Conv1d(128, 256, 3, stride=4, padding=1)),
        )

        self.mu = nn.Conv1d(256, latent_dim, kernel_size=1)

        self.logvar = nn.Conv1d(256, latent_dim, kernel_size=1)

    def forward(self, x):
        # [b, 1, 66150]
        x = self.blocks(x)

        return self.mu(x), torch.clamp(self.logvar(x), min=-10, max=10)

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent = par.weight_norm(nn.Conv1d(latent_dim, 256, kernel_size=1))

        self.blocks = nn.Sequential(
            par.weight_norm(nn.ConvTranspose1d(256, 256, kernel_size=16, stride=4, padding=6)),
            ResStack1d(256),
            nn.LeakyReLU(0.2),
            par.weight_norm(nn.ConvTranspose1d(256, 128, kernel_size=16, stride=4, padding=6)),
            ResStack1d(128),
            nn.LeakyReLU(0.2),
            par.weight_norm(nn.ConvTranspose1d(128, 64, kernel_size=16, stride=4, padding=6)),
            ResStack1d(64),
            nn.LeakyReLU(0.2),
            par.weight_norm(nn.ConvTranspose1d(64, 32, kernel_size=16, stride=4, padding=6)),
            ResStack1d(32),
            nn.LeakyReLU(0.2),
            par.weight_norm(nn.ConvTranspose1d(32, 1, kernel_size=16, stride=4, padding=6))
        )

    def forward(self, z):
        h = self.latent(z)
        out = self.blocks(h)
        
        out = F.interpolate(out, size=66150, mode='linear', align_corners=False)
        
        return torch.tanh(out)

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

    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        mu, logvar = self.encoder(x)

        z = self.reparameterize(mu, logvar)

        return self.decoder(z), mu, logvar, self.classifier(torch.mean(z, dim=2))

class STFTLoss(nn.Module):
    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, x, y):
        window=torch.hann_window(self.win_length).to(y.device)
        x_stft = torch.stft(x.squeeze(1), self.n_fft, self.hop_length, self.win_length, window, return_complex=True)
        y_stft = torch.stft(y.squeeze(1), self.n_fft, self.hop_length, self.win_length, window, return_complex=True)

        x_mag = x_stft.abs()
        y_mag = y_stft.abs()

        sc_loss = torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")
        mag_loss = F.l1_loss(torch.log(x_mag + 1e-7), torch.log(y_mag + 1e-7))

        r_loss = F.l1_loss(x_stft.real, y_stft.real)
        i_loss = F.l1_loss(x_stft.imag, y_stft.imag)

        x_pha = torch.angle(x_stft)
        y_pha = torch.angle(y_stft)

        x_if = x_pha[:, :, 1:] - x_pha[:, :, :-1]
        y_if = y_pha[:, :, 1:] - y_pha[:, :, :-1]

        if_loss = F.l1_loss(torch.atan2(torch.sin(x_if - y_if), torch.cos(x_if - y_if)), torch.zeros_like(x_if))

        return sc_loss + mag_loss + r_loss + i_loss + if_loss

class MRSTFTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.losses = nn.ModuleList([
            STFTLoss(512, 50, 240),
            STFTLoss(1024, 120, 600),
            STFTLoss(2048, 240, 1200)
        ])

    def forward(self, x, y):

        return sum(loss(x, y) for loss in self.losses)