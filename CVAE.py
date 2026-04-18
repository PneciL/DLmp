import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as par

class SnakeBeta(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):
        alpha = torch.exp(self.alpha)
        beta = torch.exp(self.beta)

        return x + (1.0 / (beta + 1e-9)) * (torch.sin(alpha * x).pow(2))

class ResBlock1d(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            SnakeBeta(channels),
            par.weight_norm(nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)),
            SnakeBeta(channels),
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
            SnakeBeta(16),
            par.weight_norm(nn.Conv1d(16, 32, 7, stride=4, padding=3)),
            ResStack1d(32),
            SnakeBeta(32),
            par.weight_norm(nn.Conv1d(32, 64, 5, stride=4, padding=2)),
            ResStack1d(64),
            SnakeBeta(64),
            par.weight_norm(nn.Conv1d(64, 128, 3, stride=4, padding=1)),
            ResStack1d(128),
            SnakeBeta(128),
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
            SnakeBeta(256),
            par.weight_norm(nn.ConvTranspose1d(256, 128, kernel_size=16, stride=4, padding=6)),
            ResStack1d(128),
            SnakeBeta(128),
            par.weight_norm(nn.ConvTranspose1d(128, 64, kernel_size=16, stride=4, padding=6)),
            ResStack1d(64),
            SnakeBeta(64),
            par.weight_norm(nn.ConvTranspose1d(64, 32, kernel_size=16, stride=4, padding=6)),
            ResStack1d(32),
            SnakeBeta(32),
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

class MRSTFTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.scales = [512, 1024, 2048, 4096]
        self.hops = [50, 120, 240, 480]
        self.wins = [240, 600, 1200, 2400]

    def get_k_weighting(self, freq_bins, sr=22050):
        freqs = torch.linspace(0, sr/2, freq_bins)
        weight = torch.where(freqs > 1000, 1.5, 1.0)
        weight = weight * torch.clamp(freqs / 100, max=1.0)

        return weight.to(freqs.device)

    def forward(self, x, y):
        total_mag_loss = 0

        corr_loss = torch.tensor(0.0).to(x.device)
        if_loss = torch.tensor(0.0).to(x.device)
        gd_loss = torch.tensor(0.0).to(x.device)

        for i, n_fft in enumerate(self.scales):
            hop = self.hops[i]
            win = self.wins[i]
            window = torch.hann_window(win).to(x.device)

            x_s = torch.stft(x.squeeze(1), n_fft, hop, win, window, return_complex=True)
            y_s = torch.stft(y.squeeze(1), n_fft, hop, win, window, return_complex=True)

            weights = self.get_k_weighting(x_s.shape[1]).view(1, -1, 1).to(x.device)

            x_mag = torch.log(x_s.abs() + 1e-7) * weights
            y_mag = torch.log(y_s.abs() + 1e-7) * weights
            total_mag_loss += F.l1_loss(x_mag, y_mag)

            if n_fft == 4096:
                num = (x_s * torch.conj(y_s)).real
                denom = x_s.abs() * y_s.abs() + 1e-7
                corr_loss = 1 - torch.mean(num / denom)

                x_pha = torch.angle(x_s)
                y_pha = torch.angle(y_s)
        
                x_if = (x_pha[:, :, 1:] - x_pha[:, :, :-1] + torch.pi) % (2 * torch.pi) - torch.pi
                y_if = (y_pha[:, :, 1:] - y_pha[:, :, :-1] + torch.pi) % (2 * torch.pi) - torch.pi
                if_loss = F.l1_loss(x_if, y_if)
        
                x_gd = -(x_pha[:, 1:, :] - x_pha[:, :-1, :] + torch.pi) % (2 * torch.pi) - torch.pi
                y_gd = -(y_pha[:, 1:, :] - y_pha[:, :-1, :] + torch.pi) % (2 * torch.pi) - torch.pi
                gd_loss = F.l1_loss(x_gd, y_gd)

        return total_mag_loss + corr_loss + if_loss + gd_loss

class KWeighting(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("b1", torch.tensor([1.4802, -2.5907, 1.1105]))
        self.register_buffer("a1", torch.tensor([1.0, -1.6144, 0.6144]))

        self.register_buffer("b2", torch.tensor([1.0, -2.0, 1.0]))
        self.register_buffer("a2", torch.tensor([1.0, -1.8600, 0.8600]))

    def apply_biquad(self, x, b, a):
        x = x.squeeze(1)
        y = torch.zeros_like(x)

        return x.unsqueeze(1)

    def forward(self, x):

        return x