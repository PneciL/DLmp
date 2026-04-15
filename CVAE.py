import torch

import torch.nn as nn
import torch.nn.functional as F

class ResBlock1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, kernel_size=3, padding=4, dilation=4)
        )

    def forward(self, x):

        return x + self.block(x)

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.blocks = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(1, 16, 15, stride=4, padding=7)),
            ResBlock1d(16),
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.Conv1d(16, 32, 7, stride=4, padding=3)),
            ResBlock1d(32),
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.Conv1d(32, 64, 5, stride=4, padding=2)),
            ResBlock1d(64),
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.Conv1d(64, 128, 3, stride=4, padding=1)),
            ResBlock1d(128),
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.Conv1d(128, 256, 3, stride=4, padding=1)),
        )

        self.flatten = 256 * 65

        self.fc_mu = nn.Linear(self.flatten, latent_dim)

        self.fc_logvar = nn.Linear(self.flatten, latent_dim)

    def forward(self, x):
        # [b, 1, 66150]
        x = self.blocks(x)

        x = x.view(x.size(0), -1)

        logvar = self.fc_logvar(x)
        logvar = torch.clamp(logvar, min=-10, max=10)

        return self.fc_mu(x), logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.flatten = 256 * 65
        self.fc1 = nn.Linear(latent_dim, self.flatten)

        self.blocks = nn.Sequential(
            nn.utils.weight_norm(nn.ConvTranspose1d(256, 128, kernel_size=4, stride=4)),
            ResBlock1d(128),
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(128, 64, kernel_size=4, stride=4)),
            ResBlock1d(64),
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(64, 32, kernel_size=4, stride=4)),
            ResBlock1d(32),
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(32, 16, kernel_size=4, stride=4)),
            nn.utils.weight_norm(nn.ConvTranspose1d(16, 1, kernel_size=8, stride=4, padding=2))
        )

    def forward(self, z):
        h = self.fc1(z).view(-1, 256, 65)
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

        return self.decoder(z), mu, logvar, self.classifier(z)

class STFTLoss(nn.Module):
    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, x, y):
        window=torch.hann_window(self.win_length).to(y.device)
        x_stft = torch.stft(x.squeeze(1), self.n_fft, self.hop_length, self.win_length, window, return_complex=True).abs()
        y_stft = torch.stft(y.squeeze(1), self.n_fft, self.hop_length, self.win_length, window, return_complex=True).abs()

        sc_loss = torch.norm(y_stft - x_stft, p="fro") / torch.norm(y_stft, p="fro")
        mag_loss = F.l1_loss(torch.log(x_stft + 1e-7), torch.log(y_stft + 1e-7))

        return sc_loss + mag_loss

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