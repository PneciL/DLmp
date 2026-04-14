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

        self.conv1 = nn.Conv1d(1, 16, kernel_size=15, stride=4, padding=7) #[1, 66150] -> [16, 16538]
        self.res1 = ResBlock1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3) #[16, 16538] -> [32, 8269]
        self.res2 = ResBlock1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2) #[32, 8269] -> [64, 4135]
        self.res3 = ResBlock1d(64)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1) #[64, 4135] -> [128, 2068]
        self.res4 = ResBlock1d(128)

        self.bn = nn.BatchNorm1d(128)

        self.fc_mu = nn.Linear(128 * 2068, latent_dim)

        self.fc_logvar = nn.Linear(128 * 2068, latent_dim)

    def forward(self, x):
        # [b, 1, 66150]
        x = F.leaky_relu(self.res1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.res2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.res3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.res4(self.bn(self.conv4(x))), 0.2)

        x = torch.flatten(x, start_dim=1)

        logvar = self.fc_logvar(x)
        logvar = torch.clamp(logvar, min=-10, max=10)

        return self.fc_mu(x), logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.fc1 = nn.Linear(latent_dim, 128 * 2068)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2), # -> 4136
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            ResBlock1d(64),
            nn.LeakyReLU(0.2)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2), # -> 8272
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            ResBlock1d(32),
            nn.LeakyReLU(0.2)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2), # -> 16544
            nn.Conv1d(32, 16, kernel_size=7, padding=3),
            ResBlock1d(16),
            nn.LeakyReLU(0.2)
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=4), # -> 66176
            nn.Conv1d(16, 1, kernel_size=15, padding=7)
        )

    def forward(self, z):
        h = self.fc1(z)
        h = h.view(-1, 128, 2068)
        
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        out = self.up4(h)

        out = out[:, :, :66150]
        
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

    def forward(self, x):
        mu, logvar = self.encoder(x)

        z = self.reparameterize(mu, logvar)

        return self.decoder(z), mu, logvar, self.classifier(z)

class SpectralLoss(nn.Module):
    def __init__(self, n_fft=1024, hop_length=256, win_length=1024):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = None

    def forward(self, x, y):
        if self.window is None or self.window.device != x.device:
            self.window = torch.hann_window(self.win_length).to(x.device)

        x_stft = torch.stft(x.squeeze(1), self.n_fft, self.hop_length, self.win_length, self.window, return_complex=True).abs()
        y_stft = torch.stft(y.squeeze(1), self.n_fft, self.hop_length, self.win_length, self.window, return_complex=True).abs()

        log_x = torch.log(x_stft + 1e-7)
        log_y = torch.log(y_stft + 1e-7)

        return F.l1_loss(x_stft, y_stft) + F.l1_loss(log_x, log_y)

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, projections: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = projections.device
        B = projections.shape[0]

        # If only one projection is present in batch return 0
        if B < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Ensure embeddings are L2-normalized 
        # so magnitude of the vector does not affect calculation
        projections = F.normalize(projections, dim=1)

        # Cosine similarity matrix, dot product between all projections to measure
        # the similarity between them devided by temerature to maximize values of 
        # similar projections
        sim = torch.matmul(projections, projections.T) / self.temperature

        # Positive mask where labels are the same for both projections
        labels = labels.unsqueeze(1)                       # (B, 1)
        positive_mask = (labels == labels.T).float()          # (B, B)
        # Fill diagonal with 0 to ensure projections are not compared to themselves
        positive_mask.fill_diagonal_(0)

        # Remove self-similarity from the denominator
        identity = torch.eye(B, device=device)
        exp_sim = torch.exp(sim) * (1 - identity)             # zero diagonal

        # log P(positives | anchor)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)

        # Average over positives for each anchor
        num_positives = positive_mask.sum(dim=1)              # (B,)
        valid = num_positives > 0                             # anchors with at least 1 positive

        if not valid.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = -(positive_mask * log_prob).sum(dim=1) / (num_positives + 1e-9)
        return loss[valid].mean()