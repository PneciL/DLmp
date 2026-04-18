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
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        proj = self.projection_head(mu)
        proj = F.normalize(proj, dim=1)
        return self.decoder(z), mu, logvar, self.classifier(z), proj

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

class SupConLoss(nn.Module):

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, projections: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = projections.device
        B = projections.shape[0]

        if B < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        projections = F.normalize(projections, dim=1)

        sim = projections @ projections.T / self.temperature

        labels = labels.view(-1)
        positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        positive_mask = positive_mask.float()

        diag = torch.eye(B, device=device, dtype=torch.bool)
        positive_mask = positive_mask.masked_fill(diag, 0.0)

        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

        loss = -(positive_mask * log_prob).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-9)

        return loss.mean()