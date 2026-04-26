import torch

import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class ResBlock2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=3, padding=4, dilation=4)
        )

    def forward(self, x):

        return x + self.block(x)

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        ## fft parameters
        self.n_fft = 1024*2
        self.hop_length = 256*2*2
        self.win_length = 1024*2
        self.window = "hann"
        self.fft = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window_fn=torch.hann_window)
        self.i_fft = torchaudio.transforms.InverseSpectrogram(n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window_fn=torch.hann_window)
        
        # convolutional layers to process the spectrogram
        self.conv1 = nn.Conv2d(1, 16, kernel_size=15, stride=4, padding=7) #[1, 66150] -> [16, 16538]
        self.res1 = ResBlock2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=7, stride=2, padding=3) #[16, 16538] -> [32, 8269]
        self.res2 = ResBlock2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2) #[32, 8269] -> [64, 4135]
        self.res3 = ResBlock2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) #[64, 4135] -> [128, 2068]
        self.res4 = ResBlock2d(128)

        self.bn = nn.BatchNorm2d(128)

        self.fc_mu = nn.Linear(12672, latent_dim)

        self.fc_logvar = nn.Linear(12672, latent_dim)
        

    def forward(self, x):
        # [b, 1, 66150]
        # get the fft magnitude for the input
        x = self.fft(x)
        #split real and imag into mag and phase
        mag,phase = torch.abs(x), torch.angle(x)
        log_mag = torch.log(mag + 1e-7)
        # reshape to [b, 1, 66150]
        x = log_mag#.unsqueeze(1)
        #print(x.shape)
        x = F.leaky_relu(self.res1(self.conv1(x)), 0.2)
        #print(x.shape)
        x = F.leaky_relu(self.res2(self.conv2(x)), 0.2)
        #print(x.shape)
        x = F.leaky_relu(self.res3(self.conv3(x)), 0.2)
        #print(x.shape)
        x = F.leaky_relu(self.res4(self.bn(self.conv4(x))), 0.2)
        #print(x.shape)
        x = torch.flatten(x, start_dim=1)
        #print(x.shape)
        logvar = self.fc_logvar(x)

        logvar = torch.clamp(logvar, min=-10, max=10)

        return self.fc_mu(x), logvar



class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.fc1 = nn.Linear(latent_dim, 128*2068)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2), # -> 4136
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            ResBlock2d(64),
            nn.LeakyReLU(0.2)
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2), # -> 8272
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            ResBlock2d(32),
            nn.LeakyReLU(0.2)
        )
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2), # -> 16544
            nn.Conv2d(32, 16, kernel_size=7, padding=3),
            ResBlock2d(16),
            nn.LeakyReLU(0.2)
        )
        
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=4), # -> 66176
            nn.Conv2d(16, 1, kernel_size=15, padding=7)
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
        self.encoder.parameters().requires_grad = False
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

        #z = self.reparameterize(mu, logvar)

        return  mu, logvar



class CVAE_en(nn.Module):
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

        return  mu, logvar, self.decoder(z)
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