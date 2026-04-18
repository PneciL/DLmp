import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as par

class Discriminator(nn.Module):
    def __init__(self, n_fft=1024, hop_length=256, win_length=1024):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.convs = nn.ModuleList([
            par.weight_norm(nn.Conv2d(2, 32, (3, 9), padding=(1, 4))),
            par.weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            par.weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            par.weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            par.weight_norm(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
        ])

        self.fc = par.weight_norm(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        features = []
        x = x.squeeze(1)
        window = torch.hann_window(self.win_length).to(x.device)

        x_stft = torch.stft(x, self.n_fft, self.hop_length, self.win_length, window, return_complex=True)
        x = torch.stack([x_stft.real, x_stft.imag], dim=1)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.2)
            features.append(x)

        x = self.fc(x)
        features.append(x)
        
        return features

class MRSTFTDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            Discriminator(n_fft=1024, hop_length=256, win_length=1024),
            Discriminator(n_fft=2048, hop_length=512, win_length=2048),
            Discriminator(n_fft=512, hop_length=128, win_length=512)
        ])
        self.pooling = nn.AvgPool1d(4, stride=2, padding=2)

    def forward(self, x):
        results = []
        for d in self.discriminators:
            results.append(d(x))
            
        return results