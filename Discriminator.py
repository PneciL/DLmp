import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # patch discriminator on [1, 128, 128] mel spectrogram
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.InstanceNorm2d(64),  nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.net(x)
