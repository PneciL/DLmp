import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=4, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 64, kernel_size=41, stride=4, padding=20, groups=4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=41, stride=4, padding=20, groups=16),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=41, stride=4, padding=20, groups=16),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 1, kernel_size=5, stride=1, padding=2)
        )

    def forward(self, x):

        return self.convs(x)