import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(1, 16, kernel_size=15, stride=4, padding=7)),
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.Conv1d(16, 64, kernel_size=41, stride=4, padding=20, groups=4)),
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.Conv1d(64, 128, kernel_size=41, stride=4, padding=20, groups=16)),
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.Conv1d(128, 256, kernel_size=41, stride=4, padding=20, groups=16)),
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.Conv1d(256, 1, kernel_size=5, stride=1, padding=2))
        ])

    def forward(self, x):
        features = []
        for layer in self.convs:
            x = layer(x)
            features.append(x)
        
        return features