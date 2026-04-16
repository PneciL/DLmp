import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as par

class Discriminator(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            par.weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0))),
            par.weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0))),
            par.weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0))),
            par.weight_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0))),
            par.weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        ])

    def forward(self, x):
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad))
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        
        features = []
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, 0.2)
            features.append(x)
        
        return features

class MPDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            Discriminator(2),
            Discriminator(3),
            Discriminator(5),
            Discriminator(7),
            Discriminator(11)
        ])

    def forward(self, x):

        return [d(x) for d in self.discriminators]