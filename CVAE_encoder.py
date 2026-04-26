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

class Classifier(nn.Module):
    def __init__(self,latent_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim*2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        self.sft=nn.Softmax(dim=0)
        
    def forward(self, z):
        return self.sft(self.classifier(z))

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        ## fft parameters
        # convolutional layers to process the spectrogram
        self.conv1 = nn.Conv2d(1, 16, kernel_size=11, stride=4, padding=7) #[1, 66150] -> [16, 16538]
        self.res1 = ResBlock2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=7, stride=2, padding=3) #[16, 16538] -> [32, 8269]
        self.res2 = ResBlock2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=4) #[32, 8269] -> [64, 4135]
        self.res3 = ResBlock2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=2) #[64, 4135] -> [128, 2068]
        self.res4 = ResBlock2d(128)

        self.bn = nn.BatchNorm2d(128)

        self.fc_mu = nn.Linear(31360, latent_dim)

        self.fc_logvar = nn.Linear(31360, latent_dim)

        

    def forward(self, x):
        # [b, 1, 66150]
        # get the fft magnitude for the input

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
        mu= self.fc_mu(x)
        
        return mu, logvar



class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        ## fft parameters
        self.fc1 = nn.Linear(latent_dim, 31360)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2), # -> 4136
            nn.Conv2d(128, 64, kernel_size=3, padding=0),
            ResBlock2d(64),
            nn.LeakyReLU(0.2)
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2), # -> 8272
            nn.Conv2d(64, 32, kernel_size=5, padding=0),
            ResBlock2d(32),
            nn.LeakyReLU(0.2)
        )
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2), # -> 16544
            nn.Conv2d(32, 16, kernel_size=7, padding=0),
            ResBlock2d(16),
            nn.LeakyReLU(0.2)
        )
        
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=4), # -> 66176
            nn.Conv2d(16, 1, kernel_size=15, padding=[3,4])
        )

    def forward(self, z):
        h = self.fc1(z)
        #print(h.shape)
        h = h.view(-1, 128, 35,7)
        #print(h.shape)
        h = self.up1(h)
        #print(h.shape)
        h = self.up2(h)
        #print(h.shape)
        h = self.up3(h)
        #print(h.shape)
        out = self.up4(h)
        #print(out.shape)
        

        #out = out[:, :, :66150]
        
        return torch.tanh(out)

class CVAE(nn.Module):
    def __init__(self, latent_dim,D_active=False):
        super().__init__()
        self.D_active = D_active
        self.n_fft = 1024*2-1
        self.hop_length = 256*2
        self.win_length = 1024
        self.window = "hann"
        self.fft = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window_fn=torch.hann_window)
        self.i_fft = torchaudio.transforms.InverseSpectrogram(n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window_fn=torch.hann_window)
        
        self.encoder = Encoder(latent_dim)

        self.decoder = Decoder(latent_dim)

        self.encoder_p = Encoder(latent_dim)

        self.decoder_p = Decoder(latent_dim)

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
        
        ff_t = self.fft(x)
        #split real and imag into mag and phase
        mag,phase = torch.abs(ff_t), torch.angle(ff_t)
        log_mag = torch.log(mag + 1e-7)
        # reshape to [b, 1, 66150]
        x = log_mag#.unsqueeze(1)
        p = phase#.unsqueeze(1)

        mu, logvar = self.encoder(x)
        mu_p, logvar_p = self.encoder_p(p)

        z = self.reparameterize(mu, logvar)
        z_p = self.reparameterize(mu_p, logvar_p)
        
        out_mag= self.decoder(z)
        out_phase = self.decoder_p(z_p)

        # stack the mag and phase back into a complex spectrogram
        complex_spec = out_mag.squeeze(1) * torch.exp(1j * out_phase.squeeze(1))
        # invert the spectrogram to get the time domain signal
        song = self.i_fft(complex_spec)
        if self.D_active:
            out_mag= self.decoder(z)
            out_phase = self.decoder_p(z_p)
            complex_spec = out_mag.squeeze(1) * torch.exp(1j * out_phase.squeeze(1))
            song = self.i_fft(complex_spec)
        else:
            song = None
        logvar_both = torch.cat([logvar, logvar_p], dim=1)
        mu_both = torch.cat([mu, mu_p], dim=1)
        return  mu_both, logvar_both , song
class CVAE_EN(nn.Module):
    def __init__(self, latent_dim,D_active=False):
        super().__init__()
        # set up fourer transform parameters
        self.D_active = D_active
        self.n_fft = 1024*2-1
        self.hop_length = 256*2
        self.win_length = 1024
        self.window = "hann"
        self.fft = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window_fn=torch.hann_window)
        self.i_fft = torchaudio.transforms.InverseSpectrogram(n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window_fn=torch.hann_window)
        # setup mag and phase vaes
        self.encoder = Encoder(latent_dim)

        self.decoder = Decoder(latent_dim)

        self.encoder_p = Encoder(latent_dim)

        self.decoder_p = Decoder(latent_dim)
        
        self.decoder_p = Decoder(latent_dim)
        # freeze the decoder and phase encoder
        #self.decoder.parameters.requires_grad = False
        #self.decoder_p.parameters.requires_grad = False
        for param in self.decoder_p.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False 
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
        
        ff_t = self.fft(x)
        #split real and imag into mag and phase
        mag,phase = torch.abs(ff_t), torch.angle(ff_t)
        log_mag = torch.log(mag + 1e-7)
        # reshape to [b, 1, 66150]
        x = log_mag #.unsqueeze(1)
        p = phase #.unsqueeze(1)

        mu, logvar = self.encoder(x)
        mu_p, logvar_p = self.encoder_p(p)

        z = self.reparameterize(mu, logvar)
        z_p = self.reparameterize(mu_p, logvar_p)

        if self.D_active:
            out_mag= self.decoder(z)
            out_phase = self.decoder_p(z_p)
            complex_spec = out_mag.squeeze(1) * torch.exp(1j * out_phase.squeeze(1))
            song = self.i_fft(complex_spec)
        else:
            song = None
            
        logvar_both = torch.cat([logvar, logvar_p], dim=1)
        mu_both = torch.cat([mu, mu_p], dim=1)
        return  mu_both, logvar_both , song

class CVAE_DE(nn.Module):
    def __init__(self, latent_dim,D_active=True):
        super().__init__()
        self.D_active = D_active
        self.n_fft = 1024*2-1
        self.hop_length = 256*2
        self.win_length = 1024
        self.window = "hann"
        self.fft = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window_fn=torch.hann_window)
        self.i_fft = torchaudio.transforms.InverseSpectrogram(n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window_fn=torch.hann_window)
        
        self.encoder = Encoder(latent_dim)

        self.decoder = Decoder(latent_dim)
        
        self.encoder_p = Encoder(latent_dim)

        self.decoder_p = Decoder(latent_dim)
        
        for param in self.encoder_p.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False 

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
        
        ff_t = self.fft(x)
        #split real and imag into mag and phase
        mag,phase = torch.abs(ff_t), torch.angle(ff_t)
        log_mag = torch.log(mag + 1e-7)
        # reshape to [b, 1, 66150]
        x = log_mag#.unsqueeze(1)
        p = phase#.unsqueeze(1)

        mu, logvar = self.encoder(x)
        mu_p, logvar_p = self.encoder_p(p)

        z = self.reparameterize(mu, logvar)
        z_p = self.reparameterize(mu_p, logvar_p)
        if self.D_active:
            out_mag= self.decoder(z)
            out_phase = self.decoder_p(z_p)
            complex_spec = out_mag.squeeze(1) * torch.exp(1j * out_phase.squeeze(1))
            song = self.i_fft(complex_spec)
        else:
            song = None

        logvar_both = torch.cat([logvar, logvar_p], dim=1)
        mu_both = torch.cat([mu, mu_p], dim=1)
        return  mu_both, logvar_both , song

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


class STFTLoss(nn.Module):
    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, x, y):
        window=torch.hann_window(self.win_length).to(y.device)
        x_stft = torch.stft(x.squeeze(1), self.n_fft, self.hop_length, self.win_length, window, return_complex=True)
        y_stft = torch.stft(y.squeeze(1), self.n_fft, self.hop_length, self.win_length, window, return_complex=True)

        x_mag = x_stft.abs()
        y_mag = y_stft.abs()

        sc_loss = torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")
        mag_loss = F.l1_loss(torch.log(x_mag + 1e-7), torch.log(y_mag + 1e-7))

        r_loss = F.l1_loss(x_stft.real, y_stft.real)
        i_loss = F.l1_loss(x_stft.imag, y_stft.imag)

        x_pha = torch.angle(x_stft)
        y_pha = torch.angle(y_stft)

        x_if = x_pha[:, :, 1:] - x_pha[:, :, :-1]
        y_if = y_pha[:, :, 1:] - y_pha[:, :, :-1]

        if_loss = F.l1_loss(torch.atan2(torch.sin(x_if - y_if), torch.cos(x_if - y_if)), torch.zeros_like(x_if))

        return sc_loss + mag_loss + r_loss + i_loss + if_loss

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