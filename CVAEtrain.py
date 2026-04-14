import torch

import torch.nn.functional as F

from torch.utils.data import DataLoader

from CVAE import CVAE, SpectralLoss, SupConLoss
from Discriminator import Discriminator
from GTZAN import GTZAN

def main(): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CVAE(latent_dim=500).to(device)
    discriminator = Discriminator().to(device)

    gtzan = GTZAN(root_dir=r"datasets\GTZAN\genres_original")
    # dataloader = DataLoader(gtzan, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    dataloader = DataLoader(gtzan, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)

    optimizer_g = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    num_epochs = 50
    alpha = 5.0
    beta = 0.00001
    gamma = 1.0
    delta = 0.1
    phita = 1

    soup = SupConLoss(temperature=0.07).to(device)
    spectral_512 = SpectralLoss(n_fft=512, win_length=512).to(device)
    spectral_1024 = SpectralLoss().to(device)
    spectral_2048 = SpectralLoss(n_fft=2048, win_length=2048).to(device)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (x, labels) in enumerate(dataloader):
            x = x.to(device)
            x = x / (torch.max(torch.abs(x)) + 1e-7)
            labels = labels.to(device)

            optimizer_d.zero_grad()

            recon_x, mu, logvar, genre_pred = model(x)

            d_real = discriminator(x)
            d_fake = discriminator(recon_x.detach())
            d_loss = torch.mean((d_real - 1)**2) + torch.mean(d_fake**2)
            d_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()

            d_fake = discriminator(recon_x)
            g_loss = torch.mean((d_fake - 1)**2)

            recon_loss_mse = F.mse_loss(recon_x, x, reduction='mean')
            recon_loss_spectral = spectral_512(recon_x, x) + spectral_1024(recon_x, x) + spectral_2048(recon_x, x)
            recon_loss = recon_loss_mse + recon_loss_spectral
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            class_loss = F.cross_entropy(genre_pred, labels)
            soup_loss = soup(x, labels)

            loss = alpha * recon_loss + beta * kl_loss + gamma * class_loss + delta * g_loss + phita * soup_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_g.step()

            total_loss += loss.item()

            # if batch_idx == 0:
            #     print(f"Input shape: {x.shape}")
            #     print(f"Recon shape: {recon_x.shape}")
            #     print(f"Genre pred shape: {genre_pred.shape}")

        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {total_loss/len(dataloader):.4f}")
        print(f"Recon: {recon_loss:.4f} | KL: {kl_loss:.4f} | Acc: {class_loss:.2f}% | D: {d_loss:.2f} | G: {g_loss:.2f}")

    torch.save(model.state_dict(), "cvae_genre_model.pth")

if __name__ == "__main__":
    main()