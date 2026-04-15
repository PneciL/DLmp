import torch

import torch.nn.functional as F

from pathlib import Path

from torch.utils.data import DataLoader

from CVAE import CVAE, MRSTFTLoss
from Discriminator import Discriminator
from GTZAN import GTZAN

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CVAE(latent_dim=128).to(device)
    discriminator = Discriminator().to(device)

    root_dir = Path("datasets") / "GTZAN" / "genres_original"
    gtzan = GTZAN(root_dir=root_dir)
    # dataloader = DataLoader(gtzan, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    # dataloader = DataLoader(gtzan, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    single_song_dataset = torch.utils.data.Subset(gtzan, [0])
    dataloader = DataLoader(single_song_dataset, batch_size=1, shuffle=False, num_workers=0)

    optimizer_g = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-5)

    num_epochs = 50
    
    alpha_stft = 15.0
    beta_kl = 0.0000
    gamma_cl = 0.0
    delta_gan = 2.0
    epsilon_fm = 5.0
    zeta_mse = 0.5

    mr_stft = MRSTFTLoss().to(device)
    
    model.train()
    model.apply(model.init_weights)
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (x, labels) in enumerate(dataloader):
            x = x.to(device)
            x = x / (torch.max(torch.abs(x)) + 1e-7)
            labels = labels.to(device)

            optimizer_d.zero_grad()

            recon_x, mu, logvar, genre_pred = model(x)

            f_real = discriminator(x)
            f_fake = discriminator(recon_x.detach())
            d_real = f_real[-1]
            d_fake = f_fake[-1]
            d_loss = torch.mean((d_real - 1)**2) + torch.mean(d_fake**2)
            d_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()

            f_fake = discriminator(recon_x)
            g_loss = torch.mean((f_fake[-1] - 1)**2)

            fm_loss = 0
            for i in range(len(f_fake) -1):
                fm_loss += F.l1_loss(f_fake[i], f_real[i].detach())

            recon_loss = alpha_stft * mr_stft(recon_x, x) + zeta_mse * F.mse_loss(recon_x, x)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            class_loss = F.cross_entropy(genre_pred, labels)

            loss = recon_loss + beta_kl * kl_loss + gamma_cl * class_loss + delta_gan * g_loss + epsilon_fm * fm_loss

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_g.step()

            total_loss += loss.item()

            if batch_idx == 0:
                print(f"Recon Max: {recon_x.max().item():.6f} | Target Max: {x.max().item():.6f}")

        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {total_loss/len(dataloader):.4f}")
        print(f"Recon: {recon_loss:.4f} | KL: {kl_loss:.4f} | Acc: {class_loss:.2f}% | D: {d_loss:.2f} | G: {g_loss:.2f}")

    torch.save(model.state_dict(), "cvae_genre_model.pth")

if __name__ == "__main__":
    main()