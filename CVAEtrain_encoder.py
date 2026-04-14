import torch

import torch.nn.functional as F

from pathlib import Path

from torch.utils.data import DataLoader

from CVAE import CVAE, SpectralLoss
from Discriminator import Discriminator
from GTZAN import GTZAN

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CVAE(latent_dim=100).to(device)
    discriminator = Discriminator().to(device)

    root_dir = Path("Data")  / "genres_original"
    gtzan = GTZAN(root_dir=root_dir)
    # dataloader = DataLoader(gtzan, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    dataloader = DataLoader(gtzan, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)

    optimizer_g = torch.optim.Adam(model.parameters(), lr=1e-4)
    #optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    num_epochs = 50
    
    alpha = 5.0 # kl loss weight
    beta = 0.01 # within class scatter weight
    gamma = 0.05 # between class scatter weight
    delta = 0.001 # L1 loss weight
    epsilon = 2.0

    spectral_512 = SpectralLoss(n_fft=512, win_length=512).to(device)
    spectral_1024 = SpectralLoss().to(device)
    spectral_2048 = SpectralLoss(n_fft=2048, win_length=2048).to(device)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0

        L1_total=0
        KL_total=0
        WCS_total=0
        BCS_total=0
        for batch_idx, (x, labels) in enumerate(dataloader):
            x = x.to(device)
            x = x / (torch.max(torch.abs(x)) + 1e-7)
            labels = labels.to(device)

            #optimizer_d.zero_grad()

            mu, logvar= model(x)

            #d_real = discriminator(x)
            #d_fake = discriminator(recon_x.detach())
            #d_loss = torch.mean((d_real - 1)**2) + torch.mean(d_fake**2)
            #d_loss.backward()
            #optimizer_d.step()

            optimizer_g.zero_grad()

            #d_fake = discriminator(recon_x)
            #g_loss = torch.mean((d_fake - 1)**2)

            #recon_loss_mse = F.mse_loss(recon_x, x, reduction='mean')
            #recon_loss_spectral = spectral_512(recon_x, x) + spectral_1024(recon_x, x) + spectral_2048(recon_x, x)
            #recon_loss = recon_loss_mse + recon_loss_spectral
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            # calculate within class scatter and between class scatter for the latent space
            #
            within_class_scatter = 0
            between_class_scatter = 0
            for i in range(10):
                 class_samples = mu[labels == i]
                 class_mean = torch.mean(class_samples, dim=0)
                 if class_samples.shape[0] > 1:
                    within_class_scatter += torch.sum((class_samples - class_mean)**2)    
                    between_class_scatter += torch.sum((class_mean - torch.mean(mu, dim=0))**2) * class_samples.shape[0]
                    #print(f"Class {i}: within scatter {torch.sum((class_samples - class_mean)**2).item():.4f}, between scatter {(torch.sum((class_mean - torch.mean(mu, dim=0))**2) * class_samples.shape[0]).item():.4f}")    
            # # add scatter losses to the total loss
            
            
            L1_loss = F.l1_loss(x,torch.zeros_like(x))
        
            loss = alpha * kl_loss   + beta *  torch.log(within_class_scatter) - gamma * between_class_scatter + delta * L1_loss
            L1_total += L1_loss.item()
            KL_total += kl_loss.item()
            WCS_total += within_class_scatter.item()
            BCS_total += between_class_scatter.item()


            #class_loss = F.cross_entropy(genre_pred, labels)
             # placeholder for L1 loss between recon_x and x
            #loss =  kl_loss 
            # add l1 loss to the loss for backpropagation
            #loss = alpha * kl_loss + gamma * L1_loss    

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_g.step()

            total_loss += loss.item()
            # add l1 loss to the total loss for logging
            # if batch_idx == 0:
            #     print(f"Input shape: {x.shape}")
            #     print(f"Recon shape: {recon_x.shape}")
            #     print(f"Genre pred shape: {genre_pred.shape}")

        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {total_loss/len(dataloader):.4f}")
        print(f"KL: {KL_total:.4f}, WCS: {WCS_total:.4f}, BCS: {BCS_total:.4f}, L1: {L1_total:.4f}  ")

    torch.save(model.state_dict(), "cvae_genre_model.pth")

if __name__ == "__main__":
    main()