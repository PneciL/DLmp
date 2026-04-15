import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from CVAE import CVAE
from Discriminator import Discriminator
from GTZAN import GTZAN

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CVAE(latent_dim=256).to(device)
    discriminator = Discriminator().to(device)

    gtzan = GTZAN(split="train")
    dataloader = DataLoader(gtzan, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)

    optimizer_g = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    num_epochs = 50
    alpha = 5.0 # recon
    beta  = 0.001# KL
    gamma = 1.0 # classification
    delta = 0.1 # adversarial

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for x, labels in dataloader:
            x = x.to(device)
            labels = labels.to(device)

            # discriminator
            optimizer_d.zero_grad()
            recon_x, mu, logvar, genre_pred = model(x)
            d_real = discriminator(x)
            d_fake = discriminator(recon_x.detach())
            d_loss = torch.mean((d_real - 1)**2) + torch.mean(d_fake**2)
            d_loss.backward()
            optimizer_d.step()

            # generator
            optimizer_g.zero_grad()
            recon_x, mu, logvar, genre_pred = model(x)

            d_fake = discriminator(recon_x)
            g_loss = torch.mean((d_fake - 1)**2)
            recon_loss = F.l1_loss(recon_x, x)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            class_loss = F.cross_entropy(genre_pred, labels)

            loss = alpha * recon_loss + beta * kl_loss + gamma * class_loss + delta * g_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_g.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss/len(dataloader):.4f} | "
              f"Recon: {recon_loss:.4f} | KL: {kl_loss:.4f} | Class: {class_loss:.4f} | "
              f"D: {d_loss:.4f} | G: {g_loss:.4f}")

    torch.save(model.state_dict(), "cvae_mel_model.pth")

if __name__ == "__main__":
    main()
