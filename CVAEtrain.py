import torch

import torch.nn.functional as F

from torch.utils.data import DataLoader

from CVAE import CVAE
from GTZAN import GTZAN

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CVAE(latent_dim=500).to(device)

    gtzan = GTZAN(root_dir=r"datasets\GTZAN\genres_original")
    dataloader = DataLoader(gtzan, batch_size=32, shuffle=True, num_workers=7, pin_memory=True, persistent_workers=True)
    # dataloader = DataLoader(gtzan, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 50
    beta = 0.1
    gamma = 5
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (x, labels) in enumerate(dataloader):
            x = x.to(device)
            labels = labels.to(device)

            recon_x, mu, logvar, genre_pred = model(x)

            recon_loss = F.mse_loss(recon_x, x, reduction='mean')

            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            class_loss = F.cross_entropy(genre_pred, labels)

            loss = recon_loss + beta * kl_loss + gamma * class_loss

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            # if batch_idx == 0:
            #     print(f"Input shape: {x.shape}")
            #     print(f"Recon shape: {recon_x.shape}")
            #     print(f"Genre pred shape: {genre_pred.shape}")

        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {total_loss/len(dataloader):.4f}")
        print(f"MSE: {recon_loss:.4f} | KL: {kl_loss:.4f} | Acc: {class_loss:.2f}%")

    torch.save(model.state_dict(), "cvae_genre_model.pth")

if __name__ == "__main__":
    main()