import torch

def main():
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    vae = VAE()
    beta = gamma = 1.0
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            x = batch

            recon_x, mu, logvar, genre_pred = vae(x)

            recon_loss = F.mse_loss(recon_x, x)

            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            class_loss = F.cross_entropy(genre_pred, labels)

            loss = recon_loss + beta * kl_loss + gamma * class_loss

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()