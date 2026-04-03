import torch

def main():
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        for batch in dataloader:
            x = batch

            recon_x, mu, logvar = vae(x)

            loss = vae_loss(recon_x, x, mu, logvar)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()