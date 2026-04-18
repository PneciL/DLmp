import argparse, torch

import soundfile as sf
import torch.nn.functional as F

from pathlib import Path

from torch.utils.data import DataLoader

from CVAE import CVAE, MRSTFTLoss
from Discriminator import MRSTFTDiscriminator
from GTZAN import GTZAN

def save_checkpoint(model, optimizer_g, scheduler_g, optimizer_d, scheduler_d, epoch, path="cvae.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_g_state_dict": optimizer_g.state_dict(),
        "scheduler_g_state_dict": scheduler_g.state_dict(),
        "optimizer_d_state_dict": optimizer_d.state_dict(),
        "scheduler_d_state_dict": scheduler_d.state_dict()
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer_g, scheduler_g, optimizer_d, scheduler_d):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
    optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])

    return checkpoint['epoch']

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CVAE(latent_dim=128).to(device)
    discriminator = MRSTFTDiscriminator().to(device)

    root_dir = Path("datasets") / "GTZAN" / "genres_original"
    gtzan = GTZAN(root_dir=root_dir)
    # dataloader = DataLoader(gtzan, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    dataloader = DataLoader(gtzan, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    # single_song_dataset = torch.utils.data.Subset(gtzan, [0])
    # dataloader = DataLoader(single_song_dataset, batch_size=1, shuffle=False, num_workers=0)

    optimizer_g = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-6)
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=0.999)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=0.999)

    num_epochs = 100
    
    alpha_stft = 45.0
    start_beta_kl = 1e-5
    end_beta_kl = 5e-4
    anneal = 50
    # beta_kl = 0.00001
    gamma_cl = 0.0
    delta_gan = 1.0
    epsilon_fm = 2.0
    zeta_mse = 1.0

    mr_stft = MRSTFTLoss().to(device)

    waveform, label = gtzan[42]
    test_batch = waveform.unsqueeze(0).to(device)
    sf.write(f"original_test.wav", waveform.squeeze().numpy(), 22050)

    start_epoch = 0
    if args.resume and Path('cvae.pth').exists():
        start_epoch = load_checkpoint('cvae.pth', model, optimizer_g, scheduler_g, optimizer_d, scheduler_d) + 1
        print(f"Resuming from epoch {start_epoch}")
    else:
        model.apply(model.init_weights)
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (x, labels) in enumerate(dataloader):
            x = x.to(device)
            x = x / (torch.max(torch.abs(x)) + 1e-7)
            labels = labels.to(device)

            optimizer_d.zero_grad()

            recon_x, mu, logvar, genre_pred = model(x)

            f_real = discriminator(x)
            f_fake = discriminator(recon_x.detach())
            d_loss = 0
            for dr, df in zip(f_real, f_fake):
                d_loss += torch.mean((dr[-1] - 1)**2) + torch.mean(df[-1]**2)
            d_loss /= len(f_real)
            d_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()

            f_real = discriminator(x)
            f_fake = discriminator(recon_x)
            g_loss = 0
            fm_loss = 0
            for dr, df in zip(f_real, f_fake):
                g_loss += torch.mean((df[-1] - 1)**2)
                
                for i in range(len(f_fake) -1):
                    fm_loss += F.l1_loss(df[i], dr[i].detach())
            g_loss /= len(f_fake)
            fm_loss /= len(f_fake)

            recon_loss = mr_stft(recon_x, x)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            class_loss = F.cross_entropy(genre_pred, labels)

            beta_kl = min(end_beta_kl, start_beta_kl + (epoch * (end_beta_kl / anneal)))
                
            loss = alpha_stft * recon_loss + beta_kl * kl_loss + gamma_cl * class_loss + delta_gan * g_loss + epsilon_fm * fm_loss \
                + zeta_mse * F.l1_loss(recon_x, x)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_g.step()

            total_loss += loss.item()

            if batch_idx == 0:
                print(f"Recon Max: {recon_x.max().item():.6f} | Target Max: {x.max().item():.6f}")

        scheduler_g.step()
        scheduler_d.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                recon, _, _, _ = model(test_batch)
        
                recon_audio = recon[0].squeeze().cpu().numpy()
        
                orig_audio = test_batch[0].squeeze().cpu().numpy()
                    
                sf.write(f"reconstruction_test-{epoch+1}.wav", recon_audio, 22050)

                if epoch in [24, 49, 74, 99]:
                    torch.save(model.state_dict(), f"cvae_{epoch+1}.pth")

            model.train()

        save_checkpoint(model, optimizer_g, scheduler_g, optimizer_d, scheduler_d, epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {total_loss/len(dataloader):.4f}")
        print(f"Recon: {recon_loss:.4f} | KL: {kl_loss:.4f} | Acc: {class_loss:.2f}% | D: {d_loss:.2f} | G: {g_loss:.2f}")

    torch.save(model.state_dict(), "cvae_genre_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    
    main(args)