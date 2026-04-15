import torch

import soundfile as sf

from torch.utils.data import DataLoader

from CVAE import CVAE
from GTZAN import GTZAN

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CVAE(latent_dim=128).to(device)
    model.load_state_dict(torch.load('cvae_genre_model.pth', map_location=device))

    gtzan = GTZAN(root_dir=r"datasets\GTZAN\genres_original")
    # dataloader = DataLoader(gtzan, batch_size=32, shuffle=True)
    single_song_dataset = torch.utils.data.Subset(gtzan, [0])
    dataloader = DataLoader(single_song_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    model.eval()
    with torch.no_grad():
        test_batch, labels = next(iter(dataloader))
        test_batch = test_batch.to(device)

        recon, mu, logvar, pred = model(test_batch)

        recon_audio = recon[0].cpu().numpy()
        if recon_audio.ndim == 2:
            recon_audio = recon_audio.T

        orig_audio = test_batch[0].cpu().numpy()
        if orig_audio.ndim == 2:
            orig_audio = orig_audio.T
            
        sf.write("reconstruction_test.wav", recon_audio, 22050)
        sf.write("original_test.wav", orig_audio, 22050)

if __name__ == "__main__":
    main()