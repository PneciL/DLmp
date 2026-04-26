import torch

import soundfile as sf

from torch.utils.data import DataLoader

from CVAE_encoder import CVAE
from GTZAN import GTZAN



training_types= ["train_encoder_isolated","train_decoder_isolated","train_full"]
select=2
latent_D=300

walk_resolution = 10

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CVAE(latent_dim=latent_D,D_active=False).to(device)
    model.load_state_dict(torch.load('cvae_genre_model.pth', map_location=device))

    gtzan = GTZAN(root_dir=r"datasets\GTZAN\genres_original")
    dataloader = DataLoader(gtzan, batch_size=16, shuffle=True)
    
    sample_label_1 = "classical"
    sample_label_2 = "jazz"

    model.eval()
    with torch.no_grad():

        #load two samples from the dataloader with different labels
        sample_1 = None
        sample_2 = None
        for test_batch, labels in dataloader:
            test_batch = test_batch.to(device)
            labels = labels.to(device)

            for i in range(test_batch.size(0)):
                if sample_1 is None and labels[i] == gtzan.label_to_index[sample_label_1]:
                    sample_1 = test_batch[i].unsqueeze(0)
                elif sample_2 is None and labels[i] == gtzan.label_to_index[sample_label_2]:
                    sample_2 = test_batch[i].unsqueeze(0)

                if sample_1 is not None and sample_2 is not None:
                    break
            if sample_1 is not None and sample_2 is not None:
                break
        # compute the latent representations of the two samples
        mu_1, logvar_1 = model.encoder(sample_1)
        mu_2, logvar_2 = model.encoder(sample_2)
        z_1 = model.reparameterize(mu_1, logvar_1)
        z_2 = model.reparameterize(mu_2, logvar_2)
        # interpolate between the two latent representations
        interpolations = []
        for alpha in torch.linspace(0, 1, walk_resolution):
            z_interp = alpha * z_1 + (1 - alpha) * z_2
            recon_interp = model.decoder(z_interp)
            interpolations.append(recon_interp.squeeze(0).cpu().numpy())
        # save the interpolations as audio files
        for i, recon_audio in enumerate(interpolations):
            if recon_audio.ndim == 2:
                recon_audio = recon_audio.T
            sf.write(f"latent_walks\interpolation_{i}.wav", recon_audio, 22050)

        test_batch, labels = next(iter(dataloader))
        test_batch = test_batch.to(device)

        recon, mu, logvar, pred = model(test_batch)
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