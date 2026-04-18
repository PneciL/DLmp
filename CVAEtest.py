import torch
import soundfile as sf
import torchaudio.transforms as T
from torch.utils.data import DataLoader
from CVAE import CVAE
from GTZAN import GTZAN, N_MELS, N_FFT, HOP_LENGTH, SAMPLE_RATE

def mel_to_audio(mel_norm):
    # denormalize: [-1,1] -> [-80,0] dB -> power
    mel_db = (mel_norm - 1) * 40 # [-80, 0]
    mel_power = 10 ** (mel_db / 10)

    inv_mel = T.InverseMelScale(n_stft=N_FFT // 2 + 1, n_mels=N_MELS, sample_rate=SAMPLE_RATE)
    grifflim = T.GriffinLim(n_fft=N_FFT, hop_length=HOP_LENGTH, n_iter=128)

    spec = inv_mel(mel_power.cpu())
    audio = grifflim(spec)
    return audio

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CVAE(latent_dim=256).to(device)
    model.load_state_dict(torch.load('cvae_mel_model.pth', map_location=device, weights_only=True))

    gtzan = GTZAN(split="train")
    dataloader = DataLoader(gtzan, batch_size=32, shuffle=True)

    model.eval()
    with torch.no_grad():
        x, labels = next(iter(dataloader))
        x = x.to(device)

        recon, mu, logvar, pred, proj = model(x)

        orig_audio = mel_to_audio(x[0])
        recon_audio = mel_to_audio(recon[0])

        orig_audio  = orig_audio  / orig_audio.abs().max()  * 0.9
        recon_audio = recon_audio / recon_audio.abs().max() * 0.9

        sf.write("original_test.wav", orig_audio.squeeze().numpy(),  SAMPLE_RATE)
        sf.write("reconstruction_test.wav", recon_audio.squeeze().numpy(), SAMPLE_RATE)

        predicted_genres = pred.argmax(dim=1)
        print(f"True labels: {labels[:8].tolist()}")
        print(f"Predicted genres: {predicted_genres[:8].cpu().tolist()}")

if __name__ == "__main__":
    main()