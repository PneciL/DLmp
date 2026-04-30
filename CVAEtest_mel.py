import os, torch, torchaudio

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from CVAE import CVAE
from GTZAN import GTZAN, N_FFT, HOP_LENGTH, N_MELS, SAMPLE_RATE

def mel_to_audio(mel_normalized, filename, device):
    mel_db = (mel_normalized.squeeze() - 1) * 40.0
    
    mel_power = 10.0 ** (mel_db / 10.0)
    
    inverse_mel = torchaudio.transforms.InverseMelScale(n_stft=N_FFT // 2 + 1, n_mels=N_MELS, sample_rate=SAMPLE_RATE).to(device)
    
    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=N_FFT, hop_length=HOP_LENGTH).to(device)
    
    lin_spec = inverse_mel(mel_power)
    waveform = griffin_lim(lin_spec)

    waveform_np = waveform.squeeze().cpu().numpy()
    sf.write(filename, waveform_np, SAMPLE_RATE)

def save_spec_image(spec, filename):
    plt.figure(figsize=(10, 4))
    plt.imshow(spec.squeeze().cpu().numpy(), aspect='auto', origin='lower', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def test_classification(model, dataset, device, num_samples=100):
    print(f"--- Running Classification Test on {num_samples} samples ---")
    correct = 0
    model.eval()
    
    with torch.no_grad():
        for i in range(num_samples):
            x, label = dataset[i]
            x = x.unsqueeze(0).to(device)
            
            _, _, _, preds, _ = model(x)
            pred_label = torch.argmax(preds, dim=1).item()
            
            if pred_label == label:
                correct += 1
                
    acc = (correct / num_samples) * 100
    print(f"Accuracy over {num_samples} samples: {acc:.2f}%")

def test_reconstruction(model, dataset, device, num_samples=3):
    print(f"\n--- Running Reconstruction Test on {num_samples} samples ---")
    os.makedirs("test_outputs", exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        for i in range(num_samples):
            x, label = dataset[i]
            x = x.unsqueeze(0).to(device)
            
            recon_x, _, _, _, _ = model(x)
            
            save_spec_image(x, f"test_outputs/{label}_{i}_original.png")
            mel_to_audio(x, f"test_outputs/{label}_{i}_original.wav", device)
            
            save_spec_image(recon_x, f"test_outputs/{label}_{i}_recon.png")
            mel_to_audio(recon_x, f"test_outputs/{label}_{i}_recon.wav", device)
            
            print(f"Saved pair {i} (Genre ID: {label}) to test_outputs/")

def test_compression(model, dataset, device):
    print("\n--- Running Compression Test ---")

    x, _ = dataset[0]
    x = x.unsqueeze(0).to(device)
    
    with torch.no_grad():
        mu, _ = model.encoder(x)
    
    input_elements = x.numel()
    latent_elements = mu.numel()
    ratio = input_elements / latent_elements
    
    print(f"Input Tensor Shape: {x.shape} -> {input_elements} floats")
    print(f"Latent Tensor Shape: {mu.shape} -> {latent_elements} floats")
    print(f"Compression Ratio: {ratio:.1f}x smaller in the latent space!")

def test_encryption(model, dataset, device):
    print("\n--- Running Encryption & Decryption Test ---")
    
    os.makedirs("test_outputs", exist_ok=True)
    model.eval()
    
    x, _ = dataset[0]
    x = x.unsqueeze(0).to(device)
    
    with torch.no_grad():
        mu, _ = model.encoder(x)
        
        key = torch.randn_like(mu) * 0.5 
        z_encrypted = mu + key
        
        z_decrypted = z_encrypted - key
        
        out_clean = model.decoder(mu)
        out_encrypted = model.decoder(z_encrypted)
        out_decrypted = model.decoder(z_decrypted)
        
    save_spec_image(out_clean, "test_outputs/crypto_clean.png")
    save_spec_image(out_encrypted, "test_outputs/crypto_encrypted.png")
    save_spec_image(out_decrypted, "test_outputs/crypto_decrypted.png")
    
    print("Saved clean, encrypted, and decrypted spectrograms.")

def test_generation(model, device, latent_dim=128, num_samples=3):
    print(f"\n--- Running Generation Test ({num_samples} new songs) ---")
    
    os.makedirs("test_outputs", exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        for i in range(num_samples):
            z = torch.randn(1, latent_dim).to(device)
            generated_mel = model.decoder(z)
            
            save_spec_image(generated_mel, f"test_outputs/generation_{i}.png")
            mel_to_audio(generated_mel, f"test_outputs/generation_{i}.wav", device)
            
            print(f"Generated brand new sample {i} from pure noise.")

def test_interpolation(model, dataset, device, steps=5):
    print(f"\n--- Running Editing (Interpolation) Test ---")
    
    os.makedirs("test_outputs", exist_ok=True)
    model.eval()
    
    x1, g1 = dataset[0]
    x2, g2 = dataset[-1]
    x1, x2 = x1.unsqueeze(0).to(device), x2.unsqueeze(0).to(device)
    
    with torch.no_grad():
        z1, _ = model.encoder(x1)
        z2, _ = model.encoder(x2)
        
        alphas = torch.linspace(0, 1, steps)
        for i, alpha in enumerate(alphas):
            z_mix = (1.0 - alpha) * z1 + alpha * z2
            mel_mix = model.decoder(z_mix)
            
            save_spec_image(mel_mix, f"test_outputs/interp_step_{i}.png")
            mel_to_audio(mel_mix, f"test_outputs/interp_step_{i}.wav", device)
            print(f"Saved Interpolation step {i} (Alpha {alpha:.2f})")

def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    latent_dim = 128
    model = CVAE(latent_dim=latent_dim).to(device)
    
    model.load_state_dict(torch.load("cvae_mel_model_test.pth", map_location=device))
    
    dataset = GTZAN(root_dir="datasets/GTZAN/genres_original")
    
    test_classification(model, dataset, device, num_samples=100)
    test_reconstruction(model, dataset, device, num_samples=3)
    test_compression(model, dataset, device)
    test_encryption(model, dataset, device)
    test_generation(model, device, latent_dim=128, num_samples=3)
    test_interpolation(model, dataset, device, steps=5)

if __name__ == "__main__":
    main()