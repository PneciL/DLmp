import os, torch

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from CVAE import CVAE 
from GTZAN import GTZAN

SAMPLE_RATE = 22050

def save_waveform(waveform_tensor, filename):
    wav_np = waveform_tensor.squeeze().cpu().numpy()
    sf.write(filename, wav_np, SAMPLE_RATE)

def save_waveform_image(waveform_tensor, filename):
    wav_np = waveform_tensor.squeeze().cpu().numpy()
    
    plt.figure(figsize=(10, 4))
    plt.plot(wav_np, color='teal', linewidth=0.5)
    plt.ylim(-1.1, 1.1)
    plt.title("Generated Audio Waveform")
    plt.xlabel("Time (Samples)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def test_classification(model, dataset, device, num_samples=100):
    print(f"\n--- Running Classification Test on {num_samples} samples ---")

    correct = 0
    model.eval()
    
    with torch.no_grad():
        for i in range(num_samples):
            x, label = dataset[i]
            x = x.unsqueeze(0).to(device)
            
            _, _, _, preds = model(x) 
            pred_label = torch.argmax(preds, dim=1).item()
            
            if pred_label == label:
                correct += 1
                
    acc = (correct / num_samples) * 100
    print(f"Accuracy over {num_samples} samples: {acc:.2f}%")

def test_reconstruction(model, dataset, device, num_samples=3):
    print(f"\n--- Running Reconstruction Test on {num_samples} samples ---")

    os.makedirs("test_outputs_wav", exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        for i in range(num_samples):
            x, label = dataset[i]
            x = x.unsqueeze(0).to(device)
            
            recon_x, _, _, _ = model(x)
            
            save_waveform_image(x, f"test_outputs_wav/{label}_{i}_original.png")
            save_waveform(x, f"test_outputs_wav/{label}_{i}_original.wav")
            
            save_waveform_image(recon_x, f"test_outputs_wav/{label}_{i}_recon.png")
            save_waveform(recon_x, f"test_outputs_wav/{label}_{i}_recon.wav")
            
            print(f"Saved pair {i} (Genre ID: {label}) to test_outputs_wav/")

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

    os.makedirs("test_outputs_wav", exist_ok=True)
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
        
    save_waveform_image(out_clean, "test_outputs_wav/crypto_clean.png")
    save_waveform_image(out_encrypted, "test_outputs_wav/crypto_encrypted.png")
    save_waveform_image(out_decrypted, "test_outputs_wav/crypto_decrypted.png")
    
    print("Saved clean, encrypted, and decrypted waveforms.")

def test_generation(model, device, latent_dim=128, latent_frames=65, num_samples=3):
    print(f"\n--- Running Generation Test ({num_samples} new songs) ---")
    
    os.makedirs("test_outputs_wav", exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        for i in range(num_samples):
            z = torch.randn(1, latent_dim, latent_frames).to(device)
            generated_wav = model.decoder(z)
            
            save_waveform_image(generated_wav, f"test_outputs_wav/generation_{i}.png")
            save_waveform(generated_wav, f"test_outputs_wav/generation_{i}.wav")
            
            print(f"Generated brand new sample {i} from pure noise.")

def test_interpolation(model, dataset, device, steps=5):
    print(f"\n--- Running Editing (Interpolation) Test ---")
    
    os.makedirs("test_outputs_wav", exist_ok=True)
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
            wav_mix = model.decoder(z_mix)
            
            save_waveform_image(wav_mix, f"test_outputs_wav/interp_step_{i}.png")
            save_waveform(wav_mix, f"test_outputs_wav/interp_step_{i}.wav")
            print(f"Saved Interpolation step {i} (Alpha {alpha:.2f})")

def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    latent_dim = 128
    model = CVAE(latent_dim=latent_dim).to(device)

    checkpoint = torch.load("cvae.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    dataset = GTZAN(root_dir="datasets/GTZAN/genres_original")
    
    test_classification(model, dataset, device, num_samples=100)
    test_reconstruction(model, dataset, device, num_samples=3)
    test_compression(model, dataset, device)
    test_encryption(model, dataset, device)
    test_generation(model, device, latent_dim=128, latent_frames=65, num_samples=3)
    test_interpolation(model, dataset, device, steps=5)

if __name__ == "__main__":
    main()