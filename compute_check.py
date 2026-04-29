import torch

import matplotlib.pyplot as plt

from torchinfo import summary

from CVAE import CVAE

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CVAE(latent_dim=128).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    model_stats = summary(model, input_size=(1, 1, 66560), device=device)
    print(str(model_stats))

    plt.figure(figsize=(12, 18))
    plt.text(0.01, 0.99, str(model_stats), fontfamily='monospace', fontsize=10, va='top')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("model_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()