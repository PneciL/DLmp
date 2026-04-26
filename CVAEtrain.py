import torch

import torch.nn.functional as F

from pathlib import Path

from torch.utils.data import DataLoader

from CVAE_encoder import CVAE,CVAE_EN,CVAE_DE, SpectralLoss , MRSTFTLoss, Classifier
from Discriminator import Discriminator
from GTZAN import GTZAN
training_types= ["train_encoder_isolated","train_decoder_isolated","train_full","train_classifier_isolated","train_classifier_full"]
select=3
latent_D=300
print(torch.cuda.is_available())
D_active = False

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    classifier = None
    if training_types[select] == "train_encoder_isolated":
        model = CVAE_EN(latent_dim=latent_D,D_active=D_active).to(device)
    elif training_types[select] == "train_decoder_isolated":
        model = CVAE_DE(latent_dim=latent_D,D_active=D_active).to(device)
    elif training_types[select] == "train_full":
        model = CVAE(latent_dim=latent_D,D_active=D_active).to(device)
    elif training_types[select] == "train_classifier_isolated":
        model = CVAE_EN(latent_dim=latent_D,D_active=D_active).to(device)
        classifier = Classifier(latent_dim=latent_D).to(device)
    elif training_types[select] == "train_classifier_full":
        model = CVAE(latent_dim=latent_D,D_active=D_active).to(device)
        classifier = Classifier(latent_dim=latent_D).to(device)

 

    root_dir = Path("Data")  / "genres_original"
    gtzan = GTZAN(root_dir=root_dir)
    # dataloader = DataLoader(gtzan, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    dataloader = DataLoader(gtzan, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)

    optimizer_g = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer_c = torch.optim.Adam(classifier.parameters(), lr=1e-4)

    num_epochs = 150 
    
    alpha =1 # kl loss weight
    beta = 0.005 # within class scatter weight
    gamma = 0.005 # between class scatter weight
    delta = 0.001 # L1 loss weight
    epsilon = 1 # reconstruction loss weight
    yota = 4.0 # spectral loss weight


    spectral_512 = SpectralLoss(n_fft=512, win_length=512).to(device)
    spectral_1024 = SpectralLoss().to(device)
    spectral_2048 = SpectralLoss(n_fft=2048, win_length=2048).to(device)
    
    # load model if training encoder or decoder in isolation
    if training_types[select] == "train_encoder_isolated":
        # try to load the pre trained decoder weights, otherswise start from scratch
        try:
            model.train()
            model.load_state_dict(torch.load(f"cvae_genre_model_train_decoder_isolated_{latent_D}.pth", map_location=device))
            model.train()
        except FileNotFoundError:
            pass
    elif training_types[select] == "train_decoder_isolated":
        try:
            model.load_state_dict(torch.load(f"cvae_genre_model_train_encoder_isolated_{latent_D}.pth", map_location=device))
            model.train()
        except FileNotFoundError:
            pass
    elif training_types[select] == "train_full":
        try:
            model.load_state_dict(torch.load(f"cvae_genre_model_train_full_{latent_D}.pth", map_location=device))
            model.train()
        except FileNotFoundError:
            pass
    elif training_types[select] == "train_classifier_isolated":
        try:
            model.load_state_dict(torch.load(f"cvae_genre_model_train_encoder_isolated_{latent_D}.pth", map_location=device))
            classifier.load_state_dict(torch.load(f"classifier_model_train_classifier_isolated_{latent_D}.pth", map_location=device))
            model.eval()
            classifier.train()
        except FileNotFoundError:
            pass
    elif training_types[select] == "train_classifier_full":
        try:
            model.load_state_dict(torch.load(f"cvae_genre_model_train_full_{latent_D}.pth", map_location=device))
            classifier.load_state_dict(torch.load(f"classifier_model_train_classifier_full_{latent_D}.pth", map_location=device))
            model.eval()
            classifier.train()
        except FileNotFoundError:
            pass

    mr_stft = MRSTFTLoss().to(device)
    model.train()
    classifier.train()
    for epoch in range(num_epochs):
        total_loss = 0

        L1_total=0
        KL_total=0
        WCS_total=0
        BCS_total=0
        Recon_MSE_total=0
        Recon_Spectral_total=0
        total_class_loss = 0

        for batch_idx, (x, labels) in enumerate(dataloader):
            x = x.to(device)
            x = x / (torch.max(torch.abs(x)) + 1e-7)
            labels = labels.to(device)

            #optimizer_d.zero_grad()
            x = x[:, :1, :66049]  # Ensure input is the correct shape for the encoder
            mu, logvar,recon_x= model(x)
            z = model.reparameterize(mu, logvar)
            #d_real = discriminator(x)
            #d_fake = discriminator(recon_x.detach())
            #d_loss = torch.mean((d_real - 1)**2) + torch.mean(d_fake**2)
            #d_loss.backward()
            #optimizer_d.step()
            # step for classifier if training classifier
            if training_types[select] in ["train_classifier_isolated", "train_classifier_full"]:
                optimizer_c.zero_grad()
                genre_pred = classifier(z)
                class_loss = F.cross_entropy(genre_pred, labels)
                class_loss.backward()
                optimizer_c.step()
                total_class_loss += class_loss.item()
            else:

                optimizer_g.zero_grad()

            #d_fake = discriminator(recon_x)
            #g_loss = torch.mean((d_fake - 1)**2)

                recon_loss_mse = F.mse_loss(recon_x.unsqueeze(1), x, reduction='mean')
                recon_loss_spectral = spectral_512(recon_x, x) + spectral_1024(recon_x, x) + spectral_2048(recon_x, x)
                recon_loss = recon_loss_mse + recon_loss_spectral
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            # calculate within class scatter and between class scatter for the latent space
            #
                within_class_scatter = torch.tensor(0.0, device=device)
                between_class_scatter = torch.tensor(0.0, device=device)
                for i in range(10):
                    class_samples = mu[labels == i]
                    class_mean = torch.mean(class_samples, dim=0)
                    if class_samples.shape[0] > 1:
                        within_class_scatter += torch.sum((class_samples - class_mean)**2)    
                        between_class_scatter += torch.sum((class_mean - torch.mean(mu, dim=0))**2) * class_samples.shape[0]
                    #print(f"Class {i}: within scatter {torch.sum((class_samples - class_mean)**2).item():.4f}, between scatter {(torch.sum((class_mean - torch.mean(mu, dim=0))**2) * class_samples.shape[0]).item():.4f}")    
            # # add scatter losses to the total loss
                loss = alpha * kl_loss
            
                L1_loss = F.l1_loss(x,torch.zeros_like(x))
                if training_types[select] == "train_encoder_isolated":
                #print("encoder loss")
                #print(loss)
                    loss = alpha * kl_loss   + beta * torch.log(within_class_scatter+1e-6)  + delta * L1_loss
                #print(loss)
                elif training_types[select] == "train_decoder_isolated":
                #print("decoder loss")
                    loss = recon_loss_mse*epsilon + recon_loss_spectral*yota+ delta * L1_loss + mr_stft(recon_x.unsqueeze(1), x) * epsilon
                
                elif training_types[select] == "train_full":
                #print("full loss")
                #loss = recon_loss*epsilon + alpha * kl_loss   + beta*torch.log(within_class_scatter)  + delta * L1_loss + yota * recon_loss_spectral
                    loss = mr_stft(recon_x.unsqueeze(1), x) * epsilon + alpha * kl_loss   
            #print(f"Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}, KL: {kl_loss.item():.4f}, WCS: {within_class_scatter.item():.4f}, BCS: {between_class_scatter.item():.4f}, L1: {L1_loss.item():.4f}, Recon MSE: {recon_loss_mse.item():.4f}, Recon Spectral: {recon_loss_spectral.item():.4f}")
                L1_total += L1_loss.item()*delta
                KL_total += kl_loss.item()*alpha
                WCS_total += beta * torch.log(within_class_scatter+1e-7).item()
                BCS_total += between_class_scatter.item()*gamma
                Recon_MSE_total += recon_loss_mse.item()*epsilon
                Recon_Spectral_total += recon_loss_spectral.item()*yota

            #class_loss = F.cross_entropy(genre_pred, labels)
             # placeholder for L1 loss between recon_x and x
            #loss =  kl_loss 
            # add l1 loss to the loss for backpropagation
            #loss = alpha * kl_loss + gamma * L1_loss    

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer_g.step()

                total_loss += loss.item()
            # add l1 loss to the total loss for logging
            # if batch_idx == 0:
            #     print(f"Input shape: {x.shape}")
            #     print(f"Recon shape: {recon_x.shape}")
            #     print(f"Genre pred shape: {genre_pred.shape}")
        print(f"class loss: {total_class_loss/len(dataloader):.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {total_loss/len(dataloader):.4f} {training_types[select]}")
        print(f"KL: {KL_total:.4f}, WCS: {WCS_total:.4f}, BCS: {BCS_total:.4f}, L1: {L1_total:.4f}, Recon MSE: {Recon_MSE_total:.4f}, Recon Spectral: {Recon_Spectral_total:.4f}")

        # save model every 50 epochs
        if (epoch + 1) % 50 == 0:
            if training_types[select] in ["train_classifier_isolated", "train_classifier_full"]:
                torch.save(classifier.state_dict(), f"classifier_model_{training_types[select]}_{latent_D}.pth")
            else:
                torch.save(model.state_dict(), f"cvae_genre_model_{training_types[select]}_{latent_D}.pth")


    torch.save(model.state_dict(), f"cvae_genre_model_{training_types[select]}_{latent_D}.pth") 

if __name__ == "__main__":
    main()