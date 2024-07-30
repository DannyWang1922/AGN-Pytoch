import logging

import torch.nn as nn
from torch import optim
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, activation=nn.ReLU()):
        super(VariationalAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        # encoder
        self.encoder_input2hidden = nn.Linear(input_dim, hidden_dim)
        self.z_mean = nn.Linear(hidden_dim, latent_dim)
        self.z_log_var = nn.Linear(hidden_dim, latent_dim)

        # decoder
        self.decoder_latent2hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_hidden2output = nn.Linear(hidden_dim, input_dim)

        self.sigmoid = nn.Sigmoid()

    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        epsilon = torch.randn_like(std)
        return z_mean + epsilon * std

    def encode(self, x):
        hidden = self.activation(self.encoder_input2hidden(x))
        z_mean = self.z_mean(hidden)
        z_log_var = self.z_log_var(hidden)
        return z_mean, z_log_var

    def decode(self, z):
        hidden = self.activation(self.decoder_latent2hidden(z))
        output = self.decoder_hidden2output(hidden)
        decoded = self.sigmoid(output)
        return decoded

    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        decoded = self.decode(z)
        return decoded, z_mean, z_log_var

    def vae_loss(self, y_pred, y_true, z_mean, z_log_var):
        reconstruction_loss = F.binary_cross_entropy(y_pred, y_true)
        kl_loss = 1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var)
        kl_loss = torch.mean(kl_loss)
        kl_loss *= -0.5
        return reconstruction_loss + kl_loss

    def trainEncoder(self, data, batch_size, epochs, device):
        print("Begin training encoder")
        optimizer = optim.Adam(self.parameters(), lr=1e-5)
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        self.train()
        for epoch in range(epochs):
            for x in data_loader:
                x = x.to(device)
                optimizer.zero_grad()
                y_pred, z_mean, z_log_var = self.forward(x)
                loss = self.vae_loss(y_pred, x, z_mean, z_log_var)
                loss.backward()
                optimizer.step()
            print(f"\tEpoch {epoch + 1}, Loss: {loss.item()}")
            logging.info(f"\tEpoch {epoch + 1}, Loss: {loss.item()}")
        print("VAE Encoder training finish.")
        print()

    def predict(self, data, batch_size, device):
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        self.eval()
        all_preds = []

        with torch.no_grad():
            for x in data_loader:
                x = x.to(device)
                z_mean, z_log_var = self.encode(x)
                z = self.reparameterize(z_mean, z_log_var)
                all_preds.append(z)

        all_preds = torch.cat(all_preds, dim=0)
        return all_preds


# AutoEncoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, activation=nn.ReLU()):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, latent_dim),
            activation
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        encoded = self.encoder(x)
        return encoded

    def decode(self, encoded):
        decoded = self.decoder(encoded)
        return decoded

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

    def ae_loss(self, y_pred, y_true):
        loss = F.binary_cross_entropy(y_pred, y_true)
        return loss

    def trainEncoder(self, data, batch_size, epochs, device):
        print("Begin training autoencoder")
        logging.info("Begin training autoencoder.")
        optimizer = optim.Adam(self.parameters(), lr=1e-5)
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        self.train()
        for epoch in range(epochs):
            for x in data_loader:
                x = x.to(device)
                optimizer.zero_grad()
                y_pred = self.forward(x)
                loss = self.ae_loss(y_pred, x)
                loss.backward()
                optimizer.step()
            print(f"\tEpoch {epoch + 1}, Loss: {loss.item()}")
            logging.info(f"\tEpoch {epoch + 1}, Loss: {loss.item()}")
        print("Autoencoder training finish.")
        logging.info("Autoencoder training finish.\n")
        print()

    def predict(self, data, batch_size, device):
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        self.eval()
        all_preds = []

        with torch.no_grad():
            for x in data_loader:
                x = x.to(device)
                z = self.encode(x)
                all_preds.append(z)

        all_preds = torch.cat(all_preds, dim=0)
        return all_preds
