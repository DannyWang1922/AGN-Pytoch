import json

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils import shuffle
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


class Sampling(nn.Module):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def forward(self, z_mean, z_log_var):
        batch = z_mean.size()[0]
        dim = z_mean.size()[1]
        epsilon = torch.randn(batch, dim, device=z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

# VAE
class VaeEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, activation):
        super(VaeEncoder, self).__init__()

        self.encoder_linear = nn.Linear(input_dim, hidden_dim)
        self.z_mean = nn.Linear(hidden_dim, latent_dim)
        self.z_log_var = nn.Linear(hidden_dim, latent_dim)
        self.sampling = Sampling()
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.encoder_linear(x))
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        encoded = self.sampling(z_mean, z_log_var)
        return encoded, z_mean, z_log_var


class VaeDecoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VaeDecoder, self).__init__()
        self.decoder_linear = nn.Linear(latent_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoded):
        decoded = self.sigmoid(self.decoder_linear(encoded))
        return decoded


def vae_loss(y_pred, y_true, z_mean, z_log_var):
    reconstruction_loss = F.binary_cross_entropy(y_pred, y_true)
    kl_loss = 1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var)
    kl_loss = torch.mean(kl_loss)
    kl_loss *= -0.5
    return reconstruction_loss + kl_loss


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, activation=nn.ReLU()):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VaeEncoder(input_dim, hidden_dim, latent_dim, activation)
        self.decoder = VaeDecoder(input_dim, latent_dim)

    def forward(self, x):
        encoded, z_mean, z_log_var = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, z_mean, z_log_var

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
                loss = vae_loss(y_pred, x, z_mean, z_log_var)
                loss.backward()
                optimizer.step()
            print(f"\tEpoch {epoch + 1}, Loss: {loss.item()}")
        print("Encoder training finish.")
        print()

    def predict(self, data, batch_size, device):
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        self.eval()
        all_encoded = []

        with torch.no_grad():
            for x in data_loader:
                x = x.to(device)
                encoded, z_mean, z_log_var = self.encoder.forward(x)
                all_encoded.append(encoded)

        all_encoded = torch.cat(all_encoded, dim=0)
        return all_encoded


# AutoEncoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, activation=nn.ReLU()):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.encoder(x))
        x = self.activation(self.decoder(x))
        return x

    def trainEncoder(self, data_loader, epochs, device):
        optimizer = optim.Adam(self.parameters())
        autoencoder_loss = nn.MSELoss()
        self.train()
        for epoch in range(epochs):
            for x in data_loader:
                x = x.to(device)
                optimizer.zero_grad()
                y_pred = self(x)
                loss = autoencoder_loss(y_pred, x)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


class DataGenerator:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.train_set = None
        self.train_size = None
        self.train_steps = None

    def set_dataset(self, train_set):
        self.train_set = train_set
        self.train_size = len(self.train_set)
        self.train_steps = len(self.train_set) // self.batch_size
        if self.train_size % self.batch_size != 0:
            self.train_steps += 1

    def __iter__(self, shuffle=True):
        while True:
            idxs = list(range(self.train_size))
            if shuffle:
                np.random.shuffle(idxs)
            batch_token_ids, batch_segment_ids, batch_tcol, batch_label_ids = [], [], [], []
            for idx in idxs:
                d = self.train_set[idx]
                batch_token_ids.append(d['token_ids'])
                batch_segment_ids.append(d['segment_ids'])
                batch_tcol.append(d['tcol_ids'])
                batch_label_ids.append(d['label_id'])
                if len(batch_token_ids) == self.batch_size or idx == idxs[-1]:
                    batch_token_ids = pad_sequence(batch_token_ids, batch_first=True, padding_value=0)
                    batch_segment_ids = pad_sequence(batch_segment_ids, batch_first=True, padding_value=0)
                    # batch_tcol = np.array(batch_tcol)
                    # batch_label_ids = np.array(batch_label_ids)
                    yield [batch_token_ids, batch_segment_ids, batch_tcol], batch_label_ids
                    batch_token_ids, batch_segment_ids, batch_tcol, batch_label_ids = [], [], [], []

    @property
    def steps_per_epoch(self):
        return self.train_steps


