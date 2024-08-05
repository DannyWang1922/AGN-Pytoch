import json
import os
import random
import re
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class GlobalMaxPool1D(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1D, self).__init__()
        # 创建一个 AdaptiveMaxPool1d 层，设置输出大小为 1
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, x):  # x: (batch_size, length, channels)
        x_permuted = x.permute(0, 2, 1)  # (batch_size, channels, length)
        pooled = self.pool(x_permuted)
        pooled = pooled.permute(0, 2, 1)
        pooled = pooled.squeeze(1)
        return pooled


class ConditionalLayerNormalization(nn.Module):
    """ Conditional Layer Normalization
        https://arxiv.org/abs/2108.00449
        """

    def __init__(self, input_shapes, center=True, scale=True, offset=True, epsilon=1e-5):
        super(ConditionalLayerNormalization, self).__init__()
        self.center = center
        self.scale = scale
        self.offset = offset
        self.epsilon = epsilon

        input_shape, cond_shape = input_shapes
        if self.offset is True:
            self.beta = nn.Parameter(torch.zeros(input_shape))
            self.beta_cond = nn.Parameter(torch.zeros(cond_shape, input_shape))

        if self.scale is True:
            self.gamma = nn.Parameter(torch.ones(input_shape))
            self.gamma_cond = nn.Parameter(torch.zeros((cond_shape, input_shape)))

    def forward(self, inputs):
        inputs, cond = inputs
        if self.center:
            mean = inputs.mean(-1, keepdim=True)
            var = torch.square(inputs).mean(-1, keepdim=True)
            inputs = (inputs - mean) / torch.sqrt(var + self.epsilon)

        o = inputs
        if self.scale:
            gamma = self.gamma + torch.matmul(cond, self.gamma_cond)
            o *= gamma
        if self.offset:
            beta = self.beta + torch.matmul(cond, self.beta_cond)
            o += beta
        return o

    def get_config(self):
        config = {
            'center': self.center,
            'epsilon': self.epsilon,
            'scale': self.scale,
            'offset': self.offset,
        }
        base_config = super(ConditionalLayerNormalization, self).get_config()
        return dict(base_config, **config)


class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


def move_to_device(data, device):
    """Moves all data in the dictionary to the device"""
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.to(device)
    return data


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = string.replace("\n", "")
    string = string.replace("\t", "")
    string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def get_save_dir(base_dir_name):
    # 1. Check if the result directory exists; if not, create it.
    if not os.path.exists(base_dir_name):
        os.makedirs(base_dir_name)
        print(f"The directory '{base_dir_name}' has been created.")

    # 2. Check if there are any subdirectories in the result directory; if not, create result_0.
    subdirectories = [d for d in os.listdir(base_dir_name) if os.path.isdir(os.path.join(base_dir_name, d))]
    if not subdirectories:
        initial_subdir = os.path.join(base_dir_name, f"{os.path.basename(base_dir_name)}_1")
        os.makedirs(initial_subdir)
        print(f"The subdirectory '{initial_subdir}' has been created.")
        return initial_subdir  # Return immediately, as this is the first created subdirectory.

    # 3. Get the directory with the largest index.
    max_index = -1
    for subdir in subdirectories:
        if subdir.startswith(f"{os.path.basename(base_dir_name)}_"):
            try:
                index = int(subdir.split("_")[-1])
                if index > max_index:
                    max_index = index
            except ValueError:
                # If conversion fails, ignore this subdirectory.
                continue

    # 4. Create the directory result/result_{max_index+1}.
    next_index = max_index + 1
    next_subdir = os.path.join(base_dir_name, f"{os.path.basename(base_dir_name)}_{next_index}")
    os.makedirs(next_subdir)
    print(f"The subdirectory '{next_subdir}' has been created.")

    # 5. Return the path of the above directory.
    return next_subdir


def read_json(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line.strip())
            data.append(record)
    return data


def set_seed(seed):
    random.seed(seed)  # Python Random
    np.random.seed(seed)  # Numpy
    torch.manual_seed(seed)  # PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def collate_fn(batch):
    batch_token_ids = [item['token_ids'] for item in batch]
    batch_segment_ids = [item['segment_ids'] for item in batch]
    batch_sf = [item['sf_vector'] for item in batch]
    batch_label_ids = [item['label_ids'] for item in batch]

    # Padding
    batch_token_ids_padded = torch.nn.utils.rnn.pad_sequence(batch_token_ids, batch_first=True, padding_value=0)
    attention_masks = torch.nn.utils.rnn.pad_sequence([torch.ones_like(ids) for ids in batch_token_ids],
                                                      batch_first=True, padding_value=0)
    batch_segment_ids_padded = torch.nn.utils.rnn.pad_sequence(batch_segment_ids, batch_first=True, padding_value=0)
    batch_sf_padded = torch.nn.utils.rnn.pad_sequence(batch_sf, batch_first=True, padding_value=0)
    batch_label_ids_padded = torch.nn.utils.rnn.pad_sequence(batch_label_ids, batch_first=True, padding_value=-1)

    return {
        'token_ids': batch_token_ids_padded,
        'segment_ids': batch_segment_ids_padded,
        'sf_vector': batch_sf_padded,
        'attention_mask': attention_masks,
        'label_ids': batch_label_ids_padded
    }


def remove_cls_token(emissions, labels):
    # Iterate over each batch to remove [CLS] and [SEP]
    new_emissions_list = []
    new_labels_list = []

    for i in range(emissions.size(0)):
        # Find the indices of the first and last 1 in the attention mask
        token_101_idx = (labels[i] == -1).nonzero()[0].item()
        token_102_idx = (labels[i] == -1).nonzero()[1].item()

        # Remove the first and last tokens (corresponding to [CLS] and [SEP])
        new_emissions = torch.cat((emissions[i][:token_101_idx], emissions[i][token_101_idx + 1:token_102_idx],
                                   emissions[i][token_102_idx + 1:]), dim=0)
        new_labels = torch.cat(
            (labels[i][:token_101_idx], labels[i][token_101_idx + 1:token_102_idx], labels[i][token_102_idx + 1:]),
            dim=0)

        new_emissions_list.append(new_emissions)
        new_labels_list.append(new_labels)

    # Stack the lists back into tensors
    new_emissions = torch.stack(new_emissions_list)
    new_labels = torch.stack(new_labels_list)

    return new_emissions, new_labels
