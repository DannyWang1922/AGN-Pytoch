import json
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
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
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

