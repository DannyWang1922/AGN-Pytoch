import json
import pickle
from collections import defaultdict
import re
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from model import VariationalAutoencoder, Autoencoder


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


class AGNDataLoader:
    def __init__(self, tokenizer, device, ae_latent_dim=128, use_vae=False, batch_size=64, ae_epochs=100,
                 max_length=128):
        self._train_set = []
        self._dev_set = []
        self._test_set = []

        self.use_vae = use_vae
        self.batch_size = batch_size
        self.ae_latent_dim = ae_latent_dim
        self.ae_epochs = ae_epochs
        self.train_steps = 0
        self.tokenizer = tokenizer
        self.max_len = max_length
        self.device = device

        self.tcol_info = defaultdict(dict)
        self.tcol = {}
        self.label2idx = {}
        self.token2cnt = defaultdict(int)

        self.pad = '<pad>'
        self.unk = '<unk>'
        self.autoencoder = None

    def set_train(self, train_path):
        """set train dataset"""
        self._train_set = self._read_data(train_path, build_vocab=True)

    def set_dev(self, dev_path):
        """set dev dataset"""
        self._dev_set = self._read_data(dev_path)

    def set_test(self, test_path):
        """set test dataset"""
        self._test_set = self._read_data(test_path)

    @property
    def train_set(self):
        return self._train_set

    @property
    def dev_set(self):
        return self._dev_set

    @property
    def test_set(self):
        return self._test_set

    def save_autoencoder(self, save_path):
        torch.save(self.autoencoder.state_dict(), save_path)

    def save_vocab(self, save_path):
        with open(save_path, 'wb') as writer:
            pickle.dump({
                'label2idx': self.label2idx,
            }, writer)

    @property
    def label_size(self):
        return len(self.label2idx)

    def init_autoencoder(self):
        if self.autoencoder is None:
            in_dims = self.label_size * self.max_len
            if self.use_vae:
                self.autoencoder = VariationalAutoencoder(input_dim=in_dims, latent_dim=self.ae_latent_dim,
                                                          hidden_dim=128, activation=nn.LeakyReLU())
            else:
                self.autoencoder = Autoencoder(input_dim=in_dims, latent_dim=self.ae_latent_dim, hidden_dim=128,
                                               activation=nn.ReLU())
            self.autoencoder = self.autoencoder.to(self.device)

    def train_autoencoder(self, device):
        if self.autoencoder is None:
            in_dims = self.label_size * self.max_len
            if self.use_vae:
                self.autoencoder = VariationalAutoencoder(input_dim=in_dims, latent_dim=self.ae_latent_dim,
                                                          hidden_dim=128, activation=nn.ReLU())
            else:
                self.autoencoder = Autoencoder(input_dim=in_dims, latent_dim=self.ae_latent_dim, hidden_dim=128,
                                               activation=nn.ReLU())
            self.autoencoder = self.autoencoder.to(device)

    def add_tcol_info(self, token, label):
        """ add TCoL
        """
        if label not in self.tcol_info[token]:
            self.tcol_info[token][label] = 1
        else:
            self.tcol_info[token][label] += 1

    def set_tcol(self):
        """ set TCoL
        """
        self.tcol[0] = np.array([0] * self.label_size)  # pad
        self.tcol[1] = np.array([0] * self.label_size)  # unk
        self.tcol[0] = np.reshape(self.tcol[0], (1, -1))
        self.tcol[1] = np.reshape(self.tcol[1], (1, -1))
        for token, label_dict in self.tcol_info.items():
            vector = [0] * self.label_size
            for label_id, cnt in label_dict.items():
                vector[label_id] = cnt / self.token2cnt[token]
            vector = np.array(vector)
            self.tcol[token] = np.reshape(vector, (1, -1))

    def parse_tcol_ids(self, data, build_vocab=False):
        if self.use_vae:
            # 通常在使用变分自编码器（VAE）或需要确保每个批次具有固定数量样本的其他模型时使用
            # 对数据裁切,确保数据集的大小能够被批量大小整除，从而在训练过程中每个批次都保持一致的样本数量
            print("Batch alignment...")
            print("\tPrevious data size:", len(data))
            keep_size = len(data) // self.batch_size
            data = data[:keep_size * self.batch_size]
            print("\tAfter alignment data size:", len(data))
        if build_vocab:  # build_vocab only be True in training
            print("Setting tcol...")
            self.set_tcol()
            print("\tToken size:", len(self.tcol))
            # print("Finish TCol setting")
        tcol_vectors = []
        # 代码的目的是将 token_ids 扩充到一个固定的最大长度 (self.max_len), 并基于这些 token_ids 获取或构建与之相关的 tcol_vector
        for obj in data:
            padded = [0] * (self.max_len - len(obj['token_ids']))
            token_ids = obj['token_ids'] + padded
            tcol_vector = np.concatenate([self.tcol.get(token, self.tcol[1]) for token in token_ids[:self.max_len]])
            tcol_vector = np.reshape(tcol_vector, (1, -1))
            tcol_vectors.append(tcol_vector)
        if len(tcol_vectors) > 1:
            X = np.concatenate(tcol_vectors)
        else:
            X = tcol_vectors[0]
        X = torch.from_numpy(X).to(dtype=torch.float32)
        if build_vocab:
            self.init_autoencoder()
            self.autoencoder.trainEncoder(data=X, batch_size=self.batch_size, epochs=self.ae_epochs, device=self.device)
        X = self.autoencoder.predict(X, batch_size=self.batch_size, device=self.device)
        # decomposite
        assert len(X) == len(data)
        for x, obj in zip(X, data):
            obj['token_ids'] = torch.tensor(obj['token_ids'])
            obj['segment_ids'] = torch.tensor(obj['segment_ids'])
            obj['tcol_ids'] = x
        return data

    def _read_data(self, fpath, build_vocab=False):
        data = []
        tfidf_corpus = []
        with open(fpath, "r", encoding="utf-8") as reader:
            for line in reader:
                obj = json.loads(line)
                obj['text'] = clean_str(obj['text'])
                tfidf_corpus.append(obj['text'])
                if build_vocab:
                    if obj['label'] not in self.label2idx:
                        self.label2idx[obj['label']] = len(self.label2idx)
                tokenized = self.tokenizer(obj['text'])
                token_ids, segment_ids = tokenized["input_ids"], tokenized[
                    "token_type_ids"]  # Change: Remove segment_ids attribute
                for token in token_ids:
                    self.token2cnt[token] += 1
                    self.add_tcol_info(token, self.label2idx[obj['label']])
                data.append({
                    'raw_text': obj['text'],
                    'token_ids': token_ids,
                    'segment_ids': segment_ids,
                    'label_id': self.label2idx[obj['label']]
                })

        # # For test only ====================================
        # with open('Test_loadData.txt', 'w', encoding='utf-8') as file:
        #     for item in data:
        #         item_str = json.dumps(item, ensure_ascii=False)
        #         file.write(item_str + '\n')
        # # ========================================================

        data = self.parse_tcol_ids(data, build_vocab=build_vocab)

        return data
