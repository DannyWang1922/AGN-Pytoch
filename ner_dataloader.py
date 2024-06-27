import pickle
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn
from torch.utils.data import Dataset

from model_V_Net import Autoencoder, VariationalAutoencoder

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def convert_to_tensor(data, dtype=torch.float):
    return torch.tensor(data, dtype=dtype)


class NerDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dic = self.data[idx]
        data = {
            'token_ids': torch.tensor(dic['token_ids'], dtype=torch.long).clone().detach(),
            'segment_ids': torch.tensor(dic['segment_ids'], dtype=torch.long).clone().detach(),
            'tfidf_vector': torch.tensor(dic['tfidf_vector'], dtype=torch.float).clone().detach(),
            'label_ids': torch.tensor(dic['label_ids'], dtype=torch.long).clone().detach()
        }
        return data


def collate_fn(batch):
    batch_token_ids = [item['token_ids'] for item in batch]
    batch_segment_ids = [item['segment_ids'] for item in batch]
    batch_tfidf = [item['tfidf_vector'] for item in batch]
    batch_label_ids = [item['label_ids'] for item in batch]

    batch_token_ids_padded = torch.nn.utils.rnn.pad_sequence(batch_token_ids, batch_first=True, padding_value=0)
    attention_masks = torch.nn.utils.rnn.pad_sequence([torch.ones_like(ids) for ids in batch_token_ids],
                                                      batch_first=True, padding_value=0)
    batch_segment_ids_padded = torch.nn.utils.rnn.pad_sequence(batch_segment_ids, batch_first=True, padding_value=0)
    batch_tfidf = torch.stack(batch_tfidf)
    batch_label_ids_padded = torch.nn.utils.rnn.pad_sequence(batch_label_ids, batch_first=True, padding_value=-1)

    return {
        'token_ids': batch_token_ids_padded,
        'segment_ids': batch_segment_ids_padded,
        'tfidf_vector': batch_tfidf,
        'attention_mask': attention_masks,
        'label_ids': batch_label_ids_padded
    }


def generate_tcol_vectors(all_tokens, all_labels):
    # 提取所有可能的标签
    possible_labels = set(label for labels in all_labels for label in labels)
    token_label_counts = defaultdict(lambda: defaultdict(int))
    num_labels = len(possible_labels)

    # 扁平化token和标签列表
    tokens = [word_token for sentence_token in all_tokens for word_token in sentence_token]
    labels = [word_label for sentence_token in all_labels for word_label in sentence_token]

    # 统计每个token对应的标签频率
    for token, label in zip(tokens, labels):
        token_label_counts[token][label] += 1

    # 生成TCol向量
    token_tcol = {}
    token_tcol[0] = np.array([0] * num_labels)  # pad
    token_tcol[1] = np.array([0] * num_labels)  # unk
    token_tcol[0] = np.reshape(token_tcol[0], (1, -1))
    token_tcol[1] = np.reshape(token_tcol[1], (1, -1))

    for token, label_counts in token_label_counts.items():
        total_counts = sum(label_counts.values())
        vector = [label_counts[label] / total_counts for label in sorted(possible_labels)]
        token_tcol[token] = np.reshape(np.array(vector), (1, -1))

    return token_tcol  # {'I': [0.0, 0.0, 1.0], 'live': [0.0, 0.0, 1.0]}

def sentence_tcol_vectors(all_tokens, token_tcol, max_len=128):
    tcol_vectors = []
    for obj in all_tokens:
        padded = [0] * (max_len - len(obj))
        token_ids = obj + padded
        tcol_vector = np.concatenate([token_tcol.get(token, token_tcol[1]) for token in token_ids[:max_len]])
        tcol_vector = np.reshape(tcol_vector, (-1))
        tcol_vectors.append(tcol_vector)
    return np.array(tcol_vectors)


class NerDataLoader:
    def __init__(self, dataset_name, tokenizer, device, feature, max_len=512, ae_latent_dim=128, use_vae=False,
                 batch_size=64,
                 ae_epochs=20):
        self.dataset_name = dataset_name
        self.use_vae = use_vae
        self.batch_size = batch_size
        self.ae_latent_dim = ae_latent_dim
        self.ae_epochs = ae_epochs
        self.train_steps = 0
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad = '<pad>'
        self.unk = '<unk>'
        self.tfidf_Vectorizer = TfidfVectorizer(stop_words='english', min_df=3, max_features=5000)
        self.autoencoder = None
        self.device = device
        self.feature = feature

    def load_vocab(self, save_path):
        with open(save_path, 'rb') as reader:
            obj = pickle.load(reader)
        for key, val in obj.items():
            setattr(self, key, val)

    def save_autoencoder(self, save_path):
        torch.save(self.autoencoder.state_dict(), save_path)

    def load_autoencoder(self, save_path):
        self.init_autoencoder()
        self.autoencoder.load_state_dict(torch.load(save_path))

    def set_train(self):
        self._train_set = self._read_data(self.dataset_name, "train", is_train=True)

    def set_dev(self):
        self._dev_set = self._read_data(self.dataset_name, "validation")

    def set_test(self):
        self._test_set = self._read_data(self.dataset_name, "test")

    @property
    def train_set(self):
        return NerDataset(self._train_set)

    @property
    def dev_set(self):
        return NerDataset(self._dev_set)

    @property
    def test_set(self):
        return NerDataset(self._test_set)

    @property
    def label_size(self):
        return self._label_size

    def init_autoencoder(self, in_dims):
        # 根据self.use_vae决定初始化哪种自动编码器，需要自行定义Autoencoder和VariationalAutoencoder类
        if self.autoencoder is None:
            if self.use_vae:
                self.autoencoder = VariationalAutoencoder(input_dim=in_dims, latent_dim=self.ae_latent_dim,
                                                          hidden_dim=128, activation=nn.ReLU())
            else:
                self.autoencoder = Autoencoder(input_dim=in_dims, latent_dim=self.ae_latent_dim, hidden_dim=128,
                                               activation=nn.ReLU())
            self.autoencoder = self.autoencoder.to(self.device)

    def get_tcol_feature(self, data):
        df_data = pd.DataFrame(data)
        all_tokens = df_data['token_ids'].tolist()
        all_labels = df_data['label_ids'].tolist()

        token_tocol = generate_tcol_vectors(all_tokens, all_labels)
        with open('token_tocol.txt', 'w', encoding='utf-8') as file:
            for item in token_tocol:
                file.write(f"{str(item)}\n")

        sentence_tocol = sentence_tcol_vectors(all_tokens, token_tocol, max_len=self.max_len)
        return sentence_tocol

    def prepare_stat_feature(self, data, is_train=False):
        if self.use_vae:
            print("Batch alignment...")
            print("\tPrevious data size:", len(data))
            keep_size = len(data) // self.batch_size
            data = data[:keep_size * self.batch_size]
            print("\tAlignment data size:", len(data))

        if self.feature == "tfidf":
            print('\tUsing TF-IDF...')
            X = self.tfidf_Vectorizer.transform([obj['raw_text'] for obj in data]).todense()
            print('\tTF-IDF vector shape:', X.shape)
        elif self.feature == "tcol":
            print('\tUsing TCol...')
            X = self.get_tcol_feature(data)
            print('\tTCol vector shape:', X.shape)
        X = torch.from_numpy(X).to(dtype=torch.float32)

        if is_train:
            self.init_autoencoder(in_dims=X.shape[1])
            self.autoencoder.trainEncoder(data=X, batch_size=self.batch_size, epochs=self.ae_epochs, device=self.device)
        X = self.autoencoder.predict(X, batch_size=self.batch_size, device=self.device)
        # decomposite
        assert len(X) == len(data)
        for x, obj in zip(X, data):
            obj['tfidf_vector'] = x.tolist()
        return data

    def _read_data(self, name, split, is_train=False):
        dataset = load_dataset(name)[split]
        data = []
        tfidf_corpus = []
        all_label_set = set()
        for obj in dataset:
            raw_text = ' '.join(obj['tokens'])
            tfidf_corpus.append(raw_text)
            first, last = None, None
            token_ids, tag_ids = [], []
            for i, (tag, token) in enumerate(zip(obj['ner_tags'], obj['tokens'])):
                all_label_set.add(tag)
                tok = self.tokenizer(token)
                if i == 0:
                    first, last = tok["input_ids"][0], tok["input_ids"][-1]
                token_ids.extend(tok["input_ids"][1:-1])
                tag_ids.extend([tag] * len(tok["input_ids"][1:-1]))
            token_ids = [first] + token_ids[:self.max_len - 2] + [last]
            tag_ids = [0] + tag_ids[:self.max_len - 2] + [0]
            assert len(token_ids) == len(tag_ids)

            data.append({
                'raw_text': raw_text,
                'token_ids': token_ids,
                'segment_ids': [0] * len(token_ids),
                'label_ids': tag_ids
            })
            # fit tf-idf

        if is_train:
            if self.feature == "tfidf":
                self.tfidf_Vectorizer.fit(tfidf_corpus)
            self._label_size = len(all_label_set)
        data = self.prepare_stat_feature(data, is_train=is_train)

        return data
