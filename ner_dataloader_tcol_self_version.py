import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch import nn

from model_V_Net import VariationalAutoencoder, Autoencoder
from utils import clean_str


def load_conll(fpath, tokenizer, build_vocab=False):
    data = []
    with open(fpath, "r", encoding="utf-8") as reader:
        for line in reader:
            obj = json.loads(line)
            obj['text'] = clean_str(obj['text'])
            tokenized = tokenizer(obj['text'])
            token_ids = tokenized["input_ids"]  # input_ids, token_type_ids, attention_mask

            # print("token_ids: ", token_ids )
            # print("token_ids length: ", len(token_ids))
            # print("token_ids type: ", type(token_ids))

            data.append({
                'raw_text': obj['text'],
                'token_ids': token_ids,
                'pos_tags': obj['pos_tags'],
                'chunk_tags': obj['chunk_tags'],
                'label_id': obj["label"]
            })
    return data


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
        tcol_vector = np.reshape(tcol_vector, (1, -1))
        tcol_vectors.append(tcol_vector)
    return tcol_vectors


def parse_tcol_ids(data):
    df_data = pd.DataFrame(data)
    all_tokens = df_data['token_ids'].tolist()
    all_labels = df_data['label_id'].tolist()

    token_tocol = generate_tcol_vectors(all_tokens, all_labels)
    sentence_tocol = sentence_tcol_vectors(all_tokens, token_tocol)

    df_data['tcol_ids'] = sentence_tocol
    df_data['tcol_ids'] = df_data['tcol_ids'].apply(lambda x: x.tolist())

    data = df_data.to_dict(orient='records')
    return data


def train_V_Net(data, config, use_vae=True):
    in_dims = len(data[0]['tcol_ids'][0])
    if use_vae:
        v_net_model = VariationalAutoencoder(input_dim=in_dims, latent_dim=config["ae_latent_dim"],
                                             hidden_dim=128, activation=nn.LeakyReLU())
    else:
        v_net_model = Autoencoder(input_dim=in_dims, latent_dim=config["ae_latent_dim"], hidden_dim=128,
                                  activation=nn.ReLU())
    v_net_model = v_net_model.to(config["device"])

    df_data = pd.DataFrame(data)
    X = df_data['tcol_ids'].tolist()
    X = np.array(X, dtype=np.float32)
    X = np.squeeze(X, axis=1)

    v_net_model.trainEncoder(data=X, batch_size=config["batch_size"], epochs=config["ae_epochs"],
                             device=config["device"])
    X = v_net_model.predict(X, batch_size=config["batch_size"], device=config["device"])

    # decomposite
    assert len(X) == len(data)
    for x, obj in zip(X, data):
        obj['token_ids'] = torch.tensor(obj['token_ids'])
        obj['tcol_ids'] = x
    return data

