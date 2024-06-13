import argparse
import json

import torch
import numpy as np
import random
import os
from transformers import BertTokenizer

from AGNDataloader import AGNDataLoader
from metrics import ClfMetrics
from model import DataGenerator
from AGN_model import AGNPaddedDataset, AGNModel, train_agn_model

import os
import torch
from torch.utils.data import DataLoader


def set_seed(seed=42):
    random.seed(seed)  # Python Random
    np.random.seed(seed)  # Numpy
    torch.manual_seed(seed)  # PyTorch

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 为所有 GPU 设置种子
        torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU，为所有 GPU 设置种子
        torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
        torch.backends.cudnn.benchmark = False  # 如果输入数据维度或类型上变化不大，设置为 True 可以提高性能


def check_device():
    """Check for device"""
    if torch.cuda.is_available():
        print("CUDA is available.")
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("MPS is available.")
        device = torch.device("mps")
    else:
        print("Using CPU.")
        device = torch.device("cpu")
    return device


def main():
    parser = argparse.ArgumentParser(description='AGN-Plus Configuration')
    parser.add_argument('--config', type=str, default="data/sst2/config.json")
    args = parser.parse_args()

    config_file = args.config
    with open(config_file, "r") as reader:
        config = json.load(reader)
    formatted_json = json.dumps(config, indent=4, sort_keys=True)  # json read friendly format
    print("Config:")
    # print(formatted_json)
    print()

    # create save_dir folder if not exists
    if not os.path.exists(config['save_dir']):
        os.makedirs(config['save_dir'])

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(config['pretrained_model_dir'], do_lower_case=True)
    tokenizer.model_max_length = config['max_len']

    print("Loading data...")
    AGNDataloader = AGNDataLoader(tokenizer,
                                  device=device,
                                  ae_latent_dim=config['ae_latent_dim'],
                                  use_vae=True,
                                  batch_size=config["batch_size"],
                                  ae_epochs=config['ae_epochs'],
                                  max_length=config['max_len'])
    AGNDataloader.set_train(config['train_path'])
    AGNDataloader.set_dev(config['dev_path'])
    AGNDataloader.save_autoencoder(os.path.join(config['save_dir'], 'autoencoder_weights.pth'))
    AGNDataloader.save_vocab(os.path.join(config['save_dir'], 'vocab.pickle'))
    config['output_size'] = AGNDataloader.label_size
    print()

    print("Begin training AGN")
    accuracy_list = []
    f1_list = []
    for idx in range(1, config['iterations'] + 1):
        print(f"Starting iteration {idx}")
        train_dataset = AGNPaddedDataset(AGNDataloader.train_set[:300])
        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_dataset = AGNPaddedDataset(AGNDataloader.dev_set)
        val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

        model = AGNModel(config)
        model = model.to(device)

        # 训练模型，并可能添加额外的回调以跟踪验证精度和F1分数
        acc, f1 = train_agn_model(model, train_dataloader, val_dataloader, device, epochs=config["epochs"],
                                  learning_rate=config["learning_rate"],
                                  save_path=config["save_dir"])
        # 保存每次迭代的性能
        accuracy_list.append(acc)
        f1_list.append(f1)
        print(f"Iteration {idx} End.  Accuracy: {acc}, F1: {f1}")
        print()

    # 计算所有迭代的平均精度和F1分数
    print("Average accuracy of all iterations:", sum(accuracy_list) / len(accuracy_list))
    print("Average f1 of all iterations:", sum(f1_list) / len(f1_list))




    # print("Begin training AGN")
    # train_dataset = AGNPaddedDataset(AGNDataloader.train_set[:300])
    # train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    #
    # val_dataset = AGNPaddedDataset(AGNDataloader.dev_set)
    # val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)
    #
    # model = AGNModel(config)
    # model = model.to(device)
    #
    # # train_agn_model(model=model, data_loader=train_dataloader, device=device,
    # #                 epochs=config['epochs'], learning_rate=config['learning_rate'])
    #
    # train_agn_model(model, train_dataloader, val_dataloader, device, epochs=config["epochs"], learning_rate=config["learning_rate"], save_path=config["save_dir"])


if __name__ == '__main__':
    set_seed(42)
    device = check_device()
    main()
