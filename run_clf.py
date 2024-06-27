import argparse
import json
import logging

import numpy as np
import random
from transformers import BertTokenizer
from clf_dataloader import ClfDataLoader, ClfPaddedDataset
from model_AGN import AGNModel, train_agn_model, test_agn_model
import os
import torch
from torch.utils.data import DataLoader


def set_seed(seed=42):
    random.seed(seed)  # Python Random
    np.random.seed(seed)  # Numpy
    torch.manual_seed(seed)  # PyTorch

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def main(device):
    parser = argparse.ArgumentParser(description='AGN-Plus Configuration')
    parser.add_argument('--config', type=str, default="data/sst2/sst2.bert.json")
    args = parser.parse_args()
    config_file = args.config

    # Load config
    # config_file = "data/sst2/sst2.json"
    # config_file = "data/sst2/sst2.bert.json"

    with open(config_file, "r") as reader:
        config = json.load(reader)
    formatted_json = json.dumps(config, indent=4, sort_keys=True)  # json read friendly format
    print("Config:")
    print(formatted_json)
    print()

    config["device"] = device
    config["task"] = "clf"

    # create save_dir folder if not exists
    if not os.path.exists(config['save_dir']):
        os.makedirs(config['save_dir'])

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(config['pretrained_model_dir'], do_lower_case=True)
    tokenizer.model_max_length = config['max_len']

    print("Loading data...")
    clf_dataloader = ClfDataLoader(tokenizer,
                                   device=device,
                                   ae_latent_dim=config['ae_latent_dim'],
                                   use_vae=True,
                                   batch_size=config["batch_size"],
                                   ae_epochs=config['ae_epochs'],
                                   max_length=config['max_len'])
    clf_dataloader.set_train(config['train_path'])
    clf_dataloader.set_dev(config['dev_path'])
    clf_dataloader.save_autoencoder(os.path.join(config['save_dir'], 'autoencoder_weights.pth'))
    clf_dataloader.save_vocab(os.path.join(config['save_dir'], 'vocab.pickle'))
    config['output_size'] = clf_dataloader.label_size
    print()

    logging.basicConfig(
        filename='ner_results.log',
        level=logging.INFO,
        format=' %(message)s',
        filemode='w',
    )

    print("Begin training AGN")
    accuracy_list = []
    f1_list = []
    for idx in range(1, config['iterations'] + 1):
        print(f"Starting iteration {idx}")
        train_dataset = ClfPaddedDataset(clf_dataloader.train_set[:128])
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_dataset = ClfPaddedDataset(clf_dataloader.dev_set[:128])
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

        model = AGNModel(config)
        model = model.to(device)

        callback = train_agn_model(model, train_loader, val_loader, device, config=config)

        accuracy_list.append(callback.history["val_acc"])
        f1_list.append(callback.history["val_macro_f1"])
        print(f"Iteration {idx} End.")
        print()

    # avg_accuracy = np.mean(accuracy_list)
    # avg_f1 = np.mean(f1_list)
    #
    # print(f"Average accuracy of all iterations: {avg_accuracy}")
    # print(f"Average f1 of all iterations: {avg_f1}")
    #
    # with open(os.path.join(config['save_dir'], 'train_log.txt'), "w") as file:
    #     file.write(f"Average accuracy of all iterations: {avg_accuracy}\n"
    #                f"Average f1 of all iterations: {avg_f1}\n"
    #                f"{config}")

    model = AGNModel(config).to(device)
    test_dataset = ClfPaddedDataset(clf_dataloader.dev_set[:128])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    avg_loss, acc, macro_f1, micro_f1, report = test_agn_model(model, test_loader, config)

    print("Test Report:" + "\n")
    print(report)
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Macro F1-Score: {macro_f1:.4f}")
    print(f"Test Micro F1-Score: {micro_f1:.4f}")

    logging.info("\n" + "Test Report" + "\n" + report)
    logging.info(f"Test Loss: {avg_loss:.4f}")
    logging.info(f"Test Accuracy: {acc:.4f}")
    logging.info(f"Test Macro F1-Score: {macro_f1:.4f}")
    logging.info(f"Test Micro F1-Score: {micro_f1:.4f}")

    logging.info("\n" + str(formatted_json))


if __name__ == '__main__':
    set_seed(42)
    device = check_device()
    main(device)
