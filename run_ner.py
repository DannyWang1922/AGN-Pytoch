import argparse
import json
import random
import torch
from transformers import BertTokenizer
from ner_dataloader import NerDataLoader, NerDataset, collate_fn
from model_AGN import AGNModel, train_agn_model, test_agn_model
from torch.utils.data import DataLoader
import numpy as np
import os
import logging
from utils import get_save_dir


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
    else:
        print("Using CPU.")
        device = torch.device("cpu")
    return device


def main(device):
    parser = argparse.ArgumentParser(description='AGN-Plus Configuration')
    parser.add_argument('--config', type=str, default="data/ner/conll2003.json")
    args = parser.parse_args()
    config_file = args.config

    # Load config
    # config_file = "data/ner/conll2003.json"
    # config_file = "data/ner/conll2003.bert.json"

    with open(config_file, "r") as reader:
        config = json.load(reader)
    formatted_json = json.dumps(config, indent=4, sort_keys=True)
    print("Config:")
    print(formatted_json)
    print()

    config["device"] = device
    config["task"] = "ner"
    config['save_dir'] = get_save_dir(config['save_dir'])

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(config['pretrained_model_dir'], do_lower_case=True)
    tokenizer.model_max_length = config['max_len']

    print("Load data...")
    ner_dataloader = NerDataLoader(config['dataset_name'], tokenizer, feature=config["feature"],
                                   device=config["device"],
                                   max_len=config['max_len'],
                                   ae_latent_dim=config['ae_latent_dim'], use_vae=config['use_vae'],
                                   batch_size=config["batch_size"],
                                   ae_epochs=config['ae_epochs'])
    ner_dataloader.set_train(config['train_path'])
    ner_dataloader.set_dev(config['dev_path'])
    ner_dataloader.set_test(config['test_path'])
    ner_dataloader.save_autoencoder(os.path.join(config['save_dir'], 'autoencoder.weights'))
    config['label_size'] = ner_dataloader.label_size
    print()

    logging.basicConfig(
        filename=config["save_dir"]+'/ner_results.log',
        level=logging.INFO,
        format=' %(message)s',
        filemode='w',
    )

    print("Begin training AGN")
    accuracy_list = []
    macro_f1_list = []
    micro_f1_list = []
    for idx in range(1, config['iterations'] + 1):
        print(f"Starting iteration {idx}")
        train_dataset = NerDataset(ner_dataloader.train_set)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)

        val_dataset = NerDataset(ner_dataloader.dev_set)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

        model = AGNModel(config).to(device)

        callback = train_agn_model(model, train_loader, val_loader, device, config=config)

        accuracy_list.append(callback.history["val_acc"])
        macro_f1_list.append(callback.history["val_macro_f1"])
        micro_f1_list.append(callback.history["val_micro_f1"])
        print(f"Iteration {idx} End.")
        print()

    model_class = AGNModel(config).to(device)
    test_dataset = NerDataset(ner_dataloader.test_set)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    avg_loss, acc, macro_f1, micro_f1, report = test_agn_model(model_class, test_loader, config)

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
