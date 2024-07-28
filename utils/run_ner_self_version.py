import argparse
import json
import numpy as np
import random
from transformers import BertTokenizer
import os
import torch
from ner_dataloader_tcol_self_version import load_conll, parse_tcol_ids, train_V_Net


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
    # parser = argparse.ArgumentParser(description='AGN-Plus Configuration')
    # parser.add_argument('--config', type=str, default="data/sst2/sst2.json")
    # args = parser.parse_args()
    # config_file = args.config

    # config_file = "data/sst2/sst2.json"
    config_file = "../data/ner/conll2003_AGN_none_sigmoid.json"

    with open(config_file, "r") as reader:
        config = json.load(reader)
    formatted_json = json.dumps(config, indent=4, sort_keys=True)  # json read friendly format
    # print("Config:")
    # print(formatted_json)
    # print()
    config["device"] = device

    # create save_dir folder if not exists
    if not os.path.exists(config['save_dir']):
        os.makedirs(config['save_dir'])

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(config['pretrained_model_dir'], do_lower_case=True)
    tokenizer.model_max_length = config['max_len']

    print("Loading data...")
    data = load_conll(config['train_path'], tokenizer)

    print("Loading sentence tcol...")
    data_with_raw_tcol = parse_tcol_ids(data)

    print("Training V-Net...")
    data_with_trained_tcol = train_V_Net(data_with_raw_tcol, config)
    print(data_with_trained_tcol[0])


if __name__ == '__main__':
    set_seed(42)
    device = check_device()
    main(device)