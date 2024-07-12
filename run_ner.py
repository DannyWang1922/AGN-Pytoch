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
from utils import get_save_dir, set_seed


def check_device():
    """Check for device"""
    if torch.cuda.is_available():
        print("CUDA is available.")
        device = torch.device("cuda:0")
    else:
        print("Using CPU.")
        device = torch.device("cpu")
    return device


def main():
    parser = argparse.ArgumentParser(description='AGN-Plus Configuration')
    parser.add_argument('--config', type=str, default="data/ner/conll2003.bert.json")
    parser.add_argument('--valve_rate', type=float, default=0.4, help='Valve rate parameter')
    args = parser.parse_args()
    config_file = args.config

    set_seed(42)
    device = check_device()

    # Load config
    # config_file = "data/ner/conll2003.json"
    # config_file = "data/ner/conll2003.bert.json"

    with open(config_file, "r") as reader:
        config = json.load(reader)
    if args.valve_rate is not None:
        config["valve_rate"] = args.valve_rate
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
    ner_dataloader = NerDataLoader(config['dataset_name'],
                                   tokenizer=tokenizer,
                                   feature=config["feature"],
                                   device=config["device"],
                                   max_len=config['max_len'],
                                   ae_latent_dim=config['ae_latent_dim'], use_vae=config['use_vae'],
                                   batch_size=config["batch_size"],
                                   ae_epochs=config['ae_epochs'])
    ner_dataloader.set_train(config['train_path'])
    ner_dataloader.set_test(config['test_path'])
    ner_dataloader.save_autoencoder(os.path.join(config['save_dir'], 'autoencoder.weights'))
    config['label_size'] = ner_dataloader.label_size
    print()
    set_seed(42)

    logging.basicConfig(
        filename=config["save_dir"] + '/ner_results.log',
        level=logging.INFO,
        format=' %(message)s',
        filemode='w',
    )

    print("Begin training AGN")
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    for idx in range(1, config['iterations'] + 1):
        print(f"Begin iteration {idx}")
        train_dataset = NerDataset(ner_dataloader.train_set.data[:128])
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)

        test_dataset = NerDataset(ner_dataloader.test_set.data[:128])
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

        model = AGNModel(config).to(device)

        callback = train_agn_model(model=model, train_loader=train_loader, evl_loader=test_loader, config=config)

        max_acc = max(callback.history["val_acc"])
        max_precision = max(callback.history["val_precision"])
        max_recall = max(callback.history["val_recall"])
        max_f1 = max(callback.history["val_f1"])

        accuracy_list.append(max_acc)
        precision_list.append(max_precision)
        recall_list.append(max_recall)
        f1_list.append(max_f1)

        print(
            f"Iteration {idx} finish. acc: {max_acc}, precision: {max_precision}, recall: {max_recall}, f1: {max_f1}")
        logging.info(
            f"Iteration {idx}: acc: {max_acc}, precision: {max_precision}, recall: {max_recall}, f1: {max_f1} \n")
        print()

    avg_accuracy = round(sum(accuracy_list) / len(accuracy_list), 4)
    avg_precision = round(sum(precision_list) / len(precision_list), 4)
    avg_recall = round(sum(recall_list) / len(recall_list), 4)
    avg_f1 = round(sum(f1_list) / len(f1_list), 4)

    logging.info("avg_accuracy: " + str(avg_accuracy))
    logging.info("avg_precision: " + str(avg_precision))
    logging.info("avg_recall: " + str(avg_recall))
    logging.info("avg_f1: " + str(avg_f1))

    logging.info("\n" + str(formatted_json))


if __name__ == '__main__':
    main()
