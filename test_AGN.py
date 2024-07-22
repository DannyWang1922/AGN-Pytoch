import json
import logging
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from model_AGN import AGNModel, test_agn_model
from ner_dataloader import NerDataset, NerDataLoader
from utils import set_seed

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


config_file = "data/ner/conll2003_bert.json"
with open(config_file, "r") as reader:
    config = json.load(reader)
config["task"] = "ner"
target_size = config["label_size"] = 9
config["device"] = device = "cpu"

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


model_path = "ckpts/conll03_bert/conll03_bert_1/autoencoder.weights"
state_dict = torch.load(model_path, map_location=torch.device(config['device']))
vae_in_dims = state_dict["encoder.encoder_linear.weight"].size(1)
ner_dataloader.load_autoencoder(model_path, vae_in_dims)
ner_dataloader.set_dev("data/ner/validation.json")
ner_dataloader.set_test(config['test_path'])
config['label_size'] = ner_dataloader.label_size
print()
set_seed(42)

val_dataset = NerDataset(ner_dataloader.dev_set)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
test_dataset = NerDataset(ner_dataloader.test_set)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

model_class = AGNModel(config).to(device)
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