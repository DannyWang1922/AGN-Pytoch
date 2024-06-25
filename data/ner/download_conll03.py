import json

from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset("eriktks/conll2003")
print(dataset)

with open('original_dataset/train.json', 'w') as writer:
    for obj in tqdm(dataset['train']):
        writer.writelines(json.dumps({'id': obj['id'], 'tokens': obj['tokens'],
                                      "pos_tags": obj["pos_tags"], "chunk_tags": obj["chunk_tags"],
                                      "ner_tags": obj["ner_tags"]}, ensure_ascii=False) + '\n')

with open('original_dataset/validation.json', 'w') as writer:
    for obj in tqdm(dataset['validation']):
        writer.writelines(json.dumps({'id': obj['id'], 'tokens': obj['tokens'],
                                      "pos_tags": obj["pos_tags"], "chunk_tags": obj["chunk_tags"],
                                      "ner_tags": obj["ner_tags"]}, ensure_ascii=False) + '\n')

with open('original_dataset/test.json', 'w') as writer:
    for obj in tqdm(dataset['test']):
        writer.writelines(json.dumps({'id': obj['id'], 'tokens': obj['tokens'],
                                      "pos_tags": obj["pos_tags"], "chunk_tags": obj["chunk_tags"],
                                      "ner_tags": obj["ner_tags"]}, ensure_ascii=False) + '\n')
