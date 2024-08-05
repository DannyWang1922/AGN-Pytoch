import os

# config_list = ["data/ner/conll2003_bert.json",
#                "data/ner/conll2003_AGN_ae_sigmoid.json",
#                "data/ner/conll2003_AGN_vae_sigmoid.json",
#                "data/ner/conll2003_AGN_none_sigmoid.json",
#                "data/ner/conll2003_AGN_ae_softmax.json",
#                "data/ner/conll2003_AGN_vae_softmax.json",
#                "data/ner/conll2003_AGN_none_softmax.json"]


seed_list = [37, 42, 52, 67, 80]
for i in seed_list:
    print("=================================================================")
    print(f"Running experiment with seed={i}")
    os.system(f"python run_ner.py --ae_epochs 1 --epochs 100 --config data/ner/conll2003_bert.json --random_seed {i}")
