import os

# config_list = ["data/ner/conll2003_bert.json",
#                "data/ner/conll2003_AGN_ae_sigmoid.json",
#                "data/ner/conll2003_AGN_vae_sigmoid.json",
#                "data/ner/conll2003_AGN_none_sigmoid.json",
#                "data/ner/conll2003_AGN_ae_softmax.json",
#                "data/ner/conll2003_AGN_vae_softmax.json",
#                "data/ner/conll2003_AGN_none_softmax.json"]

seed_list = [37, 42, 52, 67, 80]
batch_size_list = [16, 32]
learning_rate_list = [5e-5, 3e-5, 2e-5]
for seed in seed_list:
    for bs in batch_size_list:
        for lr in learning_rate_list:
            print("=================================================================")
            print(f"Running experiment with seed={seed}, bs={bs}, lr={lr}")
            os.system(f"python run_ner.py --ae_epochs 1 --epochs 1 --config data/ner/conll2003_bert.json --batch_size {bs} --learning_rate {lr} --random_seed {seed}")
