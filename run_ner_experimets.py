import os

# config_list = [
#                "data/ner/conll2003_AGN_ae_sigmoid.json",
#                "data/ner/conll2003_AGN_vae_sigmoid.json",
#                "data/ner/conll2003_AGN_none_sigmoid.json",
#                "data/ner/conll2003_AGN_ae_softmax.json",
#                "data/ner/conll2003_AGN_none_softmax.json"]

config1 = "data/ner/conll2003_AGN_none_softmax.json"
config2 = "data/ner/conll2003_AGN_ae_softmax.json"
config3 = "data/ner/conll2003_AGN_vae_sigmoid.json"
config = config1

# 循环从0到1.0，每个间隔为0.1
for i in range(0, 10):
    valve_rate = i * 0.1
    print("=================================================================")
    print(f"Running experiment with config={config}, valve_rate {valve_rate}")
    os.system(f"python run_ner.py --ae_epochs 100 --epochs 100 --config {config} --valve_rate_sigmoid {valve_rate} --valve_rate_softmax {valve_rate}")
    print()


# for i in config_list:
#     print("=================================================================")
#     print(f"Running experiment with config={i}")
#     os.system(f"python run_ner.py --ae_epochs 100 --epochs 100 --config {i} ")
#     print()

# # 循环从0到1.0，每个间隔为0.1
# for i in range(0, 55, 5):
#     valve_rate = i * 0.01
#     print()
#     print("=================================================================")
#     print(f"Running experiment with valve_rate={valve_rate}")
#     os.system(f"python run_ner.py --config data/ner/conll2003_AGN.json --valve_rate {valve_rate}")

# seed_list = [12, 37, 52, 67, 80]
# for i in seed_list:
#     print("=================================================================")
#     print(f"Running experiment with seed={i}")
#     os.system(f"python run_ner.py --config data/ner/conll2003_AGN.json --random_seed {i}")


# import itertools
# import os
#
# # 定义每个属性的值列表
# learning_rate_list = [2e-05]
# batch_size_list = [8, 16, 32, 128]
# decay_rate_list = [0.8, 0.85]
# decay_steps_list = [250, 500, 750, 1000, 1500, 2000, 3000]
# weight_decay_list = [0]
#
# # 生成所有组合用于网格搜索
# combinations = itertools.product(learning_rate_list, batch_size_list, decay_steps_list,
#                                  decay_rate_list, weight_decay_list)
#
# for combination in combinations:
#     learning_rate, batch_size, decay_steps, decay_rate, weight_decay = combination
#
#     print()
#     print("=================================================================")
#     print(
#         f"Running experiment with learning_rate={learning_rate}, batch_size={batch_size}, decay_steps={decay_steps}, decay_rate={decay_rate}, weight_decay={weight_decay}")
#     cmd = (
#         f"python run_ner.py --config data/ner/conll2003_bert.json "
#         f" --learning_rate {learning_rate} "
#         f"--batch_size {batch_size} --decay_steps {decay_steps} "
#         f"--decay_rate {decay_rate} --weight_decay {weight_decay}"
#     )
#     os.system(cmd)
