# import os
#
# # 循环从0到1.0，每个间隔为0.1
# for i in range(0, 55, 5):
#     valve_rate = i * 0.01
#     print()
#     print("=================================================================")
#     print(f"Running experiment with valve_rate={valve_rate}")
#     os.system(f"python run_ner.py --config data/ner/conll2003.json --valve_rate {valve_rate}")


import itertools
import os

# # 定义每个属性的值列表
# learning_rate_list = [2e-05, 1e-05]
# batch_size_list = [32, 64, 128]
# decay_rate_list = [0.8, 0.85]
# decay_steps_list = [750, 1000, 1250]
# weight_decay_list = [0, 0.0001, 0.001]

# 定义每个属性的值列表
learning_rate_list = [2e-05]
batch_size_list = [8, 16, 32, 128]
decay_rate_list = [0.8, 0.85]
decay_steps_list = [250, 500, 750, 1000, 1500, 2000, 3000]
weight_decay_list = [0]

# 生成所有组合用于网格搜索
combinations = itertools.product(learning_rate_list, batch_size_list, decay_steps_list,
                                 decay_rate_list, weight_decay_list)

for combination in combinations:
    learning_rate, batch_size, decay_steps, decay_rate, weight_decay = combination

    print()
    print("=================================================================")
    print(
        f"Running experiment with learning_rate={learning_rate}, batch_size={batch_size}, decay_steps={decay_steps}, decay_rate={decay_rate}, weight_decay={weight_decay}")
    # cmd = (
    #     f"python run_ner.py --config data/ner/conll2003.bert.json "
    #     f" --learning_rate {learning_rate} "
    #     f"--batch_size {batch_size} --decay_steps {decay_steps} "
    #     f"--decay_rate {decay_rate} --weight_decay {weight_decay}"
    # )
    # os.system(cmd)
