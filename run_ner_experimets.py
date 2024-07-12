import os

# 循环从0到1.0，每个间隔为0.1
for i in range(0, 55, 5):
    valve_rate = i * 0.01
    print()
    print("=================================================================")
    print(f"Running experiment with valve_rate={valve_rate}")
    os.system(f"python run_ner.py --config data/ner/conll2003.json --valve_rate {valve_rate}")