# import os
# import json
# import re
# import pandas as pd
#
#
# def extract_config_and_avg_f1(log_file_path):
#     with open(log_file_path, 'r') as file:
#         content = file.read()
#
#     # Extract the config block
#     config_match = re.search(r'Config:\s*(\{.*?\})', content, re.DOTALL)
#     if not config_match:
#         return None
#
#     config_str = config_match.group(1)
#     config = json.loads(config_str)
#
#     # Extract avg_f1
#     avg_f1_match = re.search(r'avg_f1:\s*([\d.]+)', content)
#     if not avg_f1_match:
#         return None
#
#     avg_f1 = float(avg_f1_match.group(1))
#
#     # Extract required fields from config
#     extracted_data = {
#         "batch_size": config.get("batch_size"),
#         "decay_rate": config.get("decay_rate"),
#         "decay_steps": config.get("decay_steps"),
#         "learning_rate": config.get("learning_rate"),
#         "valve_rate": config.get("valve_rate"),
#         "weight_decay": config.get("weight_decay"),
#         "avg_f1": avg_f1,
#     }
#
#     return extracted_data
#
#
# def calculate_slope(batch_size, decay_rate, decay_steps):
#     total_step_in_one_epoch = 14041 // batch_size
#     num_decay = total_step_in_one_epoch / decay_steps
#     num_epoch_of_once_decay = 1 / num_decay
#     return f"{num_epoch_of_once_decay:.2f} {decay_rate}"
#
#
# def traverse_and_collect_data(root_dir):
#     data = []
#     for experiment in os.listdir(root_dir):
#         experiment_path = os.path.join(root_dir, experiment)
#         if os.path.isdir(experiment_path):
#             for subdir, _, files in os.walk(experiment_path):
#                 if 'ner_results.log' in files:
#                     log_file_path = os.path.join(subdir, 'ner_results.log')
#                     extracted_data = extract_config_and_avg_f1(log_file_path)
#                     if extracted_data:
#                         extracted_data["subdir"] = subdir
#                         extracted_data["lr_decay_slope"] = calculate_slope(
#                             extracted_data["batch_size"],
#                             extracted_data["decay_rate"],
#                             extracted_data["decay_steps"]
#                         )
#                         data.append(extracted_data)
#     return data
#
#
# def save_to_csv(data, output_csv):
#     df = pd.DataFrame(data)
#
#     # Sort the DataFrame by the specified columns
#     sort_columns = ["batch_size", "decay_rate", "decay_steps", "learning_rate", "weight_decay"]
#     df = df.sort_values(by=sort_columns)
#
#     df.to_csv(output_csv, index=False)
#
#
#
# if __name__ == "__main__":
#     root_dir = "Experiment_1.8"
#     output_csv = "Experiment_res_1.8.csv"
#     data = traverse_and_collect_data(root_dir)
#     save_to_csv(data, output_csv)
#     print(f"Data saved to {output_csv}")


import os
import json
import re
import pandas as pd


def extract_config_and_avg_f1(log_file_path, required_fields):
    with open(log_file_path, 'r') as file:
        content = file.read()

    # Extract the config block
    config_match = re.search(r'Config:\s*(\{.*?\})', content, re.DOTALL)
    if not config_match:
        return None

    config_str = config_match.group(1)
    config = json.loads(config_str)

    # Extract avg_f1
    avg_f1_match = re.search(r'avg_f1:\s*([\d.]+)', content)
    if not avg_f1_match:
        return None

    avg_f1 = float(avg_f1_match.group(1))

    # Extract required fields from config
    extracted_data = {}
    for field in required_fields:
        extracted_data[field] = config.get(field)

    extracted_data["avg_f1"] = avg_f1

    return extracted_data


def traverse_and_collect_data(root_dir, required_fields):
    data = []
    for experiment in os.listdir(root_dir):
        experiment_path = os.path.join(root_dir, experiment)
        if os.path.isdir(experiment_path):
            for subdir, _, files in os.walk(experiment_path):
                if 'ner_results.log' in files:
                    log_file_path = os.path.join(subdir, 'ner_results.log')
                    extracted_data = extract_config_and_avg_f1(log_file_path, required_fields)
                    if extracted_data:
                        extracted_data["subdir"] = subdir
                        data.append(extracted_data)
    return data


def save_to_csv(data, output_csv, required_fields):
    df = pd.DataFrame(data)

    # Sort the DataFrame by the specified columns
    sort_columns = [col for col in required_fields if col in df.columns]
    df = df.sort_values(by=sort_columns)

    # Move "avg_f1" and "subdir" to the last two columns
    columns = [col for col in df.columns if col not in ["avg_f1", "subdir"]]
    columns.append("avg_f1")
    columns.append("subdir")
    df = df[columns]

    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    root_dir = "Exp_3.2"
    output_csv = "Exp_3.2_res.csv"
    required_fields = ["batch_size",  "learning_rate", "random_seed"]
    data = traverse_and_collect_data(root_dir, required_fields)
    save_to_csv(data, output_csv, required_fields)
    print(f"Data saved to {output_csv}")