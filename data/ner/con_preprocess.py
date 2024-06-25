import json


def conll_preprocess(input_filename, output_filename):
    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        # 逐行读取输入文件
        for line in infile:
            try:
                # 将行解析为 JSON
                data = json.loads(line.strip())

                # 处理 tokens
                text = " ".join(data["tokens"])

                # 创建新的字典以存储更新后的数据
                updated_data = {
                    "id": data["id"],
                    "text": text,
                    "pos_tags": data["pos_tags"],
                    "chunk_tags": data["chunk_tags"],
                    "label": data["ner_tags"]  # 将 'ner_tags' 改名为 'label'
                }

                # 将更新后的数据转换为 JSON 字符串
                json_string = json.dumps(updated_data)

                # 将 JSON 字符串写入输出文件，每个对象后添加换行符
                outfile.write(json_string + '\n')

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")


conll_preprocess("original_dataset/train.json", "train.json")
conll_preprocess("original_dataset/validation.json", "validation.json")
conll_preprocess("original_dataset/test.json", "test.json")
