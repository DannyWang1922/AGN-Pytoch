from collections import defaultdict
import numpy as np


def generate_tcol_vectors(all_tokens, all_labels):
    # 提取所有可能的标签
    possible_labels = set(label for labels in all_labels for label in labels)
    num_labels = len(possible_labels)
    token_label_counts = defaultdict(lambda: defaultdict(int))


    # 扁平化token和标签列表
    tokens = [word_token for sentence_token in all_tokens for word_token in sentence_token]
    labels = [word_label for sentence_token in all_labels for word_label in sentence_token]

    # 统计每个token对应的标签频率
    for token, label in zip(tokens, labels):
        token_label_counts[token][label] += 1

    # 生成TCol向量
    token_tcol = {}
    token_tcol[0] = np.array([0] * num_labels)  # pad
    token_tcol[0] = np.reshape(token_tcol[0], (1, -1))

    for token, label_counts in token_label_counts.items():
        total_counts = sum(label_counts.values())
        vector = [label_counts[label] / total_counts for label in sorted(possible_labels)]
        token_tcol[token] = np.reshape(np.array(vector), (1, -1))

    return possible_labels, token_tcol  # {'I': [0.0, 0.0, 1.0], 'live': [0.0, 0.0, 1.0]}

def sentence_tcol_vectors(all_tokens, token_tcol, max_len=128):
    tcol_vectors = []
    for obj in all_tokens:
        padded = [0] * (max_len - len(obj))
        token_ids = obj + padded
        obj_tcol = np.concatenate([token_tcol.get(token) for token in token_ids])
        obj_tcol = np.reshape(obj_tcol, (1, -1))
        tcol_vectors.append(obj_tcol)
    return tcol_vectors


if __name__ == '__main__':
    all_tokens = [
        ["I", "live", "in", "New", "York", "New"],
        ["He", "is", "from", "Japan"],
        ["Visit", "New", "today"]
    ]

    all_labels = [
        ["O", "O", "O", "B-LOC", "I-LOC", "O"],
        ["O", "O", "O", "B-LOC"],
        ["O", "I-LOC", "O"]
    ]

    # 运行函数并打印结果
    possible_labels, tcol_vectors = generate_tcol_vectors(all_tokens, all_labels)
    print(possible_labels)
    print(tcol_vectors)


    sentence_vectors = sentence_tcol_vectors(all_tokens, tcol_vectors)
    # print(sentence_vectors)

    # 打印每个句子的TCol向量
    # for i, sentence_vector in enumerate(sentence_vectors):
    #     print(f"Sentence {i + 1} TCol Vectors: {sentence_vector.shape}")
