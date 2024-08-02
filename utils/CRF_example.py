import torch
import torch.nn as nn
from torchcrf import CRF
import torch.optim as optim
from sklearn.metrics import classification_report
import numpy as np


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, sentences):
        embeds = self.embedding(sentences)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.hidden2tag(lstm_out)
        return emissions

    def loss(self, emissions, tags, mask):
        return -self.crf(emissions, tags, mask=mask)

    def predict(self, emissions, mask):
        return self.crf.decode(emissions, mask=mask)


# 示例用法
def flatten(tensor, mask):
    return [item for sublist, m in zip(tensor, mask) for item, mm in zip(sublist, m) if mm]


def calculate_f1(model, sentences, tags, mask):
    model.eval()
    with torch.no_grad():
        emissions = model(sentences)
        predictions = model.predict(emissions, mask)

    y_true = flatten(tags, mask)
    y_pred = flatten(predictions, mask)

    report = classification_report(y_true, y_pred, output_dict=True)
    return report['weighted avg']['f1-score']


vocab_size = 5000  # 词汇表大小
tagset_size = 9  # 标签数
embedding_dim = 100
hidden_dim = 128

model = BiLSTM_CRF(vocab_size, tagset_size, embedding_dim, hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)  # 调整学习率
sentences = torch.randint(0, vocab_size, (32, 10))  # 示例输入
tags = torch.randint(0, tagset_size, (32, 10))  # 示例标签
mask = torch.ones((32, 10), dtype=torch.bool)  # 掩码

# 训练循环
for epoch in range(1):
    model.train()
    optimizer.zero_grad()
    emissions = model(sentences)
    loss = model.loss(emissions, tags, mask)
    loss.backward()
    optimizer.step()

    f1_score = calculate_f1(model, sentences, tags, mask)
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}, F1 Score: {f1_score:.4f}")