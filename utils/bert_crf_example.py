import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torchcrf import CRF
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence


class BERT_BiLSTM_CRF(nn.Module):
    def __init__(self, model_name, tagset_size, hidden_dim):
        super(BERT_BiLSTM_CRF, self).__init__()
        self.bert = BertModel.from_pretrained(model_name, add_pooling_layer=False)
        self.bert_hidden_size = self.bert.config.hidden_size
        self.lstm = nn.LSTM(self.bert_hidden_size, hidden_dim // 2, num_layers=2, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, sentences, attention_mask):
        bert_output = self.bert(sentences, attention_mask)
        bert_sequence_output = bert_output.last_hidden_state
        lstm_out, _ = self.lstm(bert_sequence_output)
        emissions = self.hidden2tag(lstm_out)
        return emissions

    def loss(self, emissions, tags, mask):
        return -self.crf(emissions, tags, mask=mask)

    def predict(self, emissions, mask):
        return self.crf.decode(emissions, mask=mask)


def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    attention_mask = [item[1] for item in batch]
    tags = [item[2] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_mask = pad_sequence(attention_mask, batch_first=True)
    tags = pad_sequence(tags, batch_first=True, padding_value=-1)

    return input_ids, attention_mask, tags


vocab_size = 5000  # 词汇表大小
model_name = 'bert-base-uncased'
tagset_size = 9  # 标签数量
hidden_dim = 768  # 双层768维BiLSTM

model = BERT_BiLSTM_CRF(model_name, tagset_size, hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)  # 调整学习率

# 准备数据
sentences = ["EU rejects German call to boycott British lamb.", "Peter Blackburn"]
tags = torch.tensor([[3, 0, 7, 0, 0, 0, 7, 0, 0], [1, 2, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long)  # padded

tokenizer = BertTokenizer.from_pretrained(model_name)
inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=128)
attention_mask = inputs["attention_mask"]

# 创建 DataLoader
dataset = TensorDataset(inputs["input_ids"], attention_mask, tags)
dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

# 训练循环
num_epochs = 1
for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids, attention_mask, tags = batch

        tag_space = model(input_ids, attention_mask)
        tag_space = tag_space[:, 1:-1, :]  # remove 101 and 102
        attention_mask = attention_mask[:, 1:-1].bool()

        # 计算损失
        loss = model.loss(tag_space, tags, attention_mask)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# 预测
with torch.no_grad():
    for batch in dataloader:
        input_ids, attention_mask, _ = batch
        tag_space = model(input_ids, attention_mask)
        tag_space = tag_space[:, 1:-1, :]  # remove 101 and 102
        attention_mask = attention_mask[:, 1:-1].bool()
        predictions = model.predict(tag_space, attention_mask.bool())
        print("Predictions:", predictions)
