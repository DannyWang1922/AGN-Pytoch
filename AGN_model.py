from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertModel, BertConfig, AdamW
import torch.nn.functional as F

import torch
from torch.utils.data import Dataset

from metrics import ClfMetrics
from utils import GlobalMaxPool1D, ConditionalLayerNormalization, Swish, move_to_device


class AGNPaddedDataset(Dataset):
    def __init__(self, data):
        self.data = data
        # 分别计算每种类型的最大长度
        self.max_token_length = max(len(item['token_ids']) for item in data)
        self.max_segment_length = max(len(item['segment_ids']) for item in data)
        self.max_tcol_length = max(len(item['tcol_ids']) for item in data)

        # 对整个数据集进行预填充
        self.padded_data = [self.pad_item(item) for item in data]

    def pad_item(self, item):
        # token_ids
        token_ids_tensor = item['token_ids']
        token_padding_length = self.max_token_length - len(item['token_ids'])
        if token_padding_length > 0:
            padded_token_ids = torch.cat(
                (token_ids_tensor, torch.zeros(token_padding_length, dtype=token_ids_tensor.dtype)), dim=0)
            # 创建 token_attention_mask
            token_attention_mask = torch.cat(
                (torch.ones(len(item['token_ids']), dtype=torch.bool),
                 torch.zeros(token_padding_length, dtype=torch.bool)), dim=0)
        else:
            padded_token_ids = token_ids_tensor
            token_attention_mask = torch.ones(len(item['token_ids']), dtype=torch.bool)

        # segment_ids
        segment_ids_tensor = item['segment_ids']
        segment_padding_length = self.max_segment_length - len(item['segment_ids'])
        if segment_padding_length > 0:
            padded_segment_ids = torch.cat(
                (segment_ids_tensor, torch.zeros(segment_padding_length, dtype=segment_ids_tensor.dtype)), dim=0)
        else:
            padded_segment_ids = segment_ids_tensor

        # tcol_ids
        tcol_ids_tensor = item['tcol_ids']
        tcol_padding_length = self.max_tcol_length - len(item['tcol_ids'])
        if tcol_padding_length > 0:
            padded_tcol_ids = torch.cat(
                (tcol_ids_tensor, torch.zeros(tcol_padding_length, dtype=tcol_ids_tensor.dtype)), dim=0)

        else:
            padded_tcol_ids = tcol_ids_tensor

        return {
            'token_ids': padded_token_ids,
            'segment_ids': padded_segment_ids,
            'tcol_ids': padded_tcol_ids,
            'label_id': torch.tensor(item['label_id']),
            'attention_mask': token_attention_mask,
        }

    def __len__(self):
        return len(self.padded_data)

    def __getitem__(self, idx):
        return self.padded_data[idx]


class SelfAttention(nn.Module):
    def __init__(self, feature_size, units=None, return_attention=False, is_residual=True,
                 activation="swish", dropout_rate=0.0):
        super(SelfAttention, self).__init__()
        self.feature_size = feature_size
        self.units = units if units is not None else feature_size
        self.return_attention = return_attention
        self.is_residual = is_residual
        self.activation = activation
        self.dropout_rate = dropout_rate

        # 定义Q, K, V和输出的线性变换层
        self.q_dense = nn.Linear(feature_size, self.units)
        self.k_dense = nn.Linear(feature_size, self.units)
        self.v_dense = nn.Linear(feature_size, self.units)
        self.o_dense = nn.Linear(self.units, feature_size)

        # 可选的残差连接和层归一化
        if self.is_residual:
            self.layernorm = nn.LayerNorm(feature_size)

        # Dropout 层
        self.dropout = nn.Dropout(dropout_rate)

        # 激活函数
        if activation == "swish":
            self.activation_func = Swish(beta=1.0)
        else:
            self.activation_func = F.relu  # 默认激活函数

    def forward(self, inputs, mask=None):
        if isinstance(inputs, list):
            q, k, v = inputs
        else:
            q = inputs.clone()
            k = inputs.clone()
            v = inputs.clone()

        qw = self.q_dense(q)
        kw = self.k_dense(k)
        vw = self.v_dense(v)

        # 计算注意力分数
        attention_scores = torch.bmm(qw, kw.transpose(1, 2))
        attention_scores = attention_scores / (self.units ** 0.5)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 应用 Dropout
        attention_weights = self.dropout(attention_weights)

        # 计算输出
        out = torch.bmm(attention_weights, vw)
        out = self.o_dense(out)

        # 应用激活函数
        if self.activation_func:
            out = self.activation_func(out)

        # 残差连接和层归一化
        if self.is_residual:
            out = out + q
            out = self.layernorm(out)

        # 根据 flag 返回注意力权重
        if self.return_attention:
            return out, attention_weights
        return out


class AGN(nn.Module):
    def __init__(self, feature_size, activation='swish', dropout_rate=0.1, valve_rate=0.3, dynamic_valve=False):
        super(AGN, self).__init__()
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.valve_rate = valve_rate
        self.dynamic_valve = dynamic_valve

        self.valve_transform = nn.Linear(feature_size, feature_size, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 动态阀门调整
        if self.dynamic_valve:
            self.dynamic_valve_layer = nn.Dropout(1.0 - self.valve_rate)

        self.dropout = nn.Dropout(self.dropout_rate)

        self.attn = SelfAttention(feature_size, activation=self.activation, dropout_rate=self.dropout_rate,
                                  return_attention=True)

    def forward(self, x, gi):
        valve = self.sigmoid(self.valve_transform(x))
        if self.dynamic_valve:
            valve = self.dynamic_valve_layer(valve)
        else:
            valve_mask = (valve > 0.5 - self.valve_rate) & (valve < 0.5 + self.valve_rate)
            valve = valve * valve_mask.float()

        enhanced = x + valve * gi
        enhanced = self.dropout(enhanced)
        output, attn_weights = self.attn(enhanced)
        return output, attn_weights


class AGNModel(nn.Module):
    def __init__(self, config, task='clf'):
        super(AGNModel, self).__init__()
        self.config = config
        self.task = task

        # 加载预训练的 BERT 模型
        self.bert = BertModel.from_pretrained(config['pretrained_model_dir'])
        feature_size = self.bert.config.hidden_size

        # GI 层
        self.gi_linear = nn.Linear(self.config["ae_latent_dim"], feature_size)
        self.gi_dropout = nn.Dropout(self.config.get('dropout_rate', 0.1))

        # AGN 层
        self.agn = AGN(feature_size=feature_size, activation='swish',
                       dropout_rate=self.config.get('dropout_rate', 0.1),
                       valve_rate=self.config.get('valve_rate', 0.3),
                       dynamic_valve=self.config.get('use_dynamic_valve', False))

        if self.task == 'clf':
            self.output_layer = nn.Sequential(
                nn.Linear(feature_size, config.get('hidden_size', 256)),
                nn.ReLU(),
                nn.Dropout(config.get('dropout_rate', 0.1)),
                nn.Linear(config.get('hidden_size', 256), config['output_size']),
                nn.Softmax(dim=-1)
            )
        # elif self.task == 'ner':
        #     self.output_dense = nn.Linear(feature_size, self.config['output_size'])
        #     # 假设 CRF 层已经定义
        #     self.crf = CRF(self.config['output_size'])
        elif self.task == 'sts':
            # 定义回归或其他输出层
            pass

    def forward(self, input):
        token_ids, segment_ids, gi, attention_mask = input["token_ids"], input["segment_ids"], input["tcol_ids"], input[
            "attention_mask"]
        outputs = self.bert(input_ids=token_ids, token_type_ids=segment_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # torch.Size([64, 65, 768])

        # GI 处理
        gi = self.gi_linear(gi)
        gi = self.gi_dropout(gi)
        gi = gi.unsqueeze(1)  # 扩展维度以匹配序列维度 gi shape: torch.Size([64, 1, 768])

        # AGN 处理
        agn_output, attn_weight = self.agn(sequence_output, gi)

        if self.task == 'clf':
            pooled_output = torch.max(agn_output, dim=1)[0]
            output = self.output_layer(pooled_output)

            # output = self.output_activation(output)
        elif self.task == 'ner':
            output = self.output_linear(agn_output)
            output = self.crf(output)
        elif self.task == 'sts':
            # 处理回归任务的输出
            pass
        return output





def update_learning_rate(optimizer, decay_rate=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_rate


# def train_agn_model(model, data_loader, device, epochs=10, learning_rate=1e-5):
#     loss_fn = torch.nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     model.train()
#     step = 0
#     decay_steps = 100
#
#     for epoch in range(epochs):
#         total_loss = 0
#         # Create progress bar using tqdm
#         progress_bar = tqdm(data_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=True)
#         for data in progress_bar:
#             data = move_to_device(data, device)
#             inputs, labels = data, data["label_id"]
#
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = loss_fn(outputs, labels)
#             total_loss += loss.item()
#
#             loss.backward()
#             optimizer.step()
#
#             progress_bar.set_postfix(loss=loss.item())
#             step += 1
#             if step % decay_steps == 0:  # update learning rate
#                 update_learning_rate(optimizer)
#                 # print(f"Step {step}: Learning rate decayed to {optimizer.param_groups[0]['lr']}")
#         avg_loss = total_loss / len(data_loader)
#         print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")

def train_agn_model(model, train_loader, val_loader, device, epochs=10, learning_rate=1e-5, save_path='best_model.pth'):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    accuracy_list = []
    f1_list = []

    # 初始化回调
    callback = ClfMetrics(model, val_loader, device, save_path, epochs)

    step = 0
    decay_steps = 100

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=True)

        for data in progress_bar:
            data = move_to_device(data, device)
            inputs, labels = data, data["label_id"]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            step += 1
            if step % decay_steps == 0:  # update learning rate
                update_learning_rate(optimizer)
                # print(f"Step {step}: Learning rate decayed to {optimizer.param_groups[0]['lr']}")

        avg_loss = total_loss / len(train_loader)
        callback.on_epoch_end(epoch, avg_loss)

        # 检查是否需要提前停止训练
        if hasattr(model, 'stop_training') and model.stop_training:
            print("Training stopped early.")
            break

    return max(callback.history['val_acc']), max(callback.history['val_f1'])

    #     accuracy = max(callback.history["val_acc"])
    #     f1 = max(callback.history["val_f1"])
    #     accuracy_list.append(accuracy)
    #     f1_list.append(f1)
    #     log = f"iteration {epoch} accuracy: {accuracy}, f1: {f1}\n"
    #     print(log)
    #
    # print("Average accuracy:", sum(accuracy_list) / len(accuracy_list))
    # print("Average f1:", sum(f1_list) / len(f1_list))


def test_agn_model(model, val_loader, device, loss_fn):
    model.eval()  # 设置模型为评估模式
    total_val_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data in val_loader:
            data = move_to_device(data, device)
            inputs, labels = data, data["label_id"]
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_val_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = total_correct / total_samples
    print(f'Validation Loss: {avg_val_loss}, Accuracy: {val_accuracy}')
    return avg_val_loss, val_accuracy
