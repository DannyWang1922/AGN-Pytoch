import logging
import os
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch import optim
import torch.nn as nn
from tqdm import tqdm
from transformers import BertModel
import torch.nn.functional as F
import torch
from clf_metrics import ClfMetrics
from ner_metrics import NerMetrics
from utils import GlobalMaxPool1D, ConditionalLayerNormalization, Swish, move_to_device, set_seed


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
    def __init__(self, feature_size, dropout_rate=0.1, valve_rate=0.3, dynamic_valve=False):
        super(AGN, self).__init__()
        self.dropout_rate = dropout_rate
        self.valve_rate = valve_rate
        self.dynamic_valve = dynamic_valve

        self.valve_transform = nn.Linear(feature_size, feature_size, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 动态阀门调整
        if self.dynamic_valve:
            self.dynamic_valve_layer = nn.Dropout(1.0 - self.valve_rate)

        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x, gi):
        valve = self.sigmoid(self.valve_transform(x))
        if self.dynamic_valve:
            valve = self.dynamic_valve_layer(valve)
        else:
            valve_mask = (valve > 0.5 - self.valve_rate) & (valve < 0.5 + self.valve_rate)
            valve = valve * valve_mask.float()

        enhanced = x + valve * gi
        enhanced = self.dropout(enhanced)
        return enhanced


class AGNModel(nn.Module):
    def __init__(self, config):
        super(AGNModel, self).__init__()
        self.config = config
        self.task = config["task"]

        # 加载预训练的 BERT 模型
        self.bert = BertModel.from_pretrained(config['pretrained_model_dir'])
        bert_output_feature_size = self.bert.config.hidden_size

        # GI 层
        self.gi_linear = nn.Linear(self.config["ae_latent_dim"], bert_output_feature_size)
        self.gi_dropout = nn.Dropout(self.config.get('dropout_rate', 0.1))

        # AGN 层
        self.agn = AGN(feature_size=9,
                       dropout_rate=0.1,
                       valve_rate=self.config.get('valve_rate', 0.3),
                       dynamic_valve=self.config.get('use_dynamic_valve', False))

        self.attn = SelfAttention(9, activation="swish", dropout_rate=config['dropout_rate'],
                                  return_attention=False)

        if self.task == 'clf':
            self.loss_fn = nn.CrossEntropyLoss()
            self.clf_output_layer = nn.Sequential(
                nn.Linear(bert_output_feature_size, config.get('hidden_size', 256)),
                nn.ReLU(),
                nn.Dropout(config['dropout_rate']),
                nn.Linear(config["hidden_size"], config['output_size']),
                nn.Softmax(dim=-1)
            )
        elif self.task == 'ner':
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            self.ner_drop = nn.Dropout(config['dropout_rate'])
            self.ner_linear = nn.Linear(bert_output_feature_size, config["hidden_size"])
            self.ner_hidden2tag = nn.Linear(config["hidden_size"], config["label_size"])

        elif self.task == 'sts':
            # 定义回归或其他输出层
            pass

    def forward(self, input):
        if self.task == 'clf':
            token_ids, segment_ids, gi, attention_mask = input["token_ids"], input["segment_ids"], input["tcol_ids"], \
                input["attention_mask"]
        else:
            token_ids, segment_ids, gi, attention_mask = input["token_ids"], input["segment_ids"], input[
                "sf_vector"], input["attention_mask"]

        bert_output = self.bert(input_ids=token_ids, token_type_ids=segment_ids, attention_mask=attention_mask)
        bert_last_hidden_state = bert_output.last_hidden_state  # torch.Size([64, 65, 768])

        output1 = self.ner_drop(bert_last_hidden_state)
        output2 = self.ner_linear(output1)
        token_emb = self.ner_hidden2tag(output2)

        if self.config.get('use_agn'):
            agn_output = self.agn(token_emb, gi)
            preds = self.attn(agn_output)
        else:
            preds = token_emb

        return preds

    def loss(self, outputs, targets):
        if self.task == 'clf':
            loss = self.loss_fn(outputs, targets)
        elif self.task == "ner":
            active_loss = targets.view(-1) != -100  # 找到有效的标签
            active_logits = outputs.view(-1, outputs.shape[-1])[active_loss]
            active_labels = targets.view(-1)[active_loss]
            loss = self.loss_fn(active_logits, active_labels)
        return loss


def update_learning_rate(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_rate
        print(">>>Updated learning rate: {}".format(param_group['lr']))
        logging.info("Updated learning rate: {}".format(param_group['lr']))


def train_agn_model(model, train_loader, evl_loader, config):
    epochs = config["epochs"]
    save_dir = config["save_dir"]
    learning_rate = config["learning_rate"]
    device = config["device"]
    decay_steps = config["decay_steps"]
    weight_decay = config["weight_decay"]

    if config["task"] == "ner":
        callback = NerMetrics(model=model, eval_data_loader=evl_loader, config=config)
    else:
        callback = ClfMetrics(model, evl_loader, device, save_dir, epochs)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    step = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=True)

        for data in progress_bar:
            data = move_to_device(data, device)
            inputs, labels = data, data["label_ids"]

            optimizer.zero_grad()
            preds = model(inputs)

            loss = model.loss(preds, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            step += 1
            if step % decay_steps == 0:  # update learning rate
                update_learning_rate(optimizer, config["decay_rate"])

        avg_loss = total_loss / len(train_loader)
        callback.on_epoch_end(epoch, avg_loss)

        # 检查是否需要提前停止训练
        if hasattr(model, 'stop_training') and model.stop_training:
            print("Training stopped early.")
            break

    return callback


def test_agn_model(model_class, test_loader, config):
    # 定义模型保存路径
    model_path = os.path.join("ckpts/conll03_bert/conll03_bert_1", "AGN_weights.pth")

    # 确保模型文件存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    device = config["device"]
    # 加载模型
    model = model_class
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.to(device)
    model.eval()

    # 定义损失函数
    if config["task"] == "ner":
        loss_fn = model.crf
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    total_loss = 0
    y_true, y_pred = [], []
    progress_bar = tqdm(test_loader, desc="Testing", leave=True)

    with torch.no_grad():
        for data in progress_bar:
            data = move_to_device(data, device)
            inputs, labels = data, data["label_ids"]

            outputs = model(inputs)
            mask = inputs["attention_mask"]

            if config["task"] == "ner":
                loss = loss_fn.compute_loss(outputs, labels, mask)
                preds = model.crf.decode(outputs, mask)
            else:
                loss = loss_fn(outputs, labels)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

            total_loss += loss.item()

            if config["task"] == "ner":
                for i in range(len(labels)):
                    masked_true_labels = labels[i][labels[i] != -100].cpu().numpy()
                    y_true.extend(masked_true_labels)
                    y_pred.extend(preds[i])
            else:
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds)

    avg_loss = total_loss / len(test_loader)

    # 计算评估指标
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    report = classification_report(y_true, y_pred, zero_division=0)

    return avg_loss, acc, macro_f1, micro_f1, report
