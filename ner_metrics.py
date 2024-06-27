import logging
import os

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict
import torch
from sklearn.preprocessing import MultiLabelBinarizer

from utils import move_to_device
from sklearn.metrics import classification_report


class NerMetrics:
    def __init__(self, model, eval_data_loader, device, save_dir, epochs, min_delta=1e-4, patience=10):
        self.patience = patience
        self.min_delta = min_delta
        self.eval_data_loader = eval_data_loader
        self.save_dir = save_dir
        self.model = model
        self.device = device
        self.history = defaultdict(list)
        self.best = float('-inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.epochs = epochs

        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.mlb = MultiLabelBinarizer(classes=labels)
        self.mlb.fit(labels)  # 确保所有可能的标签都被考虑

    def calc_metrics(self):
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for data in self.eval_data_loader:
                data = move_to_device(data, self.device)
                inputs, true_labels = data, data["label_ids"]
                outputs = self.model(inputs)
                mask = inputs["attention_mask"]

                preds = self.model.crf.decode(outputs, mask)  # (seq_length, batch_size)

                for i in range(len(true_labels)):
                    # Apply mask to remove -1 padded true labels and corresponding predictions
                    masked_true_labels = true_labels[i][true_labels[i] != -1].cpu().numpy()
                    y_true.extend(masked_true_labels)
                    y_pred.extend(preds[i])
        # 计算并打印评估指标
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        micro_f1 = f1_score(y_true, y_pred, average="micro")
        report = classification_report(y_true, y_pred)
        # print(report)
        return acc, macro_f1, micro_f1

    def on_epoch_end(self, epoch, avg_loss):
        val_acc, val_macro_f1, val_micro_f1 = self.calc_metrics()
        self.history['val_acc'].append(val_acc)
        self.history['val_macro_f1'].append(val_macro_f1)
        self.history['val_micro_f1'].append(val_micro_f1)

        # 输出到控制台
        print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}, val_acc: {val_acc:.4f}, val_macro_f1: {val_macro_f1:.4f}, val_micro_f1: {val_micro_f1:.4f}")
        logging.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}, val_acc: {val_acc:.4f}, val_macro_f1: {val_macro_f1:.4f}, val_micro_f1: {val_micro_f1:.4f}")

        if val_macro_f1 > self.best + self.min_delta:
            self.best = val_macro_f1
            self.wait = 0
            print(f'New best model, save model to {self.save_dir}...')
            logging.info(f'New best model, save model to {self.save_dir}...')
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'AGN_weights.pth'))
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print(f'Epoch {epoch + 1}: Early stopping')
                logging.info(f'Epoch {epoch + 1}: Early stopping')

