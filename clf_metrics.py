import logging
import os

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict
import torch
from sklearn.preprocessing import MultiLabelBinarizer

from utils import move_to_device
from sklearn.metrics import classification_report


class ClfMetrics:
    def __init__(self, model, eval_loader, device, save_dir, epochs, min_delta=1e-4, patience=10):
        self.model = model
        self.eval_loader = eval_loader
        self.device = device
        self.save_dir = save_dir
        self.patience = patience
        self.min_delta = min_delta
        self.monitor_op = np.greater
        self.best = -np.Inf
        self.wait = 0
        self.history = defaultdict(list)
        self.epochs = epochs

    def calc_metrics(self):
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for data in self.eval_loader:
                data = move_to_device(data, self.device)
                inputs, labels = data, data["label_ids"]
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        micro_f1 = f1_score(y_true, y_pred, average="micro")
        # report = classification_report(y_true, y_pred, zero_division=0)
        return acc, macro_f1, micro_f1

    def on_epoch_end(self, epoch, avg_loss):
        val_acc, val_macro_f1, val_micro_f1 = self.calc_metrics()
        self.history['val_acc'].append(val_acc)
        self.history['val_macro_f1'].append(val_macro_f1)
        self.history['val_micro_f1'].append(val_micro_f1)

        print(
            f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}, val_acc: {val_acc:.4f}, val_macro_f1: {val_macro_f1:.4f}, val_micro_f1: {val_micro_f1:.4f}")
        logging.info(
            f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}, val_acc: {val_acc:.4f}, val_macro_f1: {val_macro_f1:.4f}, val_micro_f1: {val_micro_f1:.4f}")

        if self.monitor_op(val_macro_f1 - self.min_delta, self.best):
            self.best = val_macro_f1
            self.wait = 0
            print(f'New best model, saving model to {self.save_dir}...')
            logging.info(f'New best model, save model to {self.save_dir}...')
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'AGN_weights.pth'))
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(f'Epoch {epoch + 1}: Early stopping')








