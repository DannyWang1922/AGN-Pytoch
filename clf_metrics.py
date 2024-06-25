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
        f1 = f1_score(y_true, y_pred, average='macro')
        return acc, f1

    def on_epoch_end(self, epoch, avg_loss):
        val_acc, val_f1 = self.calc_metrics()
        self.history['val_acc'].append(val_acc)
        self.history['val_f1'].append(val_f1)
        print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss}, Val_acc {val_acc}, Val_f1 {val_f1}")
        if self.monitor_op(val_f1 - self.min_delta, self.best):
            self.best = val_f1
            self.wait = 0
            print(f'New best model, saving model to {self.save_dir}...')
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'AGN_clf_weights.pth'))
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(f'Epoch {epoch + 1}: Early stopping')








