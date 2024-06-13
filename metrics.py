import os

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict
import torch

from utils import move_to_device


class ClfMetrics:
    def __init__(self, model, eval_loader, device, save_path, epochs, min_delta=1e-4, patience=10):
        self.model = model
        self.eval_loader = eval_loader
        self.device = device
        self.save_path = save_path
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
                inputs, labels = data, data["label_id"]
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        return f1, acc

    def on_epoch_end(self, epoch, avg_loss, logs=None):
        val_f1, val_acc = self.calc_metrics()
        self.history['val_acc'].append(val_acc)
        self.history['val_f1'].append(val_f1)
        print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss}, Val_acc {val_acc}, Val_f1 {val_f1}")
        if self.monitor_op(val_f1 - self.min_delta, self.best):
            self.best = val_f1
            self.wait = 0
            print(f'New best model, saving model to {self.save_path}...')
            torch.save(self.model.state_dict(), os.path.join(self.save_path, 'AGN_weights.pth'))
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(f'Epoch {epoch + 1}: Early stopping')