import logging
import os
from collections import defaultdict
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from utils import move_to_device
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score


class NerMetrics:
    def __init__(self, model, eval_data_loader, device, save_dir, epochs, min_delta=1e-4, patience=100):
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

        self.label2id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7,
                         'I-MISC': 8}
        self.id2label = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC',
                         8: 'I-MISC'}

    # def calc_metrics(self):
    #     self.model.eval()
    #     y_pred, y_true = [], []
    #     with torch.no_grad():
    #         for data in self.eval_data_loader:
    #             data = move_to_device(data, self.device)
    #             inputs, true_labels = data, data["label_ids"]
    #             outputs = self.model(inputs)
    #             mask = inputs["attention_mask"]
    #
    #             preds = self.model.crf.decode(outputs, mask.bool())  # (seq_length, batch_size)
    #
    #             for i in range(len(true_labels)):
    #                 # Apply mask to remove -1 padded true labels and corresponding predictions
    #                 masked_true_labels = true_labels[i][true_labels[i] != -1].cpu().numpy()
    #                 y_pred.extend(preds[i])
    #                 y_true.extend(masked_true_labels)
    #
    #     # 计算并打印评估指标
    #     acc = accuracy_score(y_true, y_pred)
    #     macro_f1 = f1_score(y_true, y_pred, average="macro")
    #     micro_f1 = f1_score(y_true, y_pred, average="micro")
    #     # report = classification_report(y_true, y_pred, zero_division=0)
    #     # print(report)
    #     return acc, macro_f1, micro_f1

    def calc_metrics(self):
        self.model.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for data in self.eval_data_loader:
                data = move_to_device(data, self.device)
                inputs, true_labels = data, data["label_ids"]
                outputs = self.model(inputs)

                # Apply softmax to get probabilities and then argmax to get the predicted labels
                preds = torch.argmax(outputs, dim=-1)

                for i in range(len(true_labels)):
                    # Apply mask to remove -1 padded true labels and corresponding predictions
                    masked_true_labels = true_labels[i][true_labels[i] != -100].cpu().numpy().tolist()
                    masked_preds = preds[i][true_labels[i] != -100].cpu().numpy().tolist()
                    y_true.append([self.id2label[label] for label in masked_true_labels])
                    y_pred.append([self.id2label[label] for label in masked_preds])

        # 计算并打印评估指标
        acc = round(accuracy_score(y_true, y_pred), 4)
        precision = round(precision_score(y_true, y_pred), 4)
        recall = round(recall_score(y_true, y_pred), 4)
        f1 = round(f1_score(y_true, y_pred), 4)
        return acc, precision, recall, f1

    def on_epoch_end(self, epoch, avg_loss):
        val_acc, val_precision, val_recall, val_f1 = self.calc_metrics()

        # 记录评估指标到history中
        self.history['val_acc'].append(val_acc)
        self.history['val_precision'].append(val_precision)
        self.history['val_recall'].append(val_recall)
        self.history['val_f1'].append(val_f1)

        print(
            f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss}, val_acc: {val_acc}, val_precision: {val_precision}, val_recall: {val_recall}, val_f1: {val_f1}")
        logging.info(
            f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss}, val_acc: {val_acc}, val_precision: {val_precision}, val_recall: {val_recall}, val_f1: {val_f1}")

        # Early stop
        if round(val_f1, 4) > self.best + self.min_delta:
            self.best = val_f1
            self.wait = 0
            print(f'New best model, save model to {self.save_dir}...')
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'AGN_weights.pth'))
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print(f'Epoch {epoch + 1}: Early stopping')
                logging.info(f'Epoch {epoch + 1}: Early stopping')
