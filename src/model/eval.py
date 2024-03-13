import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix
import numpy as np

from tqdm import tqdm

class Evaluator:
    def __init__(self,
            model: torch.nn.Module,
            device: torch.device,
            testing_dataloader: DataLoader,
            threshold = 0.5,
    ):
        self.model = model
        self.device = device
        self.tr_dl = testing_dataloader

        self.loss_func = torch.nn.BCELoss()
        self.threshold = threshold

        self.reset_stats()

    def reset_stats(self):
        self.loss_avg = 0.0   # Average loss across batches
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1 = 0.0
        self.confusion = np.array([])
        self.roc = None

    def print_stats(self):
        def print_stat(name, val):
            print(f"{name:<15}{val:.4f}")
        print_stat("Average Loss", self.loss_avg)
        print_stat("Accuracy",     self.accuracy)
        print_stat("Precision",    self.precision)
        print_stat("Recall",       self.recall)
        print_stat("F1 Score",     self.f1)
        print("Confusion matrix:")
        print(self.confusion)

    def eval(self):
        self.reset_stats()
        print(f"Evaluating {type(self.model).__name__}...")

        correct_preds = 0
        total_preds = 0
        
        progress = tqdm(self.tr_dl, total=len(self.tr_dl), ncols=120)
        progress.set_description(f"[Tst] Loss: ... , Accuracy: ... ")

        self.model.eval()
        with torch.no_grad():
            pred_labels = []
            pred_probs  = []
            true_labels = []
            for i, (batch_inputs, batch_labels) in enumerate(progress, start=1):
                batch_inputs = batch_inputs.to(self.device)
                batch_labels = batch_labels.to(self.device)

                # Calculate predictions
                outputs = self.model(batch_inputs).squeeze(1)
                preds = outputs > self.threshold
                preds = preds.float()

                # Calculate loss
                l_s = batch_labels.size(0)
                loss = self.loss_func(outputs, batch_labels.float())
                self.loss_avg += loss.item() * l_s

                # Update stats
                correct_preds += torch.eq(preds, batch_labels).sum().item()
                total_preds += l_s
                pred_labels.extend(preds.cpu().detach().numpy())
                pred_probs.extend(outputs.cpu().detach().numpy())
                true_labels.extend(batch_labels.cpu().detach().numpy())

                progress.set_description(f"[Tst] " + f"Loss: {self.loss_avg / i :.4f}, "
                                                   + f"Accuracy: {correct_preds / total_preds :.4f} ")
            
            self.accuracy  = accuracy_score(true_labels, pred_labels)
            self.precision = precision_score(true_labels, pred_labels)
            self.recall    = recall_score(true_labels, pred_labels)
            self.f1        = f1_score(true_labels, pred_labels)
            self.confusion = confusion_matrix(true_labels, pred_labels)
            self.roc       = roc_curve(true_labels, pred_probs)

        self.loss_avg /= len(self.tr_dl)
