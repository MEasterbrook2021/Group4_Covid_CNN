import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

class Evaluator:
    def __init__(self,
            model: torch.nn.Module,
            device: torch.device,
            testing_dataloader: DataLoader,
            activation = torch.sigmoid,
            threshold = 0.5,
    ):
        self.model = model
        self.device = device
        self.dl = testing_dataloader

        self.loss_func = torch.nn.BCELoss()
        self.activation = activation
        self.threshold = threshold

        self.reset_stats()

    def reset_stats(self):
        self.loss_avg = 0.0   # Average loss across batches
        self.loss_total = 0.0 # Total loss through one run of the dataset
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1 = 0.0

    def eval(self):
        self.reset_stats()
        print(f"Evaluating {type(self.model).__name__}...")

        correct_preds = 0
        total_preds = 0
        
        self.model.eval()
        progress = tqdm(self.dl, total=len(self.dl), ncols=120)
        progress.set_description(f"[Tst] Loss: ... , Accuracy: ... ")

        with torch.no_grad():
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
                self.loss_total += loss.item() * l_s

                # Update stats
                correct_preds += torch.eq(preds, batch_labels).sum().item()
                total_preds += l_s

                progress.set_description(f"[Tst] " + f"Loss: {self.loss_total / total_preds :.4f}, "
                                                   + f"Accuracy: {correct_preds / total_preds :.4f} ")
        
        self.accuracy = correct_preds / total_preds
        self.loss_avg /= total_preds
