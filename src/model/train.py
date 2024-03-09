import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:

    def __init__(self, 
            model: torch.nn.Module,
            device: torch.device,
            training_dataloader: DataLoader,
            learning_rate: float,
            num_epochs: int
    ):
        self.device = device
        self.dl = training_dataloader
        self.model = model
        self.learning_rate = learning_rate
        self.max_epochs = num_epochs
        
        self.loss_func = torch.nn.BCELoss()
        # Maybe add in weight decay? Parameter to look into...
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate) 

        self.epochs_elapsed = 0
        self.epoch_losses = list()

    def train(self, n_epochs=1):
        print(f"Training {type(self.model).__name__} for {n_epochs} epoch{'s' if n_epochs > 1 else ''}... ")
        # Set model to training mode
        self.model.train()
        while (self.epochs_elapsed < self.max_epochs) and (n_epochs > 0):
            total_loss = 0.0
            progress = tqdm(self.dl, total=len(self.dl), ncols=120)
            progress.set_description(f"[Trn] Epoch {self.epochs_elapsed + 1}, Loss: ... ")
            for i, (batch_inputs, batch_labels) in enumerate(progress, start=1):
                batch_inputs = batch_inputs.to(self.device)
                batch_labels = batch_labels.to(self.device)

                # Zero out the optimizer gradients
                self.optimizer.zero_grad()
                outputs = self.model(batch_inputs)
                outputs = outputs.squeeze(1)
                loss = self.loss_func(outputs, batch_labels.float())
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                progress.set_description(f"[Trn] Epoch {self.epochs_elapsed + 1}, Loss: {total_loss/i:.4f}")

            self.epoch_losses.append(total_loss / len(progress))

            self.epochs_elapsed += 1
            n_epochs -= 1