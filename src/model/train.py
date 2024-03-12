import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:

    def __init__(self, 
            model: torch.nn.Module,
            device: torch.device,
            training_dataloader: DataLoader,
            validation_dataloader: DataLoader,
            learning_rate: float,
            num_epochs: int,
            validate_after_epochs: int
    ):
        self.device = device
        self.t_dl = training_dataloader
        self.v_dl = validation_dataloader
        self.model = model
        self.learning_rate = learning_rate
        
        self.loss_func = torch.nn.BCELoss()
        # Maybe add in weight decay? Parameter to look into...
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate) 

        self.max_epochs = num_epochs
        self.validate_after_epochs = validate_after_epochs
        self.epochs_elapsed = 0
        self.finished = False

        self.training_losses   = list()
        self.validation_losses = list()

    def train(self, n_epochs=0):
        if n_epochs == 0:
            n_epochs = self.max_epochs

        print(f"Training {type(self.model).__name__} for {n_epochs} epoch{'s' if n_epochs > 1 else ''}... ")
        # Set model to training mode
        self.model.train()
        while (self.epochs_elapsed < self.max_epochs) and (n_epochs > 0):
            running_loss = 0.0
            examples_count = 0
            progress = tqdm(self.t_dl, total=len(self.t_dl), ncols=120)
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

                # Calculate loss
                l_s = batch_inputs.size(0)
                running_loss += loss.item() * l_s

                # Update stats
                examples_count += l_s

                progress.set_description("[Trn] " + f"Epoch {self.epochs_elapsed + 1}, "
                                                  + f"Loss: {running_loss / examples_count:.4f}")

            self.training_losses.append((self.epochs_elapsed + 1, running_loss / examples_count))

            new_elapsed = self.epochs_elapsed + 1

            if (new_elapsed % self.validate_after_epochs) == 0:
                self.validation()

            n_epochs -= 1
            self.epochs_elapsed = new_elapsed

        if self.epochs_elapsed == self.max_epochs:
            self.finished = True

    def validation(self):
        self.model.eval()
        with torch.no_grad():
            running_loss = 0.0
            examples_count = 0
            progress = tqdm(self.v_dl, total=len(self.v_dl), ncols=120)
            progress.set_description(f"[Val] Epoch {self.epochs_elapsed + 1}, Loss: ... ")
            for i, (batch_inputs, batch_labels) in enumerate(progress, start=1):
                batch_inputs = batch_inputs.to(self.device)
                batch_labels = batch_labels.to(self.device)

                # Calculate predictions
                outputs = self.model(batch_inputs)
                outputs = outputs.squeeze(1)
                loss = self.loss_func(outputs, batch_labels.float())

                # Calculate loss
                l_s = batch_labels.size(0)
                running_loss += loss.item() * l_s

                # Update stats
                examples_count += l_s
                progress.set_description("[Val] " + f"Epoch {self.epochs_elapsed + 1}, "
                                                  + f"Loss: {running_loss / examples_count:.4f}")

            self.validation_losses.append((self.epochs_elapsed + 1, running_loss / examples_count))
        self.model.train()