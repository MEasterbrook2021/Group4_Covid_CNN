import torch
from torch.utils.data import DataLoader

class Trainer:

    def __init__(self,
            device: torch.device,
            covidx_dataloader: DataLoader, 
            model: torch.nn.Module,
            learning_rate: int,
            num_epochs: int
    ):
        self.device = device
        self.dl = covidx_dataloader
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = num_epochs
        
        self.loss_func = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate) # Maybe add in weight decay? Parameter to look into...

        self.total_loss = 0.0

    def train(self):
        loss_func = torch.nn.BCELoss()

        for epoch in range(self.epochs):
            print("Epoch {}".format(epoch + 1))
            self.model.train()
            total_training_loss = 0

            for batch_inputs, batch_labels in self.dl:
                batch_inputs = batch_inputs.to(self.device)
                batch_labels = batch_labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_inputs)
                outputs = outputs.squeeze(1)
                loss = loss_func(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()

                total_training_loss += loss.item()

        return self.model