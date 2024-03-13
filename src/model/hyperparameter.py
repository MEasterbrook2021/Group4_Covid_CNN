import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

class TuneHyperparameters:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 val_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 activation: torch.sigmoid,
                 params_to_tune: dict,
                 threshold: float,
                ):
            
        self.model = model
        self.device = device
        self.val_dl = val_dataloader
        self.test_dl = test_dataloader
        self.params_to_tune = params_to_tune
        self.threshold = threshold
        self.loss_func = torch.nn.BCELoss()
        self.activation = activation
        self.threshold = threshold

        self.epochs_elapsed = 0
        self.finished = False
        self.training_losses = list()

        self.best_model = None
        self.best_accuracy = 0.0
        self.best_params = None

        self.reset_stats()

    def reset_stats(self):
        self.loss_avg = 0.0
        self.loss_total = 0.0
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1 = 0.0

    def fit(self):
        n_epochs = self.max_epochs
        self.model.to(self.device)
        self.model.train()

        while((self.epochs_elapsed < self.max_epochs) and (n_epochs > 0)):
        # for epoch in range(self.model.max_epochs):
            self.model.train()

            running_loss = 0.0
            examples_count = 0.0

            progress = tqdm(self.test_dl, total=len(self.test_dl), ncols=120)
            progress.set_description(f"[Evl] Epoch {self.epochs_elapsed+1}, Loss: ... ")

            for i, (batch_inputs, batch_labels) in enumerate(progress, start=1):
                batch_inputs = batch_inputs.to(self.device)
                batch_labels = batch_labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(batch_inputs)
                outputs = outputs.squeeze(1)
                loss = self.loss_func(outputs, batch_labels.float())
                loss.backward()
                self.optimizer.step()

                # Calculate loss
                l_s = batch_inputs.size(0)
                running_loss += loss.item() * l_s

                progress.set_description("[Evl] " + f"Epoch {self.epochs_elapsed + 1}", 
                                                    + f"Loss: {running_loss/examples_count:.4f}")
                
            self.training_losses.append((self.epochs_elapsed + 1, running_loss/examples_count))

            new_elapsed = self.epochs_elapsed + 1
            n_epochs -= 1
            self.epochs_elapsed = new_elapsed

            if self.epochs_elapsed == self.max_epochs:
                self.finished = True


    def tune(self):
        print(f"Tuning {type(self.model).__name__}...")

        for rate in self.params_to_tune["learning_rate"]:
            print("\n Trying {} as learning rate".format(rate))
            self.learning_rate = rate
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate) 

            self.model.learning_rate = rate
            
            for epochs in self.params_to_tune["epochs"]:
                print("\n Trying {} number of epochs".format(epochs))
                self.max_epochs = epochs
                self.model.max_epochs = epochs

                for threshold in self.params_to_tune["thresholds"]:
                    print("\n Trying {} as threshold".format(threshold))
                    self.threshold = threshold # Changed this from self.model.threshold to self.threshold
                    self.model.fit()
                    # model = self.model
                    accuracy = self.evaluate() # Might need to pass in model as a param to evaluate.

                    if accuracy > self.best_accuracy:
                        self.best_accuracy = accuracy
                        self.best_model = self.model
                        self.best_params = {'learning rate' : self.learning_rate, 'nb_epochs' : self.max_epochs, 'threshold' : self.threshold}
                        
        print(f"Best validation accuracy: {self.best_accuracy}")
        print(f"Best parameters: {self.best_params}")

    def evaluate(self):

        print(f"Evaluating {type(self.model).__name__}...")

        correct_preds = 0
        total_preds = 0
        
        self.model.eval()
        progress = tqdm(self.val_dl, total=len(self.val_dl), ncols=120)
        progress.set_description(f"[Evl] Loss: ... , Accuracy: ... ")

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

        return self.accuracy

       
    

                    




    