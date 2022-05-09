import torch
from torch import nn
from torch_geometric.data import DataLoader


class ModelTrainer:
    def __init__(self,
                 model: torch.nn.Module,
                 epochs: int,
                 lr: float,
                 max_lr: float,
                 dataloader: DataLoader,
                 show_losses: bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.loss_fn = nn.MSELoss()
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.epochs = epochs
        self.dataloader = dataloader
        self.show_losses = show_losses
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=len(self.dataloader)
        )

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            print(epoch + 1, "/", self.epochs)
            for batch, data in enumerate(self.dataloader):
                data = data.to(self.device)
                pred = self.model(data)
                loss = self.loss_fn(pred, data.y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                if self.show_losses and batch % 50 == 0:
                    print(loss.item())
