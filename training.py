import math

import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.linalg import vector_norm
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
                # tan_loss = self.loss_fn(pred[:, 0]/pred[:, 1], data.y[:, 0]/data.y[:, 1])
                # loss += tan_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                if self.show_losses and batch % 50 == 0:
                    print(loss.item())


# def train_k_fold(dataset, num_repetitions=3, k=10):
#
#     for i in range(num_repetitions):
#         # Test acc. over all folds.
#         test_accuracies = []
#         train_accuracies = []
#         kf = KFold(n_splits=k, shuffle=True)
#
#         for train_index, test_index in kf.split(dataset):
#             # Determine hyperparameters
#             train_index, val_index = train_test_split(train_index, test_size=0.1)
#             best_val_acc = 0.0
#             best_gram_matrix = all_matrices[0]
#             best_c = C[0]
#
#             for gram_matrix in all_matrices:
#                 train = gram_matrix[train_index, :]
#                 train = train[:, train_index]
#                 val = gram_matrix[val_index, :]
#                 val = val[:, train_index]
#
#                 c_train = classes[train_index]
#                 c_val = classes[val_index]
#
#                 for c in C:
#                     clf = SVC(C=c, kernel="precomputed", tol=0.001)
#                     clf.fit(train, c_train)
#                     val_acc = accuracy_score(c_val, clf.predict(val)) * 100.0
#
#                     if val_acc > best_val_acc:
#                         best_val_acc = val_acc
#                         best_c = c
#                         best_gram_matrix = gram_matrix
#
#             # Determine test accuracy.
#             train = best_gram_matrix[train_index, :]
#             train = train[:, train_index]
#             test = best_gram_matrix[test_index, :]
#             test = test[:, train_index]
#
#             c_train = classes[train_index]
#             c_test = classes[test_index]
#             clf = SVC(C=best_c, kernel="precomputed", tol=0.001)
#             clf.fit(train, c_train)
#
#             best_train = accuracy_score(c_train, clf.predict(train)) * 100.0
#             best_test = accuracy_score(c_test, clf.predict(test)) * 100.0
#
#             test_accuracies.append(best_test)
#             train_accuracies.append(best_train)
#
#         test_accuracies_all.append(float(np.array(test_accuracies).mean()))
#         train_accuracies_all.append(float(np.array(train_accuracies).mean()))
