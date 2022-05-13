import random
import time

import torch
import torch_cluster
import pysr

import optuna
import data
import flocking
import training
import utils
from environment import EnvironmentBox
from flocking import ReynoldsFlockingLayer
from torch_geometric.loader import DataLoader

from regression import RegressionLayer, RegressionModel


def objective_maker(train_set, val_set):

    def objective(trial):
        lr = trial.suggest_loguniform('lr', 0.0001, 0.1)
        lr_mult = trial.suggest_uniform('lr_mult', 1, 10)
        max_lr = lr_mult * lr
        epochs = trial.suggest_int('epochs', 32, 128, log=True)
        # emb_dim = trial.suggest_int('emb_dim', 32, 128, log=True)

        train_loader = DataLoader(train_set, batch_size=32)
        val_loader = DataLoader(val_set)
        model = flocking.MPNNFlockingModel()
        trainer = training.ModelTrainer(model, epochs, lr, max_lr, train_loader)
        trainer.train()
        total_loss = 0.0
        loss_fn = torch.nn.MSELoss()
        with torch.no_grad():
            for data in val_loader:
                data.to("cuda")
                pred = model(data)
                loss = loss_fn(pred, data.y)
                total_loss += loss

        return total_loss / len(val_loader)

    return objective


if __name__ == '__main__':
    # data_writer = utils.DataWriter('.\\data\\reynolds_10k2\\raw', 'reynolds_10k2')
    # n_envs = 4
    # n_agents = 200
    # env = EnvironmentBox(n_agents, n_envs, headless=True)
    # env.step(torch.stack([torch.ones((n_agents, 2))*400]*n_envs, dim=0), delta_t=1, f_limit=0)
    # env.step(-torch.stack([torch.ones((n_agents, 2))*400]*n_envs, dim=0), delta_t=1, f_limit=0)
    # forces = torch.abs(torch.rand((n_agents, 2)))
    #
    # env.step(torch.stack([forces, forces * torch.tensor([1, -1]), -forces,   -forces * torch.tensor([1, -1])], dim=0), delta_t=9, f_limit=0)
    # env.step(torch.stack([-forces, -forces * torch.tensor([1, -1]),  forces,   forces * torch.tensor([1, -1])], dim=0), delta_t=5, f_limit=0)
    #
    # # env.step(torch.stack([forces], dim=0), delta_t=9, f_limit=0)
    # # env.step(torch.stack([-forces], dim=0), delta_t=5, f_limit=0)
    # edge_indices = env.get_close_neighbours(51)
    flocking_layer = ReynoldsFlockingLayer(min_dist=15)
    flocking_model = flocking.ReynoldsFlockingModel(min_dist=15)
    #
    # for i in range(25000):
    #     edge_indices = env.get_close_neighbours(51)
    #     upd_forces = torch.stack([flocking_layer(env.agent_locs[j], env.agent_vels[j], edge_indices[j]) for j in range(n_envs)])
    #     if random.random() < 0.1:
    #         for j in range(n_envs):
    #             data_writer.write_data(edge_indices[j], env.agent_locs[j], env.agent_vels[j], upd_forces[j])
    #     env.step(upd_forces)
    # data_writer.finish_writing()

    dataset = data.FlockingDataset('.\\data', 'reynolds_10k2')
    ds_size = len(dataset)
    train_set, val_set = torch.utils.data.random_split(dataset, [(ds_size//5)*4, ds_size - (ds_size//5)*4])

    # mpnnModel = flocking.MPNNFlockingModel(emb_dim=4, nn_layers=10)
    # loader = DataLoader(train_set, batch_size=32, shuffle=True)
    # mpnn_trainer = training.ModelTrainer(mpnnModel, 100, 0.0001, 0.0005, loader, True)
    # mpnn_trainer.train()
    # torch.save(mpnnModel.state_dict(), 'mpnnflockingE4L10.pt')
    # print('done?')

    # study = optuna.create_study()
    # study.optimize(objective_maker(train_set, val_set), n_trials=20)
    #
    # best_params = study.best_params
    # print(best_params)
    #

    # mpnnModel.load_state_dict(torch.load('mpnnflockingE4L10.pt'))
    upd_model = pysr.PySRRegressor(binary_operators=["+", "-", "*", "/"], unary_operators=["exp", "inv(x)=1/x"])
    msg_model = pysr.PySRRegressor(binary_operators=["+", "-", "*", "/"], unary_operators=["exp", "inv(x)=1/x"])
    # regression_model = RegressionModel(msg_model, upd_model, mpnnModel.conv.msg_model, mpnnModel.conv.upd_model)
    train_loader = DataLoader(train_set, batch_size=64)

    regression_model = RegressionModel(msg_model, upd_model, full_model=flocking_model.layer)

    with torch.no_grad():
        for batch, data in enumerate(train_loader):
            regression_model.forward(data)
            break

    print("done?2")
