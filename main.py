import random
import time

import torch
import torch_cluster
import pysr

import optuna
from torch_geometric.data import Data

import data
import flocking
import training
import utils
from environment import EnvironmentBox
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
    # data_writer = utils.DataWriter('.\\data\\reynolds_over100_16env\\raw', 'reynolds_over100_16env')
    n_envs = 4
    n_agents = 50
    env = EnvironmentBox(n_agents, n_envs, headless=False, keep_in_view=True)
    env.step(torch.stack([torch.ones((n_agents, 2))*400]*n_envs, dim=0), delta_t=1, f_limit=0)
    env.step(-torch.stack([torch.ones((n_agents, 2))*400]*n_envs, dim=0), delta_t=1, f_limit=0)
    forces =  torch.abs(torch.rand((n_agents, 2)))

    env.step(torch.stack([forces, forces * torch.tensor([1, -1]), -forces,   -forces * torch.tensor([1, -1])] * (n_envs//4), dim=0), delta_t=9, f_limit=0)
    env.step(torch.stack([-forces, -forces * torch.tensor([1, -1]),  forces,   forces * torch.tensor([1, -1])] * (n_envs//4), dim=0), delta_t=5, f_limit=0)

    # env.step(torch.stack([forces], dim=0), delta_t=9, f_limit=0, v_limit=0)
    # env.step(torch.stack([-forces], dim=0), delta_t=5, f_limit=0, v_limit=0)
    flocking_layer = flocking.CompactInputReynoldsLayer(min_dist=15)
    # flocking_model = flocking.ReynoldsFlockingModel(min_dist=15)

    mpnnModel = flocking.BiasedMPNNFlockingModel(emb_dim=128, nn_layers=4)
    mpnnModel.load_state_dict(torch.load('biasednolininE128L4O4.pt'))

    for i in range(5000):
        edge_indices = env.get_close_neighbours(31, remove_self_loops=False)
        datas_out = []
        for j in range(n_envs):
            data = Data(pos=env.agent_locs[j], vel=env.agent_vels[j], edge_index=edge_indices[j])
            datas_out.append(mpnnModel(data))
        upd_forces = torch.stack(datas_out)
        env.step(upd_forces)

    # for i in range(5000):
    #     edge_indices = env.get_close_neighbours(51)
    #     upd_forces = torch.stack([flocking_layer(torch.cat([env.agent_locs[j], env.agent_vels[j]], dim=1), edge_indices[j]) for j in range(n_envs)])
    #     if i > 100 and (random.random() < 0.1 or i < 1000):
    #         for j in range(n_envs):
    #             data_writer.write_data(edge_indices[j], env.agent_locs[j], env.agent_vels[j], upd_forces[j])
    #     env.step(upd_forces)
    # data_writer.finish_writing()

    dataset = data.FlockingDataset('.\\data', 'reynolds_20env')
    ds_size = len(dataset)
    train_set, val_set = torch.utils.data.random_split(dataset, [(ds_size//5)*4, ds_size - (ds_size//5)*4])
    # # #
    mpnnModel = flocking.BiasedMPNNFlockingModel(emb_dim=128, nn_layers=4, layer_out=4)
    loader = DataLoader(train_set, batch_size=16, shuffle=True)
    mpnn_trainer = training.ModelTrainer(mpnnModel, 60, 0.001, 0.005, loader, True)
    mpnn_trainer.train()
    torch.save(mpnnModel.state_dict(), 'biasednolininE128L4O4.pt')
    print('done?')

    # study = optuna.create_study()
    # study.optimize(objective_maker(train_set, val_set), n_trials=20)
    #
    # best_params = study.best_params
    # print(best_params)

    mpnnModel.load_state_dict(torch.load('splitflockingmeanaddE128L4.pt'))
    upd_model = pysr.PySRRegressor(
        binary_operators=["+", "-", "*", "/"],
        equation_file="flocking_upd",
    )
    msg_model = pysr.PySRRegressor(
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square"],
        niterations=200,
        equation_file="flocking_msg"
    )
    upd_model_2 = pysr.PySRRegressor(
        binary_operators=["+", "-", "*", "/"],
        niterations=100,
        equation_file="mpnn_upd"
    )
    msg_model_2 = pysr.PySRRegressor(
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square"],
        niterations=100,
        equation_file="mpnn_msg"
    )
    # lin_in_model = pysr.PySRRegressor(
    #     binary_operators=["+", "-", "*", "/"],
    #     unary_operators=["square"],
    #     niterations=100,
    #     equation_file="lin_in"
    # )
    lin_out_model = pysr.PySRRegressor(
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square"],
        niterations=100,
        equation_file="lin_out"
    )
    # regression_model = RegressionModel(msg_model, upd_model, mpnnModel.conv.msg_model, mpnnModel.conv.upd_model)
    flocking_layer = flocking.CompactInputReynoldsLayer(min_dist=15)
    train_loader = DataLoader(train_set, batch_size=64)
    flocking_layer.to("cuda")
    mpnnModel.to("cuda")

    regression_model = RegressionModel(msg_model, upd_model, full_model=flocking_layer)
    mpnn_regerssion_model = RegressionModel(msg_model_2, upd_model_2, full_model=mpnnModel.conv)

    with torch.no_grad():
        for batch, data in enumerate(train_loader):
            data.to("cuda")
            regression_model(data)

            h_in = torch.cat([data.pos, data.vel], dim=-1)
            # mpnn_in_data = mpnnModel.lin_in(h_in)
            # lin_in_model.fit(h_in.cpu()[:8000, :], mpnn_in_data.cpu()[:8000, :])

            # mpnn_layer_data = mpnnModel.conv(h_in, data.edge_index)
            mpnn_layer_data = mpnn_regerssion_model(data)
            mpnn_out_data = mpnnModel.lin_pred(mpnn_layer_data)

            lin_out_model.fit(mpnn_layer_data[:8000, :].cpu(), mpnn_out_data[:8000, :].cpu())
            break

    print("done?2")
