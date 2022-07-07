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


def get_full_model_name(model_name, model):
    return f"{model_name}E{model.emb_dim}L{model.nn_layers}O{model.layer_out}"


if __name__ == '__main__':
    run_training = False
    run_regression = False
    run_simulation = True
    run_testing = False

    ds_name = 'reynolds_20env'
    # models = {
        # '1aggrnolin': flocking.OneAggrMPNNFlockingModel(emb_dim=128),
        # '1aggrlinin': flocking.OneAggrMPNNFlockingModel(emb_dim=128, use_lin_in=True),
        # '2aggrnolin': flocking.BiasedMPNNFlockingModel(emb_dim=128, layer_out=128),
        # '2aggrlinin': flocking.FullEmbMPNNFlockingModel(emb_dim=128)
        # 'no_bias': flocking.MPNNFlockingModel(emb_dim=128, layer_out_dim=128),
        # 'msg_bias': flocking.MsgBiasedMPNNFlockingModel(emb_dim=128, layer_out=128),
        # 'upd_bias': flocking.UpdBiasedMPNNFlockingModel(emb_dim=128, layer_out=128),
        # 'full_bias': flocking.BiasedMPNNFlockingModel(emb_dim=128, layer_out=128),
    # }
    models = sum([[
            ('no_bias', flocking.MPNNFlockingModel(emb_dim=128, layer_out_dim=out_dim, nn_layers=4)),
            # ('msg_bias', flocking.MsgBiasedMPNNFlockingModel(emb_dim=128, layer_out=out_dim)),
            # ('upd_bias', flocking.UpdBiasedMPNNFlockingModel(emb_dim=128, layer_out=out_dim)),
            # ('full_bias', flocking.BiasedMPNNFlockingModel(emb_dim=128, layer_out=out_dim)),
        ] for out_dim in [4]]
        , [])

    models = [('reynolds', flocking.InterpretedFlockingModel())]

    dataset = data.FlockingDataset('.\\data', ds_name)
    ds_size = len(dataset)
    train_set, val_set = torch.utils.data.random_split(dataset, [(ds_size // 5) * 4, ds_size - (ds_size // 5) * 4])
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)

    if run_training:
        for model_name, model in models:
            if model_name == 'reynolds':
                continue
            mpnn_trainer = training.ModelTrainer(model, 60, 0.001, 0.005, train_loader)
            mpnn_trainer.train()
            torch.save(
                model.state_dict(),
                f'models/{model_name}E{model.emb_dim}L{model.nn_layers}O{model.layer_out}.pt'
            )
            print('training done')

    if run_testing:
        for model_name, model in models:
            if model_name == 'reynolds':
                continue
            model.load_state_dict(
                torch.load(f'models/{model_name}E{model.emb_dim}L{model.nn_layers}O{model.layer_out}.pt')
            )
            model.to("cuda")
            val_loader = DataLoader(val_set, batch_size=16, shuffle=True)
            total_loss = 0.0
            loss_fn = torch.nn.MSELoss()
            with torch.no_grad():
                for data in val_loader:
                    data.to("cuda")
                    pred = model(data)
                    loss = loss_fn(pred, data.y)
                    total_loss += loss
            print(model_name, model.layer_out, total_loss/len(val_loader))

    to_skip = 0
    if run_regression:
        for model_name, model in models:
            if to_skip:
                to_skip -= 1
                continue
            name = get_full_model_name(model_name, model)
            model.load_state_dict(
                torch.load(f'models/{model_name}E{model.emb_dim}L{model.nn_layers}O{model.layer_out}.pt')
            )
            model.to("cuda")
            upd_model = pysr.PySRRegressor(
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["square"],
                equation_file=f"sr_out/{name}_upd1",
                niterations=100,
            )
            msg_model = pysr.PySRRegressor(
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["square"],
                equation_file=f"sr_out/{name}_msg1",
                niterations=100,
            )
            lin_pred_model = pysr.PySRRegressor(
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["square"],
                equation_file=f"sr_out/{name}_pred1",
                niterations=100,
            )
            regression_model = RegressionModel(msg_model, upd_model, lin_pred_regression_model=lin_pred_model, full_model=model)
            with torch.no_grad():
                for batch, data in enumerate(train_loader):
                    data.to("cuda")
                    regression_model(data)
                    break

    if run_simulation:
        for model_name, model in models:
            n_envs = 1
            n_agents = 50
            env = EnvironmentBox(n_agents, n_envs, headless=False, keep_in_view=True)
            env.step(torch.stack([torch.ones((n_agents, 2)) * 50], dim=0), delta_t=1, f_limit=0, v_limit=0)
            env.step(-torch.stack([torch.ones((n_agents, 2)) * 50], dim=0), delta_t=1, f_limit=0, v_limit=0)
            forces = torch.abs(torch.rand((n_agents, 2)))

            env.step(torch.stack([forces], dim=0), delta_t=9, f_limit=0, v_limit=0)
            env.step(torch.stack([-forces], dim=0), delta_t=5, f_limit=0, v_limit=0)
            if model_name != 'reynolds':
                model.load_state_dict(
                    torch.load(f'models/{model_name}E{model.emb_dim}L{model.nn_layers}O{model.layer_out}.pt')
                )

            for i in range(100000):
                edge_indices = env.get_close_neighbours(31, remove_self_loops=False)
                datas_out = []
                for j in range(n_envs):
                    data = Data(pos=env.agent_locs[j], vel=env.agent_vels[j], edge_index=edge_indices[j])
                    datas_out.append(model(data))
                upd_forces = torch.stack(datas_out)
                env.step(upd_forces)
