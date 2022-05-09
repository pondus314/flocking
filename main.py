import random
import time

import torch
import torch_cluster

import data
import flocking
import trainer
import utils
from environment import EnvironmentBox
from flocking import ReynoldsFlockingLayer
from torch_geometric.data import DataLoader

if __name__ == '__main__':
    # data_writer = utils.DataWriter('.\\data', 'reynolds_3')
    n_envs = 4
    n_agents = 200
    env = EnvironmentBox(n_agents, n_envs, headless=False)
    # env.step(torch.stack([torch.ones((n_agents, 2))*400]*4, dim=0), delta_t=1, f_limit=0)
    # env.step(-torch.stack([torch.ones((n_agents, 2))*400]*4, dim=0), delta_t=1, f_limit=0)
    # forces = torch.abs(torch.rand((n_agents, 2)))
    # print(env.agent_locs)
    #
    # env.step(torch.stack([forces, forces * torch.tensor([1, -1]), -forces,   -forces * torch.tensor([1, -1])], dim=0), delta_t=9, f_limit=0)
    # env.step(torch.stack([-forces, -forces * torch.tensor([1, -1]),  forces,   forces * torch.tensor([1, -1])], dim=0), delta_t=5, f_limit=0)
    # edge_indices = env.get_close_neighbours(51)
    # flocking_layer = ReynoldsFlockingLayer(min_dist=15)
    #
    # for i in range(10000):
    #     edge_indices = env.get_close_neighbours(51)
    #     upd_forces = torch.stack([flocking_layer(env.agent_locs[j], env.agent_vels[j], edge_indices[j]) for j in range(n_envs)])
    #     if random.random() < 0.1:
    #         for j in range(n_envs):
    #             data_writer.write_data(edge_indices[j], env.agent_locs[j], env.agent_vels[j], upd_forces[j])
    #     env.step(upd_forces)

    # data_writer.finish_writing()
    dataset = data.FlockingDataset('.\\data', 'reynolds_3')
    mpnnModel = flocking.MPNNFlockingModel()
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    mpnn_trainer = trainer.ModelTrainer(mpnnModel, 100, 0.1, 0.5, loader, True)
    mpnn_trainer.train()
    print('done?')
