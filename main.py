import random
import time

import torch
import torch_cluster

import data
import flocking
import utils
from environment import EnvironmentBox
from flocking import ReynoldsFlockingLayer

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
    datas = utils.read_flocking_data('.\\data', 'reynolds_1')
    dataset = data.FlockingDataset('.\\data', 'reynolds_1')
    print(dataset[0])
    mpnnModel = flocking.MPNNFlockingModel()
    print('done?')
