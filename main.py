import time

import torch

from environment import EnvironmentBox
from flocking import ReynoldsFlockingLayer

if __name__ == '__main__':
    n_envs = 1
    n_agents = 50
    env = EnvironmentBox(n_agents, n_envs)
    env.step(torch.ones((1, n_agents, 2))*200, delta_t=1, limit_f=False)
    env.step(-torch.ones((1, n_agents, 2))*200, delta_t=1, limit_f=False)
    forces = torch.abs(torch.rand((n_agents, 2)))
    print(env.agent_locs)
    env.step(torch.stack([forces]), delta_t=15, limit_f=False)
    env.step(torch.stack([-forces]), delta_t=5, limit_f=False)
    edge_indices = [torch.Tensor(edge_idx).to(torch.long) for edge_idx in env.get_close_neighbours(51)]
    flocking_layer = ReynoldsFlockingLayer(min_dist=15)

    for i in range(20000):
        edge_indices = [torch.Tensor(edge_idx).to(torch.long) for edge_idx in env.get_close_neighbours(51)]
        upd_forces = torch.stack([flocking_layer(env.agent_locs[i], env.agent_vels[i], edge_indices[i].T) for i in range(n_envs)])
        env.step(upd_forces)
