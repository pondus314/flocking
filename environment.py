import torch
import torch_cluster
import torch_geometric.utils

from visualiser import Visualiser
from collections import deque


class EnvironmentBox:
    def __init__(self, n_agents, n_environments=1, headless=False, keep_in_view=False):
        self.agent_locs = torch.zeros((n_environments, n_agents, 2))
        self.agent_vels = torch.zeros((n_environments, n_agents, 2))
        self.past_locs = deque([torch.zeros((n_environments, n_agents, 2))])
        self.past_capacity = 50
        self.n_agents = n_agents
        self.n_environments = n_environments
        self.headless = headless
        self.visualised_idx = 0
        if not self.headless:
            self.visualiser = Visualiser()
            self.visualiser.setup((1,1))
        self.keep_in_view = keep_in_view

    def step(self, agent_forces, delta_t=0.3, f_limit=0.01, v_limit=1):
        if f_limit:
            force_norm = torch.linalg.vector_norm(agent_forces, dim=-1).T
            force_norm = force_norm.unsqueeze(0).transpose(0, 2).expand(self.n_environments, self.n_agents, 2)
            agent_forces = agent_forces / force_norm
            agent_forces = agent_forces * torch.minimum(force_norm, torch.ones_like(force_norm)*f_limit)
            agent_forces = torch.nan_to_num(agent_forces, 0.0, 0.0, 0.0)
        self.agent_vels += agent_forces * delta_t
        if v_limit:
            speed_norm = torch.linalg.vector_norm(self.agent_vels, dim=-1).T
            speed_norm = speed_norm.unsqueeze(0).transpose(0, 2).expand(self.n_environments, self.n_agents, 2)
            self.agent_vels = self.agent_vels / speed_norm
            self.agent_vels = self.agent_vels * torch.minimum(speed_norm, torch.ones_like(speed_norm)*v_limit)
            self.agent_vels = torch.nan_to_num(self.agent_vels, 0.0, 0.0, 0.0)
        self.agent_locs += self.agent_vels * delta_t
        self.past_locs.append(self.agent_locs.clone())
        if len(self.past_locs) > self.past_capacity:
            self.past_locs.popleft()
        if self.keep_in_view:
            self.agent_locs = torch.stack(
                [
                    self.agent_locs[:, :, 0] % self.visualiser.HEIGHT,
                    self.agent_locs[:, :, 1] % self.visualiser.WIDTH
                ],
                dim=2
            )
        if not self.headless:
            self.visualiser.render(
                positions=self.agent_locs[self.visualised_idx],
                velocities=self.agent_vels[self.visualised_idx],
                past_positions=torch.stack([locs[self.visualised_idx] for locs in self.past_locs], dim=1)
            )
            # pass


    def get_close_neighbours(self, separation_limit, remove_self_loops=False):
        g = [torch_cluster.radius_graph(self.agent_locs[i], separation_limit, loop=True,) for i in range(self.n_environments)]
        if remove_self_loops:
            for i, edge_index in enumerate(g):
                g[i] = torch_geometric.utils.remove_self_loops(edge_index)[0]
        return g
