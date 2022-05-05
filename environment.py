import torch

from visualiser import Visualiser
from collections import deque


class EnvironmentBox:
    def __init__(self, n_agents, n_environments=1):
        self.agent_locs = torch.zeros((n_environments, n_agents, 2))
        self.agent_vels = torch.zeros((n_environments, n_agents, 2))
        self.past_locs = deque([torch.zeros((n_environments, n_agents, 2))])
        self.past_capacity = 50
        self.n_agents = n_agents
        self.n_environments = n_environments
        self.headless = False
        self.visualised_idx = 0
        self.visualiser = Visualiser()

    def step(self, agent_forces, delta_t=0.05, limit_f=True):
        if limit_f:
            force_norm = torch.linalg.vector_norm(agent_forces, dim=-1).T
            agent_forces = agent_forces / force_norm
            agent_forces = agent_forces * torch.minimum(force_norm, torch.ones_like(force_norm)*20)
        self.agent_vels += agent_forces * delta_t
        self.agent_locs += self.agent_vels * delta_t
        self.past_locs.append(self.agent_locs.clone())
        if len(self.past_locs) > self.past_capacity:
            self.past_locs.popleft()
        if not self.headless:
            self.visualiser.render(
                positions=self.agent_locs[self.visualised_idx],
                past_positions=torch.stack([locs[self.visualised_idx] for locs in self.past_locs], dim=1)
            )

    def get_close_neighbours(self, separation_limit):
        edge_indices = [[] for _ in range(self.n_environments)]
        for i in range(self.n_agents):
            for j in range(i):
                pos_difs = self.agent_locs[:,i] - self.agent_locs[:, j]
                for idx_env in range(self.n_environments):
                    if torch.linalg.vector_norm(pos_difs[idx_env]) < separation_limit:
                        edge_indices[idx_env].append([i, j])
                        edge_indices[idx_env].append([j, i])
        return edge_indices
