import torch
from torch_scatter import scatter
from torch.special import expit
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class ReynoldsFlockingLayer(MessagePassing):
    def __init__(self, min_dist=5):
        super().__init__()
        self.node_dim = 0
        self.min_dist = min_dist
        # self.max_force = max_force

    def forward(self, pos, vel, edge_index):
        acc_out = self.propagate(edge_index, pos=pos, vel=vel)
        return acc_out

    def message(self, pos_i, pos_j, vel_i, vel_j):
        pos_dif = pos_j - pos_i
        pos_dif_norm = pos_dif.T / torch.linalg.vector_norm(pos_dif, dim=1)
        coll = - expit(-10*(torch.linalg.vector_norm(pos_dif, dim=1) - self.min_dist)) * pos_dif_norm
        vel_dif = vel_j - vel_i
        return pos_dif, vel_dif, coll.T

    def aggregate(self, inputs, index):
        pos_dif, vel_dif, coll = inputs
        coll_out = scatter(coll, index, dim=self.node_dim, reduce='add')
        coh_out = scatter(pos_dif, index, dim=self.node_dim, reduce='mean')
        all_out = scatter(vel_dif, index, dim=self.node_dim, reduce='mean')
        return coll_out, coh_out, all_out

    def update(self, aggr_out):
        coll_aggr, coh_aggr, all_aggr = aggr_out
        return coll_aggr * 25 + coh_aggr + all_aggr
