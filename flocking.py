import torch
from torch.nn import Linear, ReLU, BatchNorm1d, Module
from torch.nn import Sequential
from torch_scatter import scatter
from torch.special import expit
from torch_geometric.nn import MessagePassing, GraphConv


class ReynoldsFlockingLayer(MessagePassing):
    def __init__(self, min_dist=5, visualiser=None):
        super().__init__()
        self.node_dim = 0
        self.min_dist = min_dist
        self.visualiser = visualiser
        # self.max_force = max_force

    def forward(self, pos, vel, edge_index):
        acc_out = self.propagate(edge_index, pos=pos, vel=vel)
        return acc_out

    def message(self, pos_i, pos_j, vel_i, vel_j):
        pos_dif = pos_j - pos_i
        pos_dif_norm = pos_dif.T / torch.linalg.vector_norm(pos_dif, dim=1)
        coll = - expit(-10*(torch.linalg.vector_norm(pos_dif, dim=1) - self.min_dist)) * pos_dif_norm
        coll = torch.nan_to_num(coll.T, 0.0, 0.0, 0.0)
        vel_dif = vel_j - vel_i
        return pos_dif, vel_dif, coll

    def aggregate(self, inputs, index, pos_i):
        pos_dif, vel_dif, coll = inputs
        coll_out = scatter(coll, index, dim=self.node_dim, reduce='add')
        coh_out = scatter(pos_dif, index, dim=self.node_dim, reduce='mean')
        all_out = scatter(vel_dif, index, dim=self.node_dim, reduce='mean')
        pos_out = scatter(pos_i, index, dim=self.node_dim, reduce='mean')
        if self.visualiser is not None:
           self.visualiser.draw_forces(pos_out, coh_out/15, all_out)

        return coll_out, coh_out, all_out

    def update(self, aggr_out):
        coll_aggr, coh_aggr, all_aggr = aggr_out
        return coll_aggr * 5 + coh_aggr / 30 + all_aggr


class CompactInputReynoldsLayer(MessagePassing):
    def __init__(self, min_dist=5):
        super().__init__()
        self.node_dim = 0
        self.min_dist = min_dist
        # self.max_force = max_force

    def forward(self, h,  edge_index):
        acc_out = self.propagate(edge_index, h=h)
        return acc_out

    def message(self, h_i, h_j):
        h_dif = h_j - h_i
        pos_dif, vel_dif = torch.split(h_dif, 2, dim=1)
        pos_dif_norm = pos_dif.T / torch.linalg.vector_norm(pos_dif, dim=1)
        coll = - expit(-10*(torch.linalg.vector_norm(pos_dif, dim=1) - self.min_dist)) * pos_dif_norm
        coll = torch.nan_to_num(coll.T, 0.0, 0.0, 0.0)
        return torch.cat([pos_dif / 30 + vel_dif, coll], dim=1)

    def aggregate(self, inputs, index):
        mean_in, coll = torch.split(inputs, 2, dim=1)
        add_out = scatter(coll, index, dim=self.node_dim, reduce='add')
        mean_out = scatter(mean_in, index, dim=self.node_dim, reduce='mean')
        return torch.cat([add_out, mean_out], dim=1)

    def update(self, aggr_out, h):
        add_aggr, mean_aggr = torch.split(aggr_out, 2, dim=1)
        return add_aggr * 5 + mean_aggr


class ReynoldsFlockingModel(Module):
    def __init__(self, min_dist=5):
        super().__init__()

        self.layer = CompactInputReynoldsLayer(min_dist)

    def forward(self, data):
        h = torch.cat([data.pos, data.vel], dim=-1)

        acc_new = self.layer(h, data.edge_index)  # (n, d) -> (n, d)
        return acc_new


class MPNNFlockingLayer(MessagePassing):
    def __init__(self, emb_dim, nn_layers=2, aggr='add'):
        super().__init__(aggr=aggr)
        self.node_dim = 0
        self.emb_dim = emb_dim

        self.mlp_msg = Sequential(
            Linear(2*emb_dim, emb_dim), *([BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim)]*(nn_layers - 1)), BatchNorm1d(emb_dim), ReLU()
        )
        self.mlp_upd = Sequential(
            Linear(2 * emb_dim, emb_dim), *([BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim)]*(nn_layers - 1)), BatchNorm1d(emb_dim), ReLU()
        )

    def forward(self, h, edge_index):
        out_acc = self.propagate(edge_index, h=h)
        return out_acc

    def message(self, h_i, h_j):
        msg_h = torch.cat([h_i, h_j], dim=-1)
        msg = self.mlp_msg(msg_h)
        return msg

    def aggregate(self, msg_in, index):
        aggr_out = scatter(msg_in, index, dim=self.node_dim, reduce=self.aggr)
        return aggr_out

    def update(self, aggr_out, h):
        upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(upd_out)


class MPNNFlockingModel(Module):
    def __init__(self, nn_layers=2, emb_dim=4, out_dim=2):
        super().__init__()

        self.conv = MPNNFlockingLayer(emb_dim, nn_layers=nn_layers, aggr='add')

        self.lin_pred = Linear(emb_dim, out_dim)

    def forward(self, data):
        h = torch.cat([data.pos, data.vel], dim=-1)

        acc_new = self.conv(h, data.edge_index)  # (n, d) -> (n, d)

        out = self.lin_pred(acc_new)  # (batch_size, d) -> (batch_size, 1)

        return out


class GraphConvFlockingModel(Module):
    def __init__(self, emb_dim=64, in_dim=2, edge_dim=0, out_dim=2):
        super().__init__()

        # self.lin_in = Linear(in_dim*2, emb_dim)

        self.conv = GraphConv(emb_dim, emb_dim)

        self.lin_pred = Linear(emb_dim, out_dim)

    def forward(self, data):
        # h = self.lin_in(torch.cat([data.pos, data.vel], dim=-1))
        h = torch.cat([data.pos, data.vel], dim=-1)

        acc_new = self.conv(h, data.edge_index)  # (n, d) -> (n, d)

        out = self.lin_pred(acc_new)  # (batch_size, d) -> (batch_size, 1)

        return out


class BiasedMPNNFlockingLayer(MessagePassing):
    def __init__(self, emb_dim=64, aggr='add'):
        super().__init__(aggr=aggr)
        self.node_dim = 0
        self.emb_dim = emb_dim

        self.mlp_msg = Sequential(
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
        )
        self.mlp_upd = Sequential(
            Linear(2*emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
        )

    def forward(self, h, edge_index):
        out_acc = self.propagate(edge_index, h=h)
        return out_acc

    def message(self, h_i, h_j):
        h_dif = h_i - h_j
        msg = self.mlp_msg(h_dif)
        return msg

    def aggregate(self, msg_in, index):
        add_out = scatter(msg_in[:, :2], index, dim=self.node_dim, reduce='add')
        mean_out = scatter(msg_in[:, 2:], index, dim=self.node_dim, reduce='mean')
        aggr_out = torch.cat([add_out, mean_out], dim=1)
        return aggr_out

    def update(self, aggr_out, h):
        upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(upd_out)


class BiasedMPNNFlockingModel(Module):
    def __init__(self, emb_dim=4, out_dim=2):
        super().__init__()

        # self.lin_in = Linear(in_dim*2, emb_dim)

        self.conv = BiasedMPNNFlockingLayer(emb_dim)

        self.lin_pred = Linear(emb_dim, out_dim)

    def forward(self, data):
        h = torch.cat([data.pos, data.vel], dim=-1)

        acc_new = self.conv(h, data.edge_index)  # (n, d) -> (n, d)

        out = self.lin_pred(acc_new)  # (batch_size, d) -> (batch_size, 1)

        return out
