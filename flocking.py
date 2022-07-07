import torch
from torch.nn import Linear, Tanh, BatchNorm1d, Module
from torch.nn import Sequential
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing


class CompactInputReynoldsLayer(MessagePassing):
    def __init__(self, min_dist=5):
        super().__init__(aggr='add')
        self.node_dim = 0
        self.min_dist = min_dist
        # self.aggr = 'sum'
        # self.max_force = max_force

    def forward(self, h,  edge_index):
        acc_out = self.propagate(edge_index, h=h)
        return acc_out

    def message(self, h_i, h_j):
        h_dif = h_j - h_i
        pos_dif, vel_dif = torch.split(h_dif, 2, dim=1)
        coll = - pos_dif.T / (torch.linalg.vector_norm(pos_dif, dim=1) ** 2)
        coll = torch.nan_to_num(coll.T, 0.0, 0.0, 0.0)
        out = torch.cat([pos_dif / 20 + vel_dif, coll * 5], dim=1)
        return out

    def aggregate(self, inputs, index):
        mean_in, coll = torch.split(inputs, 2, dim=1)
        add_out = scatter(coll, index, dim=self.node_dim, reduce='add')
        mean_out = scatter(mean_in, index, dim=self.node_dim, reduce='mean')
        return torch.cat([add_out, mean_out], dim=1)

    def update(self, aggr_out, h):
        add_aggr, mean_aggr = torch.split(aggr_out, 2, dim=1)
        return add_aggr + mean_aggr


class ReynoldsFlockingModel(Module):
    def __init__(self, min_dist=5):
        super().__init__()

        self.layer = CompactInputReynoldsLayer(min_dist)

    def forward(self, data):
        h = torch.cat([data.pos, data.vel], dim=-1)

        acc_new = self.layer(h, data.edge_index)  # (n, d) -> (n, d)
        return acc_new


class MPNNFlockingLayer(MessagePassing):
    def __init__(self, emb_dim, nn_layers=2, aggr='add', out_dim=4):
        super().__init__(aggr=aggr)
        self.node_dim = 0
        self.emb_dim = emb_dim
        self.out_dim = out_dim

        self.mlp_msg = Sequential(
            Linear(2*4, emb_dim), *([BatchNorm1d(emb_dim), Tanh(),
            Linear(emb_dim, emb_dim)]*(nn_layers - 2)), BatchNorm1d(emb_dim), Tanh(),
            Linear(emb_dim, out_dim), BatchNorm1d(out_dim), Tanh()
        )
        self.mlp_upd = Sequential(
            Linear(out_dim + 4, emb_dim), *([BatchNorm1d(emb_dim), Tanh(),
            Linear(emb_dim, emb_dim)]*(nn_layers - 2)), BatchNorm1d(emb_dim), Tanh(),
            Linear(emb_dim, out_dim), BatchNorm1d(out_dim), Tanh()
        )

    def forward(self, h, edge_index):
        out_acc = self.propagate(edge_index, h=h)
        return out_acc

    def message(self, h_i, h_j):
        msg_h = torch.cat([h_i, h_j], dim=-1)
        msg = self.mlp_msg(msg_h)
        h_dif = h_i - h_j
        msg[(h_dif == 0.0).all(dim=1)] = 0.0
        return msg

    def aggregate(self, msg_in, index):
        add_out = scatter(msg_in[:, :self.out_dim//2], index, dim=self.node_dim, reduce='add')
        mean_out = scatter(msg_in[:, self.out_dim//2:], index, dim=self.node_dim, reduce='mean')
        aggr_out = torch.cat([add_out, mean_out], dim=1)
        return aggr_out

    def update(self, aggr_out, h):
        upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(upd_out)


class MPNNFlockingModel(Module):
    def __init__(self, nn_layers=2, emb_dim=4, out_dim=2, layer_out_dim=4):
        super().__init__()

        self.lin_in = Linear(4, 4)
        self.conv = MPNNFlockingLayer(emb_dim, nn_layers=nn_layers, aggr='add', out_dim=layer_out_dim)

        self.lin_pred = Linear(layer_out_dim, out_dim)
        self.emb_dim = emb_dim
        self.layer_out = layer_out_dim
        self.nn_layers = nn_layers


    def forward(self, data):
        h = torch.cat([data.pos, data.vel], dim=-1)
        h = self.lin_in(h)
        acc_new = self.conv(h, data.edge_index)  # (n, d) -> (n, d)

        out = self.lin_pred(acc_new)  # (batch_size, d) -> (batch_size, 1)

        return out


class BiasedMPNNFlockingLayer(MessagePassing):
    def __init__(self, emb_dim=64, aggr='add', out_dim=4, nn_layers=4):
        super().__init__(aggr=aggr)
        self.node_dim = 0
        self.emb_dim = emb_dim
        self.out_dim = out_dim

        self.mlp_msg = Sequential(
            Linear(4, emb_dim), *([BatchNorm1d(emb_dim), Tanh(),
            Linear(emb_dim, emb_dim)]*(nn_layers - 2)), BatchNorm1d(emb_dim), Tanh(),
            Linear(emb_dim, out_dim)
        )
        self.mlp_upd = Sequential(
            Linear(out_dim, emb_dim), *([BatchNorm1d(emb_dim), Tanh(),
            Linear(emb_dim, emb_dim)]*(nn_layers - 2)), BatchNorm1d(emb_dim), Tanh(),
            Linear(emb_dim, out_dim), BatchNorm1d(out_dim), Tanh()
        )

    def forward(self, h, edge_index):
        out_acc = self.propagate(edge_index, h=h)
        return out_acc

    def message(self, h_i, h_j):
        h_dif = h_i - h_j
        msg = self.mlp_msg(h_dif)
        msg[(h_dif == 0.0).all(dim=1)] = 0.0
        return msg

    def aggregate(self, msg_in, index):
        add_out = scatter(msg_in[:, :self.out_dim//2], index, dim=self.node_dim, reduce='add')
        mean_out = scatter(msg_in[:, self.out_dim//2:], index, dim=self.node_dim, reduce='mean')
        aggr_out = torch.cat([add_out, mean_out], dim=1)
        return aggr_out

    def update(self, aggr_out, h):
        # upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(aggr_out)


class BiasedMPNNFlockingModel(Module):
    def __init__(self, emb_dim=4, out_dim=2, nn_layers=4, layer_out=4):
        super().__init__()

        # self.lin_in = Linear(4, layer_out)

        self.conv = BiasedMPNNFlockingLayer(emb_dim, nn_layers=nn_layers, out_dim=layer_out)

        self.lin_pred = Linear(layer_out, out_dim)
        self.emb_dim = emb_dim
        self.layer_out = layer_out
        self.nn_layers = nn_layers

    def forward(self, data):
        h = torch.cat([data.pos, data.vel], dim=-1)
        # h = self.lin_in(h)
        acc_new = self.conv(h, data.edge_index)  # (n, d) -> (n, d)

        out = self.lin_pred(acc_new)  # (batch_size, d) -> (batch_size, 1)

        return out


class OneAggrMPNNFlockingLayer(MessagePassing):
    def __init__(self, emb_dim=64, aggr='add', out_dim=4, nn_layers=4, use_lin_in=False):
        super().__init__(aggr=aggr)
        self.node_dim = 0
        self.emb_dim = emb_dim
        if use_lin_in:
            self.mlp_msg = Sequential(
                Linear(emb_dim*2, emb_dim), *([BatchNorm1d(emb_dim), Tanh(),
                Linear(emb_dim, emb_dim)]*(nn_layers - 2)), BatchNorm1d(emb_dim), Tanh(),
                Linear(emb_dim, out_dim), BatchNorm1d(out_dim), Tanh()
            )
        else:
            self.mlp_msg = Sequential(
                Linear(8, emb_dim), *([BatchNorm1d(emb_dim), Tanh(),
                Linear(emb_dim, emb_dim)]*(nn_layers - 2)), BatchNorm1d(emb_dim), Tanh(),
                Linear(emb_dim, out_dim), BatchNorm1d(out_dim), Tanh()
            )
        self.mlp_upd = Sequential(
            Linear(out_dim+4, emb_dim), *([BatchNorm1d(emb_dim), Tanh(),
            Linear(emb_dim, emb_dim)]*(nn_layers - 2)), BatchNorm1d(emb_dim), Tanh(),
            Linear(emb_dim, out_dim), BatchNorm1d(out_dim), Tanh()
        )

    def forward(self, h, edge_index):
        out_acc = self.propagate(edge_index, h=h)
        return out_acc

    def message(self, h_i, h_j):
        h_dif = h_i - h_j
        msg = self.mlp_msg(torch.cat([h_i, h_j], dim=1))
        # msg[(h_dif == 0.0).all(dim=1)] = 0.0
        return msg

    # def aggregate(self, msg_in, index):
    #     add_out = scatter(msg_in[:, :2], index, dim=self.node_dim, reduce='add')
    #     mean_out = scatter(msg_in[:, 2:], index, dim=self.node_dim, reduce='mean')
    #     aggr_out = torch.cat([add_out, mean_out], dim=1)
    #     return aggr_out

    def update(self, aggr_out, h):
        upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(upd_out)


class OneAggrMPNNFlockingModel(Module):
    def __init__(self, emb_dim=128, out_dim=2, nn_layers=4, use_lin_in=False, layer_out=4):
        super().__init__()

        self.lin_in = Linear(4, layer_out)

        self.conv = OneAggrMPNNFlockingLayer(emb_dim, nn_layers=nn_layers, out_dim=layer_out, use_lin_in=use_lin_in)
        # self.use_lin_in = use_lin_in
        self.lin_pred = Linear(layer_out, out_dim)
        self.emb_dim = emb_dim
        self.layer_out = layer_out
        self.nn_layers = nn_layers

    def forward(self, data):
        h = torch.cat([data.pos, data.vel], dim=-1)
        # if self.use_lin_in:
        #     h = self.lin_in(h)

        acc_new = self.conv(h, data.edge_index)  # (n, d) -> (n, d)

        out = self.lin_pred(acc_new)  # (batch_size, d) -> (batch_size, 1)

        return out


class FullEmbMPNNFlockingLayer(MessagePassing):
    def __init__(self, emb_dim=64, aggr='add', nn_layers=4):
        super().__init__(aggr=aggr)
        self.node_dim = 0
        self.emb_dim = emb_dim

        self.mlp_msg = Sequential(
            Linear(emb_dim, emb_dim), *([BatchNorm1d(emb_dim), Tanh(),
            Linear(emb_dim, emb_dim)]*(nn_layers - 2)), BatchNorm1d(emb_dim), Tanh(),
            Linear(emb_dim, 2*emb_dim)
        )
        self.mlp_upd = Sequential(
            Linear(emb_dim, emb_dim), *([BatchNorm1d(emb_dim), Tanh(),
            Linear(emb_dim, emb_dim)]*(nn_layers - 2)), BatchNorm1d(emb_dim), Tanh(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), Tanh()
        )

    def forward(self, h, edge_index):
        out_acc = self.propagate(edge_index, h=h)
        return out_acc

    def message(self, h_i, h_j):
        h_dif = h_i - h_j
        msg = self.mlp_msg(h_dif)
        # msg[(h_dif == 0.0).all(dim=1)] = 0.0
        return msg

    def aggregate(self, msg_in, index):
        add_out = scatter(msg_in[:, :self.emb_dim], index, dim=self.node_dim, reduce='add')
        mean_out = scatter(msg_in[:, self.emb_dim:], index, dim=self.node_dim, reduce='mean')
        aggr_out = add_out + mean_out
        return aggr_out

    def update(self, aggr_out, h):
        upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(aggr_out)


class FullEmbMPNNFlockingModel(Module):
    def __init__(self, emb_dim=4, out_dim=2, nn_layers=4):
        super().__init__()
        in_dim = 4
        self.lin_in = Linear(in_dim, emb_dim)

        self.conv = FullEmbMPNNFlockingLayer(emb_dim, nn_layers=nn_layers)

        self.lin_pred = Linear(emb_dim, out_dim)
        self.emb_dim = emb_dim
        self.layer_out = emb_dim
        self.nn_layers = nn_layers

    def forward(self, data):
        h = torch.cat([data.pos, data.vel], dim=-1)
        h = self.lin_in(h)
        acc_new = self.conv(h, data.edge_index)  # (n, d) -> (n, d)

        out = self.lin_pred(acc_new)  # (batch_size, d) -> (batch_size, 1)

        return out


class UpdBiasedMPNNFlockingLayer(MessagePassing):
    def __init__(self, emb_dim=64, aggr='add', out_dim=4, nn_layers=4):
        super().__init__(aggr=aggr)
        self.node_dim = 0
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.mlp_msg = Sequential(
            Linear(8, emb_dim), *([BatchNorm1d(emb_dim), Tanh(),
            Linear(emb_dim, emb_dim)]*(nn_layers - 2)), BatchNorm1d(emb_dim), Tanh(),
            Linear(emb_dim, out_dim)
        )
        self.mlp_upd = Sequential(
            Linear(out_dim, emb_dim), *([BatchNorm1d(emb_dim), Tanh(),
            Linear(emb_dim, emb_dim)]*(nn_layers - 2)), BatchNorm1d(emb_dim), Tanh(),
            Linear(emb_dim, out_dim), BatchNorm1d(out_dim), Tanh()
        )

    def forward(self, h, edge_index):
        out_acc = self.propagate(edge_index, h=h)
        return out_acc

    def message(self, h_i, h_j):
        h_dif = h_i - h_j
        msg = self.mlp_msg(torch.cat([h_i, h_j], dim=1))
        msg[(h_dif == 0.0).all(dim=1)] = 0.0
        return msg

    def aggregate(self, msg_in, index):
        add_out = scatter(msg_in[:, :self.out_dim//2], index, dim=self.node_dim, reduce='add')
        mean_out = scatter(msg_in[:, self.out_dim//2:], index, dim=self.node_dim, reduce='mean')
        aggr_out = torch.cat([add_out, mean_out], dim=1)
        return aggr_out

    def update(self, aggr_out, h):
        # upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(aggr_out)


class UpdBiasedMPNNFlockingModel(Module):
    def __init__(self, emb_dim=4, out_dim=2, nn_layers=4, layer_out=4):
        super().__init__()

        # self.lin_in = Linear(4, layer_out)

        self.conv = UpdBiasedMPNNFlockingLayer(emb_dim, nn_layers=nn_layers, out_dim=layer_out)

        self.lin_pred = Linear(layer_out, out_dim)
        self.emb_dim = emb_dim
        self.layer_out = layer_out
        self.nn_layers = nn_layers

    def forward(self, data):
        h = torch.cat([data.pos, data.vel], dim=-1)
        # h = self.lin_in(h)
        acc_new = self.conv(h, data.edge_index)  # (n, d) -> (n, d)

        out = self.lin_pred(acc_new)  # (batch_size, d) -> (batch_size, 1)

        return out


class MsgBiasedMPNNFlockingLayer(MessagePassing):
    def __init__(self, emb_dim=64, aggr='add', out_dim=4, nn_layers=4):
        super().__init__(aggr=aggr)
        self.node_dim = 0
        self.emb_dim = emb_dim
        self.out_dim = out_dim

        self.mlp_msg = Sequential(
            Linear(4, emb_dim), *([BatchNorm1d(emb_dim), Tanh(),
            Linear(emb_dim, emb_dim)]*(nn_layers - 2)), BatchNorm1d(emb_dim), Tanh(),
            Linear(emb_dim, out_dim)
        )
        self.mlp_upd = Sequential(
            Linear(out_dim+4, emb_dim), *([BatchNorm1d(emb_dim), Tanh(),
            Linear(emb_dim, emb_dim)]*(nn_layers - 2)), BatchNorm1d(emb_dim), Tanh(),
            Linear(emb_dim, out_dim), BatchNorm1d(out_dim), Tanh()
        )

    def forward(self, h, edge_index):
        out_acc = self.propagate(edge_index, h=h)
        return out_acc

    def message(self, h_i, h_j):
        h_dif = h_i - h_j
        msg = self.mlp_msg(h_dif)
        msg[(h_dif == 0.0).all(dim=1)] = 0.0
        return msg

    def aggregate(self, msg_in, index):
        add_out = scatter(msg_in[:, :self.out_dim//2], index, dim=self.node_dim, reduce='add')
        mean_out = scatter(msg_in[:, self.out_dim//2:], index, dim=self.node_dim, reduce='mean')
        aggr_out = torch.cat([add_out, mean_out], dim=1)
        return aggr_out

    def update(self, aggr_out, h):
        upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(upd_out)


class MsgBiasedMPNNFlockingModel(Module):
    def __init__(self, emb_dim=4, out_dim=2, nn_layers=4, layer_out=4):
        super().__init__()


        self.conv = MsgBiasedMPNNFlockingLayer(emb_dim, nn_layers=nn_layers, out_dim=layer_out)

        self.lin_pred = Linear(layer_out, out_dim)
        self.emb_dim = emb_dim
        self.layer_out = layer_out
        self.nn_layers = nn_layers

    def forward(self, data):
        h = torch.cat([data.pos, data.vel], dim=-1)
        acc_new = self.conv(h, data.edge_index)  # (n, d) -> (n, d)

        out = self.lin_pred(acc_new)  # (batch_size, d) -> (batch_size, 1)

        return out


class InterpretedFlockingLayer(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')
        self.node_dim = 0

    def forward(self, h,  edge_index):
        acc_out = self.propagate(edge_index, h=h)
        return acc_out

    def message(self, h_i, h_j):
        inputs = torch.cat([h_i, h_j], dim=1)
        x0, x1, x2, x3, x4, x5, x6, x7 = torch.split(inputs, 1, dim=1)
        h_dif = h_i - h_j
        msg = torch.zeros_like(h_i)
        msg[:, 0:1] = (((x4 - x0) + ((x1 - x5) * 0.40914905)) * 0.028998906)
        msg[:, 1:2] = (((x4 - x0) + ((x1 - x5) * 0.5819344)) * -0.02637788)
        msg[:, 2:3] = (((x1 * -0.07907551) - (x4 - x0)) * -0.026721993)
        msg[:, 3:] = (((((x1 * 0.95594215) - x5) - (x0 * 0.20244296)) - (x4 * -0.17809269)) * 0.026933579)

        msg[(h_dif == 0.0).all(dim=1)] = 0.0
        return msg

    def aggregate(self, inputs, index):
        mean_in, coll = torch.split(inputs, 2, dim=1)
        add_out = scatter(coll, index, dim=self.node_dim, reduce='add')
        mean_out = scatter(mean_in, index, dim=self.node_dim, reduce='mean')
        return torch.cat([add_out, mean_out], dim=1)

    def update(self, aggr_out, h):
        upd = torch.zeros_like(aggr_out)
        inputs = torch.cat([h, aggr_out], dim=1)
        x0, x1, x2, x3, x4, x5, x6, x7 = torch.split(inputs, 1, dim=1)
        upd[:, 0:1] = ((x1 + (x7 / 0.037233025)) * -0.0020586958)
        upd[:, 1:2] = ((((x0 - ((x7)**2 * x5)) * 0.015168043) - x6) * -0.10450508)
        upd[:, 2:3] = (((x7 - (x1 * 0.027931638)) + x6) * 0.075265266)
        upd[:, 3:] = ((((((x3)**2 - x6) + x2) - x3) + (x7 * -0.33928046)) * -0.08554904)

        return upd


class InterpretedFlockingModel(Module):
    def __init__(self):
        super().__init__()

        self.layer = InterpretedFlockingLayer()

    def forward(self, data):
        h = torch.cat([data.pos, data.vel], dim=-1)

        acc_new = self.layer(h, data.edge_index)  # (4, d) -> (2, d)
        pred = torch.zeros([acc_new.shape[0], 2])
        x0, x1, x2, x3 = torch.split(acc_new, 1, dim=1)
        pred[:, :1] = (((((x0 + x3) + x0) * -0.24326763) - (x1 / 0.7301285)) - (x2 * 1.1234615))
        pred[:, 1:] = ((x2 - (x1 + x0)) + x3)
        return pred


class BiasedInterpretedFlockingLayer(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')
        self.node_dim = 0

    def forward(self, h,  edge_index):
        acc_out = self.propagate(edge_index, h=h)
        return acc_out

    def message(self, h_i, h_j):
        inputs = h_i - h_j
        x0, x1, x2, x3 = torch.split(inputs, 1, dim=1)
        h_dif = h_i - h_j
        msg = torch.zeros_like(h_i)
        msg[:, 0:1] = ((x0 - (x1 / ((x0 * 0.07104663)**2 + 1.536996))) * -0.028956918)
        msg[:, 1:2] = ((x0 - (x1 * (0.8290067 - (x0 * -0.021992652)**2))) * 0.025425926)
        msg[:, 2:3] = (((x0 - (x0 * -0.083299406)**2) * -0.024002103) - 0.22298379)
        msg[:, 3:] = (((x1 + 2.6200492) + (x0 * -0.16023761)) * 0.025031794)

        msg[(h_dif == 0.0).all(dim=1)] = 0.0
        return msg

    def aggregate(self, inputs, index):
        mean_in, coll = torch.split(inputs, 2, dim=1)
        add_out = scatter(coll, index, dim=self.node_dim, reduce='add')
        mean_out = scatter(mean_in, index, dim=self.node_dim, reduce='mean')
        return torch.cat([add_out, mean_out], dim=1)

    def update(self, aggr_out, h):
        upd = torch.zeros_like(aggr_out)
        inputs = aggr_out
        x0, x1, x2, x3 = torch.split(inputs, 1, dim=1)
        upd[:, 0:1] = (((x0 - ((x3 + (x2 * 0.15994334)**2) / 1.7044706)) - x2) * 0.16596459)
        upd[:, 1:2] = ((((x1 - ((x2 * -0.089175865)**2 * x3)) - x2) + x3) * -0.05459863)
        upd[:, 2:3] = ((x3 + x0) * 0.05392959)
        upd[:, 3:] = (x2 * (12.305774 / ((x2)**2 - -63.129406)))

        return upd


class BiasedInterpretedFlockingModel(Module):
    def __init__(self):
        super().__init__()

        self.layer = BiasedInterpretedFlockingLayer()

    def forward(self, data):
        h = torch.cat([data.pos, data.vel], dim=-1)

        acc_new = self.layer(h, data.edge_index)  # (4, d) -> (2, d)
        pred = torch.zeros([acc_new.shape[0], 2])
        x0, x1, x2, x3 = torch.split(acc_new, 1, dim=1)
        pred[:, :1] = ((((((x0 / 0.5268826) + x3) - x2) * -0.18549965) - (x1 + x2)) / 0.7328953)
        pred[:, 1:] = (((x0 * -0.8037861) - x1) + ((x3 * 1.2175907) + x2))
        return pred
