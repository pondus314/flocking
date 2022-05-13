import torch
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing


class RegressionLayer(MessagePassing):
    def __init__(self, msg_regression_model, upd_regression_model, msg_model=None, upd_model=None, full_model=None):
        super().__init__()

        self.msg_model = msg_model
        self.upd_model = upd_model
        self.full_model = full_model

        self.msg_regression_model = msg_regression_model
        self.upd_regression_model = upd_regression_model
        self.eval()

    def forward(self, h, edge_index):
        out_acc = self.propagate(edge_index, h=h)
        return out_acc

    def message(self, h_i, h_j):
        msg_h = torch.cat([h_i, h_j], dim=-1)
        if self.full_model is None:
            msg = self.msg_model(msg_h)
        else:
            msg = self.full_model.message(h_i, h_j)
        self.msg_regression_model.fit(msg_h[:8000, :], msg[:8000, :])
        return msg

    def aggregate(self, msg_in, index):
        aggr_out = scatter(msg_in, index, dim=self.node_dim, reduce=self.aggr)
        return aggr_out

    def update(self, aggr_out, h):
        upd_in = torch.cat([h, aggr_out], dim=-1)
        if self.full_model is None:
            upd_out = self.upd_model(upd_in)
        else:
            upd_out = self.full_model.update(aggr_out, h)
        self.upd_regression_model.fit(upd_in[:8000, :], upd_out[:8000, :])
        return upd_out


class RegressionModel():
    def __init__(self, msg_regression_model, upd_regression_model, msg_model=None, upd_model=None, full_model=None):
        super().__init__()

        self.conv = RegressionLayer(msg_regression_model, upd_regression_model, msg_model, upd_model, full_model)

    def forward(self, data):
        h = torch.cat([data.pos, data.vel], dim=-1)

        acc_new = self.conv(h, data.edge_index)  # (n, d) -> (n, d)

        return acc_new

