import os
import os.path as osp
import glob

import torch
import torch.nn.functional as F
import numpy as np
import torch_geometric.utils
from torch_sparse import coalesce
from torch_geometric.io import read_txt_array
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import Data


class DataWriter:
    def __init__(self, path, dataset_name):
        self.ds_name = dataset_name
        self.path = path
        self.node_counter = 0
        self.graph_counter = 0
        base_file_path = osp.join(path, dataset_name)
        os.mkdir(osp.dirname(path))
        os.mkdir(path)
        self.files = {
            name: open(base_file_path+f'_{name}.txt', 'a') for name in ['A', 'graph_ind', 'pos', 'vel', 'acc']
        }
        self.writeable = True

    def write_data(self, edge_index, node_poss, node_vels, accs_out):
        if not self.writeable:
            return
        v = node_poss.shape[0]

        index_to_file = torch_geometric.utils.remove_self_loops(edge_index)[0].T + self.node_counter
        self.node_counter += v
        np.savetxt(self.files['A'], index_to_file.numpy(), fmt='%i')
        np.savetxt(self.files['pos'], node_poss.numpy())
        np.savetxt(self.files['vel'], node_vels.numpy())
        np.savetxt(self.files['acc'], accs_out.numpy())
        print('\n'.join([str(self.graph_counter) for _ in range(v)]), file=self.files['graph_ind'])
        self.graph_counter += 1

    def finish_writing(self):
        for file in self.files:
            self.files[file].close()


def read_flocking_data(folder, prefix):
    files = glob.glob(osp.join(folder, f'{prefix}_*.txt'))
    names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in files]

    edge_index = read_file(folder, prefix, 'A', torch.long).T
    batch = read_file(folder, prefix, 'graph_ind', torch.long)

    node_poss = read_file(folder, prefix, 'pos')
    node_vels = read_file(folder, prefix, 'vel')

    y = None
    if 'acc' in names:
        y = read_file(folder, prefix, 'acc')

    edge_index = remove_self_loops(edge_index)[0]  # maybe remove this

    data = Data(pos=node_poss, vel=node_vels, edge_index=edge_index, y=y)
    data, slices = split(data, batch)

    return data, slices


def read_file(folder, prefix, name, dtype=None):
    path = osp.join(folder, '{}_{}.txt'.format(prefix, name))
    return read_txt_array(path, sep=' ', dtype=dtype)


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1) if len(seq) > 0 else None


def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    data.__num_nodes__ = torch.bincount(batch).tolist()

    slices = {'edge_index': edge_slice}
    slices['pos'] = node_slice
    slices['vel'] = node_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)

    return data, slices


