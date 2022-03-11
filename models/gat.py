"""
Extended from https://github.com/rusty1s/pytorch_geometric/tree/master/benchmark/citation
"""
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy
from torch_geometric.nn import SGConv
from torch_geometric.nn import APPNP as ModuleAPPNP
# from torch_geometric.nn import GATConv
from .mygatconv import GATConv
import numpy as np
import scipy.sparse as sp

from torch.nn import Linear
from itertools import repeat


class GAT(torch.nn.Module):

    def __init__(self, nfeat, nhid, nclass, heads=8, output_heads=1, dropout=0.5, lr=0.01,
            weight_decay=5e-4, with_bias=True, device=None, **kwargs):

        super(GAT, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay

        if 'dataset' in kwargs:
            if kwargs['dataset'] in ['ogbn-arxiv']:
                dropout = 0.7 # arxiv
            elif kwargs['dataset'] in ['reddit']:
                dropout = 0.05; self.dropout = 0.1; self.weight_decay = 5e-4
                # self.weight_decay = 5e-2; dropout=0.05; self.dropout=0.1
            elif kwargs['dataset'] in ['citeseer']:
                dropout = 0.7
                self.weight_decay = 5e-4
            elif kwargs['dataset'] in ['flickr']:
                dropout = 0.8
                # nhid=8; heads=8
                # self.dropout=0.1
            else:
                dropout = 0.7 # cora, citeseer, reddit
        else:
            dropout = 0.7
        self.conv1 = GATConv(
            nfeat,
            nhid,
            heads=heads,
            dropout=dropout,
            bias=with_bias)

        self.conv2 = GATConv(
            nhid * heads,
            nclass,
            heads=output_heads,
            concat=False,
            dropout=dropout,
            bias=with_bias)

        self.output = None
        self.best_model = None
        self.best_output = None

    # def forward(self, data):
    #     x, edge_index = data.x, data.edge_index
    #     x = F.dropout(x, p=self.dropout, training=self.training)
    #     x = F.elu(self.conv1(x, edge_index))
    #     x = F.dropout(x, p=self.dropout, training=self.training)
    #     x = self.conv2(x, edge_index)
    #     return F.log_softmax(x, dim=1)

    def forward(self, data):
        # x, edge_index = data.x, data.edge_index
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_weight=edge_weight))
        # print(self.conv1.att_l.sum())
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=1)


    def initialize(self):
        """Initialize parameters of GAT.
        """
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


    def fit(self, feat, adj, labels, idx, data=None, train_iters=600, initialize=True, verbose=False, patience=None, noval=False, **kwargs):

        data_train = GraphData(feat, adj, labels)
        data_train = Dpr2Pyg(data_train)[0]

        data_test = Dpr2Pyg(GraphData(data.feat_test, data.adj_test, None))[0]

        if noval:
            data_val = GraphData(data.feat_val, data.adj_val, None)
            data_val = Dpr2Pyg(data_val)[0]
        else:
            data_val = GraphData(data.feat_full, data.adj_full, None)
            data_val = Dpr2Pyg(data_val)[0]

        labels_val = torch.LongTensor(data.labels_val).to(self.device)

        if initialize:
            self.initialize()

        if len(data_train.y.shape) > 1:
            self.multi_label = True
            self.loss = torch.nn.BCELoss()
        else:
            self.multi_label = False
            self.loss = F.nll_loss


        data_train.y = data_train.y.float() if self.multi_label else data_train.y
        # data_val.y = data_val.y.float() if self.multi_label else data_val.y

        if verbose:
            print('=== training gat model ===')

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_acc_val = 0
        best_loss_val = 100
        for i in range(train_iters):
            # if i == train_iters // 2:
            if i in [1500]:
                lr = self.lr*0.1
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)

            self.train()
            optimizer.zero_grad()
            output = self.forward(data_train)
            loss_train = self.loss(output, data_train.y)
            loss_train.backward()
            optimizer.step()

            with torch.no_grad():
                self.eval()

                output = self.forward(data_val)
                if noval:
                    loss_val = F.nll_loss(output, labels_val)
                    acc_val = utils.accuracy(output, labels_val)
                else:
                    loss_val = F.nll_loss(output[data.idx_val], labels_val)
                    acc_val = utils.accuracy(output[data.idx_val], labels_val)


                if loss_val < best_loss_val:
                    best_loss_val = loss_val
                    self.output = output
                    weights = deepcopy(self.state_dict())

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    self.output = output
                    weights = deepcopy(self.state_dict())
                    # print(best_acc_val)
                    # output = self.forward(data_test)
                    # labels_test = torch.LongTensor(data.labels_test).to(self.device)
                    # loss_test = F.nll_loss(output, labels_test)
                    # acc_test = utils.accuracy(output, labels_test)
                    # print('acc_test:', acc_test.item())



            if verbose and i % 100 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def test(self, data_test):
        """Evaluate GCN performance
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(data_test)
        evaluate(output, data_test.y, self.args)

    # @torch.no_grad()
    # def predict(self, data):
    #     self.eval()
    #     return self.forward(data)
    @torch.no_grad()
    def predict(self, feat, adj):
        self.eval()
        data = GraphData(feat, adj, None)
        data = Dpr2Pyg(data)[0]
        return self.forward(data)

    @torch.no_grad()
    def predict_unnorm(self, feat, adj):
        self.eval()
        data = GraphData(feat, adj, None)
        data = Dpr2Pyg(data)[0]

        return self.forward(data)


class GraphData:

    def __init__(self, features, adj, labels, idx_train=None, idx_val=None, idx_test=None):
        self.adj = adj
        self.features = features
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test


from torch_geometric.data import InMemoryDataset, Data
import scipy.sparse as sp

class Dpr2Pyg(InMemoryDataset):

    def __init__(self, dpr_data, transform=None, **kwargs):
        root = 'data/' # dummy root; does not mean anything
        self.dpr_data = dpr_data
        super(Dpr2Pyg, self).__init__(root, transform)
        pyg_data = self.process()
        self.data, self.slices = self.collate([pyg_data])
        self.transform = transform

    def process____(self):
        dpr_data = self.dpr_data
        try:
            edge_index = torch.LongTensor(dpr_data.adj.nonzero().cpu()).cuda().T
        except:
            edge_index = torch.LongTensor(dpr_data.adj.nonzero()).cuda()
        # by default, the features in pyg data is dense
        try:
            x = torch.FloatTensor(dpr_data.features.cpu()).float().cuda()
        except:
            x = torch.FloatTensor(dpr_data.features).float().cuda()
        try:
            y = torch.LongTensor(dpr_data.labels.cpu()).cuda()
        except:
            y = dpr_data.labels

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = None
        data.val_mask = None
        data.test_mask = None
        return data

    def process(self):
        dpr_data = self.dpr_data
        if type(dpr_data.adj) == torch.Tensor:
            adj_selfloop = dpr_data.adj + torch.eye(dpr_data.adj.shape[0]).cuda()
            edge_index_selfloop = adj_selfloop.nonzero().T
            edge_index = edge_index_selfloop
            edge_weight = adj_selfloop[edge_index_selfloop[0], edge_index_selfloop[1]]
        else:
            adj_selfloop = dpr_data.adj + sp.eye(dpr_data.adj.shape[0])
            edge_index = torch.LongTensor(adj_selfloop.nonzero()).cuda()
            edge_weight = torch.FloatTensor(adj_selfloop[adj_selfloop.nonzero()]).cuda()

        # by default, the features in pyg data is dense
        try:
            x = torch.FloatTensor(dpr_data.features.cpu()).float().cuda()
        except:
            x = torch.FloatTensor(dpr_data.features).float().cuda()
        try:
            y = torch.LongTensor(dpr_data.labels.cpu()).cuda()
        except:
            y = dpr_data.labels


        data = Data(x=x, edge_index=edge_index, y=y, edge_weight=edge_weight)
        data.train_mask = None
        data.val_mask = None
        data.test_mask = None
        return data

    def get(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[self.data.__cat_dim__(key, item)] = slice(slices[idx],
                                                   slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def _download(self):
        pass

