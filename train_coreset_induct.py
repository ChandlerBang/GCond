from deeprobust.graph.data import Dataset
import numpy as np
import random
import time
import argparse
import torch
import sys
import deeprobust.graph.utils as utils
import torch.nn.functional as F
from configs import load_config
from utils import *
from utils_graphsaint import DataGraphSAINT
from models.gcn import GCN
from coreset import KCenter, Herding, Random
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--keep_ratio', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--nlayers', type=int, default=2, help='Random seed.')
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--inductive', type=int, default=1)
parser.add_argument('--mlp', type=int, default=0)
parser.add_argument('--method', type=str, choices=['kcenter', 'herding', 'random'])
parser.add_argument('--reduction_rate', type=float, required=True)
args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)
args = load_config(args)
print(args)

# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

data_graphsaint = ['flickr', 'reddit', 'ogbn-arxiv']
if args.dataset in data_graphsaint:
    data = DataGraphSAINT(args.dataset)
    data_full = data.data_full
    data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)
else:
    data_full = get_dataset(args.dataset, args.normalize_features)
    data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)

feat_train = data.feat_train
adj_train = data.adj_train
labels_train = data.labels_train


# Setup GCN Model
device = 'cuda'
model = GCN(nfeat=feat_train.shape[1], nhid=256, nclass=labels_train.max()+1, device=device, weight_decay=args.weight_decay)

model = model.to(device)

model.fit_with_val(feat_train, adj_train, labels_train, data,
             train_iters=600, normalize=True, verbose=False)

model.eval()
labels_test = torch.LongTensor(data.labels_test).cuda()
feat_test, adj_test = data.feat_test, data.adj_test

embeds = model.predict().detach()

output = model.predict(feat_test, adj_test)
loss_test = F.nll_loss(output, labels_test)
acc_test = utils.accuracy(output, labels_test)
print("Test set results:",
      "loss= {:.4f}".format(loss_test.item()),
      "accuracy= {:.4f}".format(acc_test.item()))


if args.method == 'kcenter':
    agent = KCenter(data, args, device='cuda')
if args.method == 'herding':
    agent = Herding(data, args, device='cuda')
if args.method == 'random':
    agent = Random(data, args, device='cuda')

idx_selected = agent.select(embeds, inductive=True)

feat_train = feat_train[idx_selected]
adj_train = adj_train[np.ix_(idx_selected, idx_selected)]

labels_train = labels_train[idx_selected]

res = []
print('shape of feat_train:', feat_train.shape)
runs = 10
for _ in tqdm(range(runs)):
    model.initialize()
    model.fit_with_val(feat_train, adj_train, labels_train, data,
                 train_iters=600, normalize=True, verbose=False, noval=True)

    model.eval()
    labels_test = torch.LongTensor(data.labels_test).cuda()

    output = model.predict(feat_test, adj_test)
    loss_test = F.nll_loss(output, labels_test)
    acc_test = utils.accuracy(output, labels_test)
    res.append(acc_test.item())
res = np.array(res)
print('Mean accuracy:', repr([res.mean(), res.std()]))

