


import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import pylab

from matplotlib.pyplot import pause
from torch.distributions import Categorical
from gcn.models_gcn import GCN_Policy_SelectNode, GCN_Sparse_Policy_SelectNode, GCN_Sparse_Memory_Policy_SelectNode

from data.graph import Graph
from utils.utils import open_dataset, varname
from utils.utils import erdosrenyi
from utils import utils


# Training argument setting
parser = argparse.ArgumentParser()
parser.add_argument('--nocuda', action= 'store_true', default=False, help='Disable Cuda')
parser.add_argument('--novalidation', action= 'store_true', default=True, help='Disable validation')
parser.add_argument('--seed', type=int, default=63, help='Radom seed') #50
parser.add_argument('--epochs', type=int, default=21, help='Training epochs')
parser.add_argument('--lr', type=float, default= 0.0001, help='Learning rate')
parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay')
parser.add_argument('--dhidden', type=int, default=1, help='Dimension of hidden features')
parser.add_argument('--dinput', type=int, default=1, help='Dimension of input features')
parser.add_argument('--doutput', type=int, default=1, help='Dimension of output features')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout Rate')
parser.add_argument('--alpha', type=float, default=0.2, help='Aplha')
parser.add_argument('--nnode', type=int, default=200, help='Number of node per graph')
parser.add_argument('--ngraph', type=int, default=200, help='Number of graph per dataset')
parser.add_argument('--p', type=int, default=0.1, help='probiblity of edges')
parser.add_argument('--nnode_test', type=int, default=300, help='Number of node per graph for test')
parser.add_argument('--ngraph_test', type=int, default=100, help='Number of graph for test dataset')
args = parser.parse_args()

args.cuda = not args.nocuda and torch.cuda.is_available()
print("cuda available: {} ".format(torch.cuda.is_available()))

np.random.seed(30)
torch.manual_seed(30)

g = Graph.erdosrenyi(n=10, p=0.3)
g2 = Graph(g.M)
g_for_model = Graph(g.M)
g2_for_model = Graph(g.M)

train_ER_small, val_ER_small, test_ER_small = open_dataset('./data/ERGcollection/erg_small.pkl')

model = GCN_Sparse_Policy_SelectNode(nin=args.dinput,
                              nhidden= args.dhidden,
                              nout=args.doutput,
                              dropout=args.dropout,
                              ) # alpha=args.alpha

for name, param in model.named_parameters():
                print('parameter name {}'.format(name),
                    'parameter value {}'.format(param.data.size()))

epoch = 0
heuristic = 'one_step_greedy' # 'one_step_greedy' 'min_degree'
train_dataset=train_ER_small

if args.cuda:
    model.load_state_dict(
        torch.load('./supervised/models/' + heuristic + '/SmallErgTraining/lrss' + str(
            args.lr) + '/per_epochs/gcn_policy_' + heuristic + '_pre_' + train_dataset.__class__.__name__
                   + '_epochs_' + str(epoch) + '_cuda.pth'))
else:
    # device = torch.device('cpu')
    model.load_state_dict(
    torch.load('./supervised/models/' + heuristic + '/SmallErgTraining/lrss' + str(
        args.lr) + '/per_epochs/gcn_policy_' + heuristic + '_pre_' + train_dataset.__class__.__name__
               + '_epochs_' + str(epoch) + '_cuda.pth',  map_location='cpu'))

for name, param in model.named_parameters():
    print('parameter name {}'.format(name),
          'parameter value {}'.format(param.data))
