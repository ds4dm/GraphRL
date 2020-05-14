import networkx as nx
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
# pylab.ion()

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

np.random.seed(30)
torch.manual_seed(30)
n = 7
g = Graph.erdosrenyi(n=n, p=0.50)
g2 = Graph(g.M)
g_for_model = Graph(g.M)
g2_for_model = Graph(g.M)
graph = nx.Graph(g.M)
pos = nx.spring_layout(graph)

train_ER_small, val_ER_small, test_ER_small = open_dataset('./data/ERGcollection/erg_small.pkl')

model = GCN_Sparse_Policy_SelectNode(nin=args.dinput,
                              nhidden= args.dhidden,
                              nout=args.doutput,
                              dropout=args.dropout,
                              ) # alpha=args.alpha

for name, param in model.named_parameters():
                print('parameter name {}'.format(name),
                    'parameter value {}'.format(param.data.size()))

epoch = 8
heuristic = 'one_step_greedy' # 'one_step_greedy' 'min_degree'
train_dataset=train_ER_small

if args.cuda:
    model.load_state_dict(
        torch.load('./supervised/models/' + heuristic + '/SmallErgTraining/lr' + str(
            args.lr) + '/per_epochs/gcn_policy_' + heuristic + '_pre_' + train_dataset.__class__.__name__
                   + '_epochs_' + str(epoch) + '_cuda.pth'))
else:
    device = torch.device('cpu')
    model.load_state_dict(
    torch.load('./supervised/models/' + heuristic + '/SmallErgTraining/layers3_test_lr' + str(
        args.lr) + '/per_epochs/gcn_policy_' + heuristic + '_pre_' + train_dataset.__class__.__name__
               + '_epochs_' + str(epoch) + '_cuda.pth',  map_location=device))

for name, param in model.named_parameters():
    print('parameter name {}'.format(name),
          'parameter value {}'.format(param.data))


def step_model(model, M, use_cuda=True):

    features = np.ones([M.shape[0], 1], dtype=np.float32)
    m = torch.FloatTensor(M)
    m = utils.to_sparse(m)  # convert to coo sparse tensor

    features = torch.FloatTensor(features)

    if use_cuda:
        m = m.cuda()
        features = features.cuda()

    output = model(features, m)
    output = output.view(-1)

    # m = Categorical(logits=output)  # logits=probs
    # action_gcn = m.sample()
    action_gcn = output.argmax()

    return action_gcn

def get_fig(g, g_for_model, model, pos, axs, i, use_cuda=True):


    # graph_0 = nx.Graph(g.M)
    # graph_1 =nx.Graph(g.M_ex)
    # graph_init = nx.Graph(g.M_init)
    M_eli = np.logical_xor(np.logical_or(g.M_ex, g.M_init), g.M)

    g_init_res = nx.Graph(np.logical_and(g.M_init, g.M))
    g_init_eli = nx.Graph(np.logical_and(g.M_init, M_eli))
    g_ex_res = nx.Graph(np.logical_and(g.M_ex, g.M))
    g_ex_eli = nx.Graph(np.logical_and(g.M_ex, M_eli))



    # graph_redu_model = nx.Graph(g_for_model.M)
    # graph_exte_model = nx.Graph(g_for_model.M_ex)
    # graph_init_model = nx.Graph(g_for_model.M_init)
    M_model_eli = np.logical_xor(np.logical_or(g_for_model.M_ex, g_for_model.M_init), g_for_model.M)

    g_Model_init_res = nx.Graph(np.logical_and(g_for_model.M_init, g_for_model.M))
    g_Model_init_eli = nx.Graph(np.logical_and(g_for_model.M_init, M_model_eli))
    g_Model_ex_res = nx.Graph(np.logical_and(g_for_model.M_ex, g_for_model.M))
    g_Model_ex_eli = nx.Graph(np.logical_and(g_for_model.M_ex, M_model_eli))


    # ax_0 = axs_row[0]
    # ax_1 = axs_row[1]
    # nx.draw(graph_0, pos=pos,ax= ax_0)
    # nx.draw(graph_init, pos=pos, ax=ax_1)
    # nx.draw(graph_1, pos=pos, edge_color='r', ax=ax_1)

    ax_0 = axs[0][i]
    if not i==0:
        axs[0][i].set_title('step '+str(i), fontsize=20)
    nx.draw(g_init_res, pos=pos, ax=ax_0)
    nx.draw(g_init_eli, pos=pos, style='dashed', ax=ax_0)
    nx.draw(g_ex_res, pos=pos, edge_color='r', ax=ax_0)
    nx.draw(g_ex_eli, pos=pos, edge_color='r', style='dashed', ax=ax_0)

    # ax_0.set_frame_on(True)

    # print(g.vertices)
    action, d_min = g.min_degree(g2.M)
    nx.draw_networkx_nodes(g_ex_eli, pos=pos, nodelist=[g2.vertices[action]], node_color='orange', ax=ax_0)
    g.eliminate_node(g2.vertices[action],reduce=False)
    g2.eliminate_node(action, reduce=True)



    # ax_2 = axs_row[2]
    # ax_3 = axs_row[3]
    # nx.draw(graph_redu_model, pos=pos, ax=ax_2)
    # nx.draw(graph_init_model, pos=pos, ax=ax_3)
    # nx.draw(graph_exte_model, pos=pos, edge_color='r', ax=ax_3)
    # # ax_0.set_frame_on(True)

    ax_1 = axs[1][i]
    if not i == 0:
        axs[1][i].set_title('step ' + str(i), fontsize=20)

    nx.draw(g_Model_init_res, pos=pos, ax=ax_1)
    nx.draw(g_Model_init_eli, pos=pos, style='dashed', ax=ax_1)
    nx.draw(g_Model_ex_res, pos=pos, edge_color='r', ax=ax_1)
    nx.draw(g_Model_ex_eli, pos=pos, edge_color='r', style='dashed', ax=ax_1)
    # ax_0.set_frame_on(True)

    action, d_min = g2_for_model.min_degree(g2_for_model.M)
    if not (d_min == 0 or d_min==1):
        action = step_model(model, g2_for_model.M,use_cuda=use_cuda)

    nx.draw_networkx_nodes(g_Model_ex_eli, pos=pos, nodelist=[g2_for_model.vertices[action]], node_color='orange', ax=ax_1)
    g_for_model.eliminate_node(g2_for_model.vertices[action], reduce=False)
    g2_for_model.eliminate_node(action, reduce=True)


    return g



fig, axs = plt.subplots(nrows=2, ncols=n-1, figsize=(20,8), constrained_layout=True)
axs[0][0].set_title('Min degree step 0', fontsize=20)
axs[1][0].set_title('GNN policy step 0', fontsize=20)

for i in range(n-1):

    g = get_fig(g=g, g_for_model=g_for_model, model=model, pos=pos, axs=axs, i=i, use_cuda=args.cuda)
plt.show()



