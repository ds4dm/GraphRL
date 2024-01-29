import torch.nn as nn
import torch.nn.functional as F

from gcn.layers_gcn import GraphConvolutionLayer, GraphConvolutionLayer_Sparse, GraphConvolutionLayer_Sparse_Memory, GraphAttentionConvLayer, GraphAttentionConvLayerMemory, MessagePassing_GNN_Layer_Sparse_Memory, MessagePassing_GNN_Layer_Sparse


class GCN_Policy_SelectNode(nn.Module):
    """
    GCN model for node selection policy
    """
    def __init__(self, nin, nhidden, nout, dropout):
        super(GCN_Policy_SelectNode, self).__init__()

        self.gc1 = GraphConvolutionLayer(nin, nhidden) # first graph conv layer
        self.gc2 = GraphConvolutionLayer(nhidden, nout) # second graph conv layer
        self.dropout = dropout


    def forward(self, features, adj_matrix):
        # first layer with relu
        features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc1(features, adj_matrix)
        features = F.relu(features)

        # second layer with softmax
        features = self.gc2(features, adj_matrix)
        features = F.log_softmax(features.t())
        features = features.t()

        return features


class GCN_Value(nn.Module):
    """
    GCN model for value function, the negative number of edges added
    """
    def __init__(self, nin, nhidden, nout, dropout):
        super(GCN_Value, self).__init__()

        self.gc1 = GraphConvolutionLayer(nin, nhidden) # first graph conv layer
        self.gc2 = GraphConvolutionLayer(nhidden, nhidden) # second graph conv layer
        self.gc3 = GraphConvolutionLayer(nhidden, nhidden)  # second graph conv layer

        self.dropout = dropout
        self.linear = nn.Linear(nhidden, nout,  bias=False)

    def forward(self, features, adj_matrix):
        # first layer with relu
        #features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc1(features, adj_matrix)
        features = F.relu(features)
        #
        # # second layer with relu
        # features = self.gc2(features, adj_matrix)
        # features = F.relu(features)
        #
        # #third layer with relu
        features = self.gc3(features, adj_matrix)
        # features = F.relu(features)
        features = features

        return features


class GCN_Sparse_Policy_SelectNode(nn.Module):
    """
    GCN model for node selection policy
    """
    def __init__(self, nin, nhidden, nout, dropout):
        super(GCN_Sparse_Policy_SelectNode, self).__init__()

        self.gc1 = GraphConvolutionLayer_Sparse(nin, nhidden) # first graph conv layer
        self.gc2 = GraphConvolutionLayer_Sparse(nhidden, nhidden)  # first graph conv layer
        # self.gc3 = GraphConvolutionLayer_Sparse(nhidden, nhidden)  # first graph conv layer
        # self.gc4 = GraphConvolutionLayer_Sparse(nhidden, nhidden)  # first graph conv layer
        self.gc3 = GraphConvolutionLayer_Sparse(nhidden, nout) # second graph conv layer
        self.dropout = dropout


    def forward(self, features, adj_matrix):
        # first layer with relu
        # features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc1(features, adj_matrix)
        features = F.relu(features)

        features = self.gc2(features, adj_matrix)
        features = F.relu(features)
        #
        # features = self.gc3(features, adj_matrix)
        # features = F.relu(features)

        # features = self.gc4(features, adj_matrix)
        # features = F.relu(features)


        # second layer with softmax
        features = self.gc3(features, adj_matrix)
        features = F.log_softmax(features.t())
        features = features.t()

        return features

class GCN_Sparse_Policy_5(nn.Module):
    """
    GCN model for node selection policy
    """
    def __init__(self, nin, nhidden, nout, dropout):
        super(GCN_Sparse_Policy_5, self).__init__()

        self.gc1 = GraphConvolutionLayer_Sparse(nin, nhidden) # first graph conv layer
        self.gc2 = GraphConvolutionLayer_Sparse(nhidden, nhidden)  # first graph conv layer
        self.gc3 = GraphConvolutionLayer_Sparse(nhidden, nhidden)  # first graph conv layer
        self.gc4 = GraphConvolutionLayer_Sparse(nhidden, nhidden)  # first graph conv layer
        self.gc5 = GraphConvolutionLayer_Sparse(nhidden, nout) # second graph conv layer
        self.dropout = dropout


    def forward(self, features, adj_matrix):
        # first layer with relu
        # features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc1(features, adj_matrix)
        features = F.relu(features)

        features = self.gc2(features, adj_matrix)
        features = F.relu(features)

        features = self.gc3(features, adj_matrix)
        features = F.relu(features)

        features = self.gc4(features, adj_matrix)
        features = F.relu(features)

        # second layer with softmax
        features = self.gc5(features, adj_matrix)
        features = F.log_softmax(features.view(-1))
        # features = features.t()

        return features

class GCN_Sparse_Policy_10(nn.Module):
    """
    GCN model for node selection policy
    """
    def __init__(self, nin, nhidden, nout, dropout):
        super(GCN_Sparse_Policy_10, self).__init__()

        self.gc1 = GraphConvolutionLayer_Sparse(nin, nhidden) # first graph conv layer
        self.gc2 = GraphConvolutionLayer_Sparse(nhidden, nhidden)  # first graph conv layer
        self.gc3 = GraphConvolutionLayer_Sparse(nhidden, nhidden)  # first graph conv layer
        self.gc4 = GraphConvolutionLayer_Sparse(nhidden, nhidden)  # first graph conv layer
        self.gc5 = GraphConvolutionLayer_Sparse(nhidden, nhidden)  # first graph conv layer
        self.gc6 = GraphConvolutionLayer_Sparse(nhidden, nhidden)  # first graph conv layer
        self.gc7 = GraphConvolutionLayer_Sparse(nhidden, nhidden)  # first graph conv layer
        self.gc8 = GraphConvolutionLayer_Sparse(nhidden, nhidden)  # first graph conv layer
        self.gc9 = GraphConvolutionLayer_Sparse(nhidden, nhidden)  # first graph conv layer
        self.gc10 = GraphConvolutionLayer_Sparse(nhidden, nout) # second graph conv layer
        self.dropout = dropout


    def forward(self, features, adj_matrix):
        # first layer with relu
        # features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc1(features, adj_matrix)
        features = F.relu(features)

        features = self.gc2(features, adj_matrix)
        features = F.relu(features)

        features = self.gc3(features, adj_matrix)
        features = F.relu(features)

        features = self.gc4(features, adj_matrix)
        features = F.relu(features)

        features = self.gc5(features, adj_matrix)
        features = F.relu(features)

        features = self.gc6(features, adj_matrix)
        features = F.relu(features)

        features = self.gc7(features, adj_matrix)
        features = F.relu(features)

        features = self.gc8(features, adj_matrix)
        features = F.relu(features)

        features = self.gc9(features, adj_matrix)
        features = F.relu(features)

        # second layer with softmax
        features = self.gc10(features, adj_matrix)
        features = F.log_softmax(features.view(-1))
        # features = features.t()

        return features


class GCN_Sparse_Memory_Policy_SelectNode(nn.Module):
    """
    GCN model for node selection policy
    """
    def __init__(self, nin, nhidden, nout, dropout):
        super(GCN_Sparse_Memory_Policy_SelectNode, self).__init__()

        self.gc1 = GraphConvolutionLayer_Sparse_Memory(nin, nhidden) # first graph conv layer
        self.gc2 = GraphConvolutionLayer_Sparse_Memory(nhidden, nhidden)  # second graph conv layer
        self.gc3 = GraphConvolutionLayer_Sparse_Memory(nhidden, nout) # second graph conv layer
        self.dropout = dropout


    def forward(self, features, adj_matrix):
        # first layer with relu
        # features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc1(features, adj_matrix)
        features = F.relu(features)

        features = self.gc2(features, adj_matrix)
        features = F.relu(features)

        # second layer with softmax
        features = self.gc3(features, adj_matrix)
        features = F.log_softmax(features.view(-1))
        # features = features.t()

        return features

class GCN_Sparse_Memory_Policy_SelectNode_5(nn.Module):
    """
    GCN model for node selection policy
    """
    def __init__(self, nin, nhidden, nout, dropout):
        super(GCN_Sparse_Memory_Policy_SelectNode_5, self).__init__()

        self.gc1 = GraphConvolutionLayer_Sparse_Memory(nin, nhidden) # first graph conv layer
        self.gc2 = GraphConvolutionLayer_Sparse_Memory(nhidden, nhidden)  # first graph conv layer
        self.gc3 = GraphConvolutionLayer_Sparse_Memory(nhidden, nhidden)  # first graph conv layer
        self.gc4 = GraphConvolutionLayer_Sparse_Memory(nhidden, nhidden)  # first graph conv layer
        self.gc5 = GraphConvolutionLayer_Sparse_Memory(nhidden, nout) # second graph conv layer
        self.dropout = dropout


    def forward(self, features, adj_matrix):
        # first layer with relu
        # features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc1(features, adj_matrix)
        features = F.relu(features)
        features = self.gc2(features, adj_matrix)
        features = F.relu(features)
        features = self.gc3(features, adj_matrix)
        features = F.relu(features)
        features = self.gc4(features, adj_matrix)
        features = F.relu(features)

        # second layer with softmax
        features = self.gc5(features, adj_matrix)
        features = F.log_softmax(features.view(-1))
        # features = features.t()

        return features


class GCN_Sparse_Memory_Policy_SelectNode_10(nn.Module):
    """
    GCN model for node selection policy
    """
    def __init__(self, nin, nhidden, nout, dropout):
        super(GCN_Sparse_Memory_Policy_SelectNode_10, self).__init__()

        self.gc1 = GraphConvolutionLayer_Sparse_Memory(nin, nhidden) # first graph conv layer
        self.gc2 = GraphConvolutionLayer_Sparse_Memory(nhidden, nhidden)  # first graph conv layer
        self.gc3 = GraphConvolutionLayer_Sparse_Memory(nhidden, nhidden)  # first graph conv layer
        self.gc4 = GraphConvolutionLayer_Sparse_Memory(nhidden, nhidden)  # first graph conv layer
        self.gc5 = GraphConvolutionLayer_Sparse_Memory(nhidden, nhidden)  # first graph conv layer
        self.gc6 = GraphConvolutionLayer_Sparse_Memory(nhidden, nhidden)  # first graph conv layer
        self.gc7 = GraphConvolutionLayer_Sparse_Memory(nhidden, nhidden)  # first graph conv layer
        self.gc8 = GraphConvolutionLayer_Sparse_Memory(nhidden, nhidden)  # first graph conv layer
        self.gc9 = GraphConvolutionLayer_Sparse_Memory(nhidden, nhidden)  # first graph conv layer
        self.gc10 = GraphConvolutionLayer_Sparse_Memory(nhidden, nout) # second graph conv layer
        self.dropout = dropout


    def forward(self, features, adj_matrix):
        # first layer with relu
        # features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc1(features, adj_matrix)
        features = F.relu(features)
        features = self.gc2(features, adj_matrix)
        features = F.relu(features)
        features = self.gc3(features, adj_matrix)
        features = F.relu(features)
        features = self.gc4(features, adj_matrix)
        features = F.relu(features)
        features = self.gc5(features, adj_matrix)
        features = F.relu(features)
        features = self.gc6(features, adj_matrix)
        features = F.relu(features)
        features = self.gc7(features, adj_matrix)
        features = F.relu(features)
        features = self.gc8(features, adj_matrix)
        features = F.relu(features)
        features = self.gc9(features, adj_matrix)
        features = F.relu(features)

        # second layer with softmax
        features = self.gc10(features, adj_matrix)
        features = F.log_softmax(features.view(-1))
        # features = features.t()

        return features



class GCN_Sparse_Value(nn.Module):
    """
    GCN model for value function, the negative number of edges added
    """
    def __init__(self, nin, nhidden, nout, dropout):
        super(GCN_Sparse_Value, self).__init__()

        self.gc1 = GraphConvolutionLayer_Sparse(nin, nhidden) # first graph conv layer
        self.gc2 = GraphConvolutionLayer_Sparse(nhidden, nhidden) # second graph conv layer
        self.gc3 = GraphConvolutionLayer_Sparse(nhidden, nhidden)  # second graph conv layer

        self.dropout = dropout
        self.linear = nn.Linear(nhidden, nout,  bias=False)

    def forward(self, features, adj_matrix):
        # first layer with relu
        #features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc1(features, adj_matrix)
        features = F.relu(features)
        #
        # # second layer with relu
        # features = self.gc2(features, adj_matrix)
        # features = F.relu(features)
        #
        # #third layer with relu
        features = self.gc3(features, adj_matrix)
        features = F.relu(features)

        features = self.linear(features)
        # features = features

        return features

class GAN(nn.Module):
    def __init__(self, nin, nhidden, nout, dropout, alpha):
        super(GAN, self).__init__()

        self.gc1 = GraphAttentionConvLayer(nin, nhidden, dropout, alpha)
        self.gc2 = GraphAttentionConvLayer(nhidden, nout, dropout, alpha)
        self.dropout = dropout

    def forward(self, features, adj_matrix):
        # features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc1(features, adj_matrix)
        features = F.relu(features)
        # features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc2(features, adj_matrix)
        features = F.log_softmax(features.t())
        features = features.t()
        return features


class GAN_5(nn.Module):
    def __init__(self, nin, nhidden, nout, dropout, alpha):
        super(GAN_5, self).__init__()

        self.gc1 = GraphAttentionConvLayer(nin, nhidden, dropout, alpha)
        self.gc2 = GraphAttentionConvLayer(nhidden, nhidden, dropout, alpha)
        self.gc3= GraphAttentionConvLayer(nhidden, nhidden, dropout, alpha)
        self.gc4 = GraphAttentionConvLayer(nhidden, nhidden, dropout, alpha)
        self.gc5 = GraphAttentionConvLayer(nhidden, nout, dropout, alpha)
        self.dropout = dropout

    def forward(self, features, adj_matrix):
        # features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc1(features, adj_matrix)
        features = F.relu(features)
        features = self.gc2(features, adj_matrix)
        features = F.relu(features)
        features = self.gc3(features, adj_matrix)
        features = F.relu(features)
        features = self.gc4(features, adj_matrix)
        features = F.relu(features)
        # features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc5(features, adj_matrix)
        features = F.log_softmax(features.t())
        features = features.t()
        return features


class GAN_Memory_5(nn.Module):
    def __init__(self, nin, nhidden, nout, dropout, alpha):
        super(GAN_Memory_5, self).__init__()

        self.gc1 = GraphAttentionConvLayerMemory(nin, nhidden, dropout, alpha)
        self.gc2 = GraphAttentionConvLayerMemory(nhidden, nhidden, dropout, alpha)
        self.gc3=  GraphAttentionConvLayerMemory(nhidden, nhidden, dropout, alpha)
        self.gc4 = GraphAttentionConvLayerMemory(nhidden, nhidden, dropout, alpha)
        self.gc5 = GraphAttentionConvLayerMemory(nhidden, nout, dropout, alpha)
        self.dropout = dropout

    def forward(self, features, adj_matrix):
        # features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc1(features, adj_matrix)
        features = F.relu(features)
        features = self.gc2(features, adj_matrix)
        features = F.relu(features)
        features = self.gc3(features, adj_matrix)
        features = F.relu(features)
        features = self.gc4(features, adj_matrix)
        features = F.relu(features)
        # features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc5(features, adj_matrix)
        features = F.log_softmax(features.view(-1))
        # features = features.t()
        return features

class GAN_Memory_10(nn.Module):
    def __init__(self, nin, nhidden, nout, dropout, alpha):
        super(GAN_Memory_10, self).__init__()

        self.gc1 = GraphAttentionConvLayerMemory(nin, nhidden, dropout, alpha)
        self.gc2 = GraphAttentionConvLayerMemory(nhidden, nhidden, dropout, alpha)
        self.gc3=  GraphAttentionConvLayerMemory(nhidden, nhidden, dropout, alpha)
        self.gc4 = GraphAttentionConvLayerMemory(nhidden, nhidden, dropout, alpha)
        self.gc5 = GraphAttentionConvLayerMemory(nhidden, nhidden, dropout, alpha)
        self.gc6 = GraphAttentionConvLayerMemory(nhidden, nhidden, dropout, alpha)
        self.gc7 = GraphAttentionConvLayerMemory(nhidden, nhidden, dropout, alpha)
        self.gc8 = GraphAttentionConvLayerMemory(nhidden, nhidden, dropout, alpha)
        self.gc9 = GraphAttentionConvLayerMemory(nhidden, nhidden, dropout, alpha)
        self.gc10 = GraphAttentionConvLayerMemory(nhidden, nout, dropout, alpha)
        self.dropout = dropout

    def forward(self, features, adj_matrix):
        # features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc1(features, adj_matrix)
        features = F.relu(features)
        features = self.gc2(features, adj_matrix)
        features = F.relu(features)
        features = self.gc3(features, adj_matrix)
        features = F.relu(features)
        features = self.gc4(features, adj_matrix)
        features = F.relu(features)
        features = self.gc5(features, adj_matrix)
        features = F.relu(features)
        features = self.gc6(features, adj_matrix)
        features = F.relu(features)
        features = self.gc7(features, adj_matrix)
        features = F.relu(features)
        features = self.gc8(features, adj_matrix)
        features = F.relu(features)
        features = self.gc9(features, adj_matrix)
        features = F.relu(features)
        # features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc10(features, adj_matrix)
        features = F.log_softmax(features.t())
        features = features.t()
        return features



class GAN_Value(nn.Module):
    """
    GCN model for value function, the negative number of edges added
    """
    def __init__(self, nin, nhidden, nout, dropout, alpha):
        super(GAN_Value, self).__init__()

        self.gc1 = GraphAttentionConvLayer(nin, nhidden, dropout, alpha)
        self.gc2 = GraphAttentionConvLayer(nhidden, nhidden, dropout, alpha) # second graph conv layer
        self.gc3 = GraphAttentionConvLayer(nhidden, nhidden, dropout, alpha)  # second graph conv layer

        self.dropout = dropout
        self.linear = nn.Linear(nhidden, nout,  bias=False)

    def forward(self, features, adj_matrix):
        # first layer with relu
        #features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc1(features, adj_matrix)
        features = F.relu(features)
        #
        # # second layer with relu
        features = self.gc2(features, adj_matrix)
        features = F.relu(features)
        #
        # #third layer with relu
        features = self.gc3(features, adj_matrix)
        # features = F.relu(features)
        features = features

        return features


class GNN_GAN(nn.Module):
    def __init__(self, nin, nhidden, nout, dropout, alpha):
        super(GNN_GAN, self).__init__()

        self.gc1 = GraphConvolutionLayer_Sparse(nin, nhidden) # first graph conv layer
        self.gc2 = GraphAttentionConvLayer(nhidden, nout, dropout, alpha)
        self.dropout = dropout

    def forward(self, features, adj_matrix):
        # features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc1(features, adj_matrix)
        features = F.relu(features)
        # features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc2(features, adj_matrix)
        features = F.log_softmax(features.t())
        features = features.t()
        return features


class GCN_Sparse_Memory_3(nn.Module):
    """
    GCN model for node selection policy
    """
    def __init__(self, nin, nhidden, nout, dropout):
        super(GCN_Sparse_Memory_3, self).__init__()

        self.gc1 = GraphConvolutionLayer_Sparse_Memory(nin, nhidden) # first graph conv layer
        self.gc2 = GraphConvolutionLayer_Sparse_Memory(nhidden, nhidden)  # first graph conv layer
        # self.gc3 = GraphConvolutionLayer_Sparse_Memory(nhidden, nhidden)  # first graph conv layer
        # self.gc4 = GraphConvolutionLayer_Sparse_Memory(nhidden, nhidden)  # first graph conv layer
        # self.gc5 = GraphConvolutionLayer_Sparse_Memory(nhidden, nout) # second graph conv layer
        # self.dropout = dropout


    def forward(self, features, adj_matrix):
        # first layer with relu
        # features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc1(features, adj_matrix)
        features = F.relu(features)
        features = self.gc2(features, adj_matrix)
        # features = F.relu(features)
        # features = self.gc3(features, adj_matrix)
        # features = F.relu(features)
        # features = self.gc4(features, adj_matrix)
        # features = F.relu(features)

        # second layer with softmax
        # features = self.gc5(features, adj_matrix)
        # features = F.relu(features)
        # features = features.t()

        return features


class GCN_Sparse_Policy(nn.Module):

    def __init__(self, nin, nhidden_gcn, nout_gcn, nhidden_policy, dropout):

        super(GCN_Sparse_Policy, self).__init__()

        self.gcn = GCN_Sparse_Memory_3(nin=nin, nhidden= nhidden_gcn, nout=nout_gcn, dropout=dropout)

        # self.policy1 = nn.Sequential(
        #     nn.Linear(nout_gcn, nhidden_policy),
        #     nn.ReLU(),
        #     nn.Linear(nhidden_policy, 1)
        # )
        self.policy1 = nn.Linear(nout_gcn, 1)


    def forward(self, features, adj_matrix):

        features = self.gcn(features, adj_matrix)
        probs = self.policy1(features).view(-1)
        probs = F.log_softmax(probs)

        return probs, features


class MLP_Value(nn.Module):

    def __init__(self, nout_gcn, nhidden_value):

        super(MLP_Value, self).__init__()

        # self.value = nn.Sequential(
        #     nn.Linear(nout_gcn, nhidden_value),
        #     nn.ReLU(),
        #     nn.Linear(nhidden_value, 1),
        #     nn.ReLU()
        # )

        self.value = nn.Sequential(
            nn.Linear(nout_gcn, 1)
        )

    def forward(self, features):

        v = self.value(features).sum()

        return v


class GCN_Sparse_Policy_Baseline1(nn.Module):
    """
    GCN model for node selection policy for A2C
    """
    def __init__(self, nin, nhidden, nout, dropout):
        super(GCN_Sparse_Policy_Baseline1, self).__init__()

        self.gc1 = MessagePassing_GNN_Layer_Sparse_Memory(nin, nhidden) # first graph conv layer
        self.gc2 = MessagePassing_GNN_Layer_Sparse_Memory(nhidden, nhidden)  # first graph conv layer
        self.gc3 = MessagePassing_GNN_Layer_Sparse_Memory(nhidden, nhidden)
        self.gc4 = MessagePassing_GNN_Layer_Sparse_Memory(nhidden, nhidden)
        self.gc5 = MessagePassing_GNN_Layer_Sparse_Memory(nhidden, nout) # second graph conv layer
        self.dropout = dropout


    def forward(self, features, adj_matrix):
        # first layer with relu
        # features = F.dropout(features, self.dropout, training=self.training)
        features = self.gc1(features, adj_matrix)
        features = F.relu(features)

        features = self.gc2(features, adj_matrix)
        features = F.relu(features)

        features = self.gc3(features, adj_matrix)
        features = F.relu(features)

        features_hidden = self.gc4(features, adj_matrix)
        features_out = F.relu(features_hidden)

        #
        # features = self.gc3(features, adj_matrix)
        # features = F.relu(features)

        # features = self.gc4(features, adj_matrix)
        # features = F.relu(features)


        # output layer with softmax
        features_out = self.gc5(features_out, adj_matrix)
        probs = F.log_softmax(features_out.t())
        probs = probs.t()

        return probs, features_hidden





