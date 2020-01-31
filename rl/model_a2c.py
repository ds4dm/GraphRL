import torch
import torch.nn as nn
import numpy as np
import time

from utils import utils
from torch.distributions import Categorical

class Model_A2C(nn.Module):

    def __init__(self,
                 actor,
                 use_critic = True,
                 use_cuda = False,
                 critic = None):
        super(Model_A2C, self).__init__()

        self.actor = actor
        self.use_critic = use_critic
        self.use_cuda = use_cuda
        self.actions = []
        self.rewards = []
        self.saved_actions = []

        if self.use_critic:
            self.critic = critic
        if self.use_cuda:
            self.actor = self.actor.cuda()
            if self.use_critic:
                self.critic = self.critic.cuda()

    def step_solver(self, sample_action):
        """Get reward and new state from the solver given action

        Args:
            sample_action: action of (batch) sample

        Returns: reward

        """
        batch_size = sample_action[0].size(0)


    def forward(self, inputs):
        """

        Args:
            inputs: graph object

        Returns:

        """

        adj_M = torch.FloatTensor(inputs.M)  # adj matrix of input graph


        features = torch.ones([inputs.n,1], dtype=torch.float32)


        # features = torch.zeros([inputs.n, 3], dtype=torch.float32) # initialize the feature matrix
        # features[:, 0] = torch.ones([inputs.n], dtype=torch.float32)
        # features[:, 1] = torch.mm(adj_M, features[:,0].view(-1,1)).view(-1)
        # features[:, 2] = inputs.n - features[:,1]



        if self.use_cuda:
            adj_M = adj_M.cuda()
            features = features.cuda()

        probs = self.actor(features, adj_M) # call actor to get a selection distribution
        probs = probs.view(-1)

        m = Categorical(logits=probs)
        node_selected = m.sample()
        # node_selected = probs.multinomial()  # choose the node given by GCN (add .squeeze(1) for batch training)
        log_prob = m.log_prob(node_selected)

        if self.use_critic: # call critic to compute the value for current state
            critic_current = self.critic(features, adj_M).sum()

        r = inputs.eliminate_node(node_selected, reduce=True) # reduce the graph and return the nb of edges added

        adj_M = torch.FloatTensor(inputs.M)  # adj matrix of reduced graph

        features = torch.ones([inputs.n, 1], dtype=torch.float32)

        # features = torch.zeros([inputs.n, 3], dtype=torch.float32)  # initialize the feature matrix
        # features[:, 0] = torch.ones([inputs.n], dtype=torch.float32)
        # features[:, 1] = torch.mm(adj_M, features[:, 0].view(-1, 1)).view(-1)
        # features[:, 2] = inputs.n - features[:, 1]
        
        if self.use_cuda:
            adj_M = adj_M.cuda()
            features = features.cuda()
        if self.use_critic: # call critic to compute the value for current state
            critic_next = self.critic(features, adj_M).sum()

        return node_selected, log_prob, r, critic_current, critic_next, inputs



class Model_A2C_Sparse(nn.Module):

    def __init__(self,
                 actor,
                 epsilon=0,
                 use_critic=True,
                 use_cuda=False,
                 critic=None):
        super(Model_A2C_Sparse, self).__init__()

        self.actor = actor
        self.use_critic = use_critic
        self.use_cuda = use_cuda
        self.epsilon = torch.tensor(epsilon, dtype=torch.float)
        self.actions = []
        self.rewards = []
        self.saved_actions = []

        if self.use_critic:
            self.critic = critic
        if self.use_cuda:
            self.actor = self.actor.cuda()
            if self.use_critic:
                self.critic = self.critic.cuda()

    def step_solver(self, sample_action):
        """Get reward and new state from the solver given action

        Args:
            sample_action: action of (batch) sample

        Returns: reward

        """
        batch_size = sample_action[0].size(0)

    def forward(self, inputs):
        """

        Args:
            inputs: graph object

        Returns:

        """

        adj_M = torch.FloatTensor(inputs.M)  # adj matrix of input graph
        adj_M = utils.to_sparse(adj_M)  # convert to coo sparse tensor

        features = torch.ones([inputs.n, 1], dtype=torch.float32)
        # features = torch.zeros([inputs.n, 3], dtype=torch.float32)  # initialize the feature matrix
        # features[:, 0] = torch.ones([inputs.n], dtype=torch.float32)
        # features[:, 1] = torch.spmm(adj_M, features[:, 0].view(-1, 1)).view(-1)
        # features[:, 2] = inputs.n - features[:, 1]

        random_choice = torch.ones(inputs.n)
        epsilon = self.epsilon

        if self.use_cuda:
            adj_M = adj_M.cuda()
            features = features.cuda()
            epsilon = epsilon.cuda()
            random_choice = random_choice.cuda()

        probs = self.actor(features, adj_M)  # call actor to get a selection distribution
        probs = probs.view(-1)

        # node_selected = torch.argmax(probs)
        # log_prob = -torch.log(probs[node_selected])# we are doing min, so use - value

        # probs = torch.exp(probs)
        m = Categorical(logits=probs) #
        # print('sum of probs: {}'.format(probs.exp().sum()))

        # node_selected = m_rand.sample()
        if torch.bernoulli(epsilon)==1 or torch.isnan(probs.exp().sum()):
            print('sum of probs: {}'.format(probs.exp().sum()))
            m_rand = Categorical(random_choice)
            node_selected = m_rand.sample()
        else:
            node_selected = m.sample()



        # node_selected = probs.multinomial()  # choose the node given by GCN (add .squeeze(1) for batch training)
        log_prob = m.log_prob(node_selected)

        if self.use_critic:  # call critic to compute the value for current state
            critic_current = self.critic(features, adj_M).sum()
        else:
            critic_current = 0


        # if node_selected < 0 or node_selected >= inputs.n:
        #     print(node_selected)
        r = inputs.eliminate_node(node_selected, reduce=True)  # reduce the graph and return the nb of edges added

        # call critic to compute the value for current state
        if self.use_critic:

            adj_M = torch.FloatTensor(inputs.M)  # adj matrix of reduced graph
            adj_M = utils.to_sparse(adj_M)  # convert to coo sparse tensor

            features = torch.ones([inputs.n, 1], dtype=torch.float32)
            # features = torch.zeros([inputs.n, 3], dtype=torch.float32)  # initialize the feature matrix
            # features[:, 0] = torch.ones([inputs.n], dtype=torch.float32)
            # features[:, 1] = torch.mm(adj_M, features[:, 0].view(-1, 1)).view(-1)
            # features[:, 2] = inputs.n - features[:, 1]

            if self.use_cuda:
                adj_M = adj_M.cuda()
                features = features.cuda()

            critic_next = self.critic(features, adj_M).sum()
        else:
            critic_next = 0

        return node_selected, log_prob, r, critic_current, critic_next, inputs

















