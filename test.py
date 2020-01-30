import numpy as np
import argparse
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

w = []

w = torch.tensor(w)

features = torch.tensor([[0.1], [0.2],[0.3],[0.5]]) # initialize the feature matrix
features = features.view(-1)
features = F.log_softmax(features)
print(features)

# f2 = torch.ones([1293], dtype=torch.float32)
# a = torch.ones([1293,1293], dtype=torch.float32)
# features[:,1] = torch.mm(a, features[:,0].view(-1,1)).view(-1)
#
# features[:,0] = f2
# print(features[:,0])




