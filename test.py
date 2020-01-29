import numpy as np
import argparse
import torch
from torch.distributions import Categorical

w = []

w = torch.tensor(w)

features = torch.zeros([1293, 3], dtype=torch.float32) # initialize the feature matrix

f2 = torch.ones([1293], dtype=torch.float32)
a = torch.ones([1293,1293], dtype=torch.float32)
features[:,1] = torch.mm(a, features[:,0].view(-1,1)).view(-1)

features[:,0] = f2
print(features[:,0])




