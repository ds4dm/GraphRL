import numpy as np
import argparse
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

a = []

a.append(1)
a.append(2)


print(a[0])


# w = [[1,2], [1,2]]
#
# w = torch.tensor(w)
# a= w.sum()
#
# print(a)

# features = torch.tensor([float('nan')]) # initialize the feature matrix
# if torch.isnan(features):
#
#     print(features)

# f2 = torch.ones([1293], dtype=torch.float32)
# a = torch.ones([1293,1293], dtype=torch.float32)
# features[:,1] = torch.mm(a, features[:,0].view(-1,1)).view(-1)
#
# features[:,0] = f2
# print(features[:,0])

lr = 2
print(len(lr))



