import torch.nn as nn
import torch

p = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
softmax = nn.Softmax(dim=1)
output = softmax(p)
print(output)
def softmax(x):

    x = torch.exp(x)
    print(x)
    s = torch.sum(x, dim=1)
    print(s)
    x[0,:] /= s[0]
    x[1,:] /= s[1]
    return x 
print(softmax(p))