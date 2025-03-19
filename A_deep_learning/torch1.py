import torch
a = torch.tensor([1, 2, 3])
b = torch.zeros((2, 3))
c = torch.ones((3, 3))
d = torch.rand((2, 2))
e = torch.arange(0, 10, 2)
f = torch.linspace(0, 1, steps=5)
print(a,b,c)