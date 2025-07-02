import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from fealpy.backend import backend_manager as bm
from fealpy.ml.data_model.step_data_2d import StepData2D

pde = StepData2D()
domain = pde.domain()   
print("Domain:", domain)

p = bm.tensor([[-1, 0], [-1.5, 1], [-4, -4], [3, -4], [4, 3], [4, 4]], dtype=bm.float64)
s = pde.source(p)
print("Source term:", s.shape, s)
d = pde.dirichlet(p[2:, :])
print("Dirichlet boundary condition:", d.shape, d)
is_boundary = pde.is_dirichlet_boundary(p)
print("Is Dirichlet boundary:", is_boundary.shape, is_boundary)