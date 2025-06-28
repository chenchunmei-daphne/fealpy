import torch
import torch.nn as nn
from fealpy.ml.modules import Solution

net = nn.Sequential(
    nn.Linear(2, 30, dtype=torch.float64), nn.Tanh(),
    nn.Linear(30, 15, dtype=torch.float64))
net = Solution(net)

# 测试网络
# X = torch.normal(0, 1, (100, 2), dtype=torch.float64)
# output = net(X)

X = torch.rand(10, 20, 2, dtype=torch.float64)  # 3D 张量
output = net.last_dim(X)    # 会调用 TensorMapping 的维度处理逻辑
print(output.shape)  # 输出形状应为 (10, 20, 15)
print(net.get_device())  # 调用 TensorMapping 的设备查询方法

import numpy as np
X_np = np.random.rand(100, 2)
output = net.from_numpy(X_np)  # 调用 TensorMapping 的NumPy转换方法

# net = net.to('cuda')     # 触发PyTorch的设备转移
from d2l import torch as d2l
d2l.tran