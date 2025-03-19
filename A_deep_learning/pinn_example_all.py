
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from numpy.typing import NDArray

from fealpy.mesh import TriangleMesh
from fealpy.ml import gradient
from fealpy.ml import Solution
from fealpy.ml.pinn import Parameter, loss, plot

para = Parameter()

wavenum = 1.
k = torch.tensor(wavenum)
NN = 32

# 定义网络层结构
'''
net_1 = nn.Sequential(nn.Linear(2, NN, dtype=torch.float64), nn.Tanh(),
                      nn.Linear(NN, NN//2, dtype=torch.float64), nn.Tanh(),
                      nn.Linear(NN//2, NN//4, dtype=torch.float64), nn.Tanh(),
                      nn.Linear(NN//4, 1, dtype=torch.float64), nn.Tanh() )
                      '''

net = nn.Sequential(nn.Linear(2, NN, dtype=torch.complex128), nn.Tanh(),
                      nn.Linear(NN, NN//2, dtype=torch.complex128), nn.Tanh(),
                      nn.Linear(NN//2, NN//4, dtype=torch.complex128), nn.Tanh(),
                      nn.Linear(NN//4, 1, dtype=torch.complex128), nn.Tanh() )


# 网络实例化
s = Solution(net)

# 选择优化器和损失函数
optim = Adam(s.parameters(), lr=para.lr, betas=(0.9, 0.99))
lr_scheduer = StepLR(optimizer=optim, step_size=50, gamma=0.9)

# 真解
def solution(p:torch.Tensor):
    x = p[..., 0:1]
    y = p[..., 1:2]
    r = torch.sqrt(x**2 + y**2)
    c = torch.complex(torch.cos(k), torch.sin(k)) / torch.complex(torch.special.bessel_j0(k), torch.special.bessel_j1(k))
    val = torch.zeros(x.shape, dtype=torch.complex128)
    val[:] = (torch.cos(k * r) - c * torch.special.bessel_j0(k * r)) / k
    return val

# 定义 pde
def pde(p):
    u = s(p)
    x = p[..., 0:1]
    y = p[..., 1:2]
    r = torch.sqrt(x**2 + y**2)

    f = torch.zeros(x.shape, dtype=torch.complex128)
    f[:] = torch.sin(k * r) / r  # 源项

    u_x, u_y = gradient(u, p, create_graph=True, split=True)
    u_xx, _ = gradient(u_x, p, create_graph=True, split=True)
    _, u_yy = gradient(u_y, p, create_graph=True, split=True)

    return  u_xx + u_yy + k*u + f

# 定义解的梯度
def grad(p):
    x = p[..., 0:1]
    y = p[..., 1:2]
    r = torch.sqrt(x ** 2 + y ** 2)

    val = torch.zeros(x.shape, dtype=torch.complex128)
    c = torch.complex(torch.cos(k), torch.sin(k)) / torch.complex(torch.special.bessel_j0(k), torch.special.bessel_j1(k))
    u_r = c * torch.special.bessel_j1(k * r) - torch.sin(k * r)
    val[..., 0:1] = u_r * x / r
    val[..., 1:2] = u_r * y / r
    return val

# 定义边界条件
def bc(p):
    u = s(p)  # 数值解
    x = torch.real(p[..., 0])  # 只取实部
    y = torch.real(p[..., 1])
    n = torch.zeros_like(p)
    n[x > torch.abs(y), 0] = 1.
    n[x < torch.abs(y), 0] = -1.
    n[y > torch.abs(x), 1] = 1.
    n[y < torch.abs(y), 1] = -1.

    grad_u = gradient(u, p, create_graph=True)  # 数值解的梯度

    kappa = torch.complex(torch.tensor(0.0), k)
    g = (grad(p) * n).sum(-1, keepdim=True) + kappa * solution(p)

    return (grad_u * n).sum(-1, keepdim=True) + kappa * u -g


# 构建网格和有限元空间
domain = [-.5, 0.5, -.5, 0.5,]
mesh = TriangleMesh.from_box(domain, nx=64, ny=64)

error = []
# 训练过程
for epoch in range(para.iter+1):
    optim.zero_grad()

    l = loss(npde=para.npde, nbc=para.nbc, pde=pde, bc=bc, domain=domain)
    l.backward(retain_graph=True)

    optim.step()
    lr_scheduer.step()

    if epoch % 10 == 0:
        error.append(l.detach.numpy())
        print(f'Epoch:{epoch}, loss={l}')
        print('\n')


bc_ = np.array([1/3, 1/3, 1/3], dtype=np.float64)
ps = torch.tensor(mesh.bc_to_point(bc_), dtype=torch.float64)

u_real = torch.tensor(solution(ps)).detach().numpy()
up_real = s(ps).detach().numpy()


# 可视化
fig, axes = plt.subplots(1, 2)
mesh.add_plot(axes[0, 0], cellcolor=u_real, linewidths=0, aspect=1)
mesh.add_plot(axes[0, 1], cellcolor=up_real, linewidths=0, aspect=1)

plot(error)

