
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
from fealpy.old.ml.sampler import BoxBoundarySampler, ISampler

para = Parameter()

wavenum = 1.
k = torch.tensor(wavenum)
NN = 32

# 定义网络层结构
net_1 = nn.Sequential(nn.Linear(2, NN, dtype=torch.float64),
                      nn.Tanh(),
                      nn.Linear(NN, NN//2, dtype=torch.float64),
                      nn.Tanh(),
                      nn.Linear(NN//2, NN//4, dtype=torch.float64),
                      nn.Tanh(),
                      nn.Linear(NN//4, 1, dtype=torch.float64))


net_2= nn.Sequential(nn.Linear(2, NN, dtype=torch.float64),
                     nn.Tanh(),
                     nn.Linear(NN, NN//2, dtype=torch.float64),
                     nn.Tanh(),
                     nn.Linear(NN//2, NN//4, dtype=torch.float64),
                     nn.Tanh(),
                     nn.Linear(NN//4, 1, dtype=torch.float64))


# 网络实例化
s_1 = Solution(net_1)  # 实部
s_2 = Solution(net_2)  # 虚部

# 选择优化器和损失函数
optim_1 = Adam(s_1.parameters(), lr=para.lr, betas=(0.9, 0.99))  # 实部
optim_2 = Adam(s_2.parameters(), lr=para.lr, betas=(0.9, 0.99))    # 虚部

lr_scheduer_1 = StepLR(optimizer=optim_1, step_size=50, gamma=0.9)
lr_scheduer_2 = StepLR(optimizer=optim_2, step_size=50, gamma=0.9)

# 真解
def solution(p:torch.Tensor)->torch.Tensor:
    x = p[..., 0:1]
    y = p[..., 1:2]
    r = torch.sqrt(x**2 + y**2)
    val = torch.zeros(x.shape, dtype=torch.complex128)
    c = torch.complex(torch.cos(k), torch.sin(k)) / torch.complex(torch.special.bessel_j0(k), torch.special.bessel_j1(k))
    val[:] = (torch.cos(k * r) - c * torch.special.bessel_j0(k * r)) / k
    return val

def solution_numpy_real(p:NDArray) -> NDArray:
    '''解的实部'''
    sol = solution(torch.tensor(p))
    real_ = torch.real(sol)
    return real_.detach().numpy()

def solution_numpy_imag(p: NDArray) -> NDArray:
    '''解的虚部'''
    sol = solution(torch.tensor(p))
    imag_ = torch.imag(sol)
    return imag_.detach().numpy()

# 定义 pde
def pde(p: torch.Tensor) -> torch.Tensor:

    u = torch.complex(s_1(p), s_2(p))
    x = p[..., 0:1]
    y = p[..., 1:2]
    r = torch.sqrt(x**2 + y**2)

    f = torch.zeros(x.shape, dtype=torch.complex128)
    f[:] = torch.sin(k * r) / r  # 源项

    u_x_real, u_y_real = gradient(u.real, p, create_graph=True, split=True)
    u_xx_real, _ = gradient(u_x_real, p, create_graph=True, split=True)
    _, u_yy_real = gradient(u_y_real, p, create_graph=True, split=True)

    u_x_imag, u_y_imag = gradient(u.imag, p, create_graph=True, split=True)
    u_xx_imag, _ = gradient(u_x_imag, p, create_graph=True, split=True)
    _, u_yy_imag = gradient(u_y_imag, p, create_graph=True, split=True)

    u_xx = torch.complex(u_xx_real, u_xx_imag)
    u_yy = torch.complex(u_yy_real, u_yy_imag)

    # return u_xx + u_yy + u + f
    return  u_xx + u_yy + k**2 * u + f

# 定义解的梯度
def grad(p: torch.Tensor) -> torch.Tensor:
    x = p[..., 0:1]
    y = p[..., 1:2]
    r = torch.sqrt(x**2 + y**2)

    val = torch.zeros(p.shape, dtype=torch.complex128)
    c = torch.complex(torch.cos(k), torch.sin(k)) / torch.complex(torch.special.bessel_j0(k), torch.special.bessel_j1(k))
    u_r = c * torch.special.bessel_j1(k * r) - torch.sin(k * r)
    val[..., 0:1] = u_r * x / r
    val[..., 1:2] = u_r * y / r
    return val

# 定义边界条件
def bc(p: torch.Tensor) -> torch.Tensor:
    u = torch.complex(s_1(p), s_2(p))  # 数值解
    x = p[..., 0]
    y = p[..., 1]
    n = torch.zeros_like(p)
    n[x > torch.abs(y), 0] = 1.0
    n[y > torch.abs(x), 1] = 1.0
    n[x < -torch.abs(y), 0] = -1.0
    n[y < -torch.abs(x), 1] = -1.0

    grad_u_real = gradient(u.real, p, create_graph=True, split=False)  # 数值解的梯度
    grad_u_imag = gradient(u.imag, p, create_graph=True, split=False)
    grad_u = torch.complex(grad_u_real, grad_u_imag)

    kappa = torch.complex(torch.tensor(0.0), k)
    g = (grad(p) * n).sum(dim=-1, keepdim=True) + kappa * solution(p)

    return (grad_u * n).sum(dim=-1, keepdim=True) + kappa * u -g


# 构建网格和有限元空间
domain = [-0.5, 0.5, -0.5, 0.5,]
mesh = TriangleMesh.from_box(domain, nx=64, ny=64)

error = []
error_real = []
error_imag = []

mse_cost_func = nn.MSELoss(reduction='mean')
# 采样器
samplerpde_1 = ISampler([-0.5, 0.5, -0.5, 0.5], requires_grad=True)
samplerpde_2 = ISampler([-0.5, 0.5, -0.5, 0.5], requires_grad=True)
samplerbc_1 = BoxBoundarySampler([-0.5, -0.5], [0.5, 0.5], requires_grad=True)
samplerbc_2 = BoxBoundarySampler([-0.5, -0.5], [0.5, 0.5], requires_grad=True)
# 超参数
num_of_point_pde = para.npde
num_of_point_bc = para.nbc
lr = para.lr
iteration = para.iter

# 训练过程
for epoch in range(para.iter+1):
    optim_1.zero_grad()
    optim_2.zero_grad()


    spde_1 = samplerpde_1.run(num_of_point_pde)
    sbc_1 = samplerbc_1.run(num_of_point_bc, num_of_point_bc)
    outpde_1 = pde(spde_1)
    outbc_1 = bc(sbc_1)

    spde_2 = samplerpde_2.run(num_of_point_pde)
    sbc_2 = samplerbc_2.run(num_of_point_bc, num_of_point_bc)
    outpde_2 = pde(spde_2)
    outbc_2 = bc(sbc_2)

    outpde_real = torch.real(outpde_1)
    outpde_imag = torch.imag(outpde_2)
    mse_pde_real = mse_cost_func(outpde_real, torch.zeros_like(outpde_real))
    mse_pde_imag = mse_cost_func(outpde_imag, torch.zeros_like(outpde_imag))

    outbc_real = torch.real(outbc_1)
    outbc_imag = torch.imag(outbc_2)
    mse_bc_real = mse_cost_func(outbc_real, torch.zeros_like(outbc_real))
    mse_bc_imag = mse_cost_func(outbc_imag, torch.zeros_like(outbc_imag))

    l = 0.5*(0.5 * mse_pde_real + 0.5 * mse_bc_real) +\
            0.5*(0.5* mse_pde_imag + 0.5 * mse_bc_imag)

    # l_real = loss(npde=para.npde, nbc=para.nbc, pde=pde, bc=bc, domain=domain, flag='real')
    # l_imag = loss(npde=para.npde, nbc=para.nbc, pde=pde, bc=bc, domain=domain, flag='imag')
    # l = 0.5 * (l_real + l_imag)
    l.backward(retain_graph=True)

    optim_1.step()
    lr_scheduer_1.step()
    optim_2.step()
    lr_scheduer_2.step()

    if epoch % 10 == 0:
        error_real_ = s_1.estimate_error(solution_numpy_real, mesh, coordtype='c')
        error_imag_ = s_2.estimate_error(solution_numpy_imag, mesh, coordtype='c')

        error_real.append(error_real_)
        error_imag.append(error_imag_)

        error.append(l.detach().numpy())
        print(f'Epoch:{epoch}, loss={l}')
        print('\n')


bc_ = np.array([1/3, 1/3, 1/3], dtype=np.float64)
ps = torch.tensor(mesh.bc_to_point(bc_), dtype=torch.float64)

u_real = torch.real(solution(ps)).detach().numpy()
u_imag = torch.imag(solution(ps)).detach().numpy()
up_real = s_1(ps).detach().numpy()
up_imag = s_2(ps).detach().numpy()


# 可视化
fig, axes = plt.subplots(2, 2, figsize = (8, 8))
mesh.add_plot(axes[0, 0], cellcolor=u_real, linewidths=0, aspect=1)
mesh.add_plot(axes[0, 1], cellcolor=u_imag, linewidths=0, aspect=1)
mesh.add_plot(axes[1, 0], cellcolor=up_real, linewidths=0, aspect=1)
mesh.add_plot(axes[1, 1], cellcolor=up_imag, linewidths=0, aspect=1)

plot(error_real, error_imag, error)