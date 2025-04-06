
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from fealpy.ml import Solution

from fealpy.ml.pinn import Parameter, Pinn_loss, plot_error, plot_mesh, Helmholtz2d


para = Parameter()  # 超参数
helmholtz =  Helmholtz2d(para.k, para.domain)  # 方程
loss = Pinn_loss(pde_fun=helmholtz.pde, bc_fun=helmholtz.robin_bc, npde=para.npde, nbc=para.nbc)  # 损失函数，内含采样
mesh = para.mesh  # 有限元网格
samplerpde = para.samplerpde  # 区域内部的采样器
samplerbc =  para.samplerbc   # 区域边界的采样器


# 定义网络层结构
net_real = para.net()  # 实部
net_imag = para.net()  # 虚部

s_1 = Solution(net_real)   # 网络实例化
s_2 = Solution(net_imag)

# 选择优化器
optim_1 = Adam(s_1.parameters(), lr=para.lr, betas=(0.9, 0.99))  # 实部
optim_2 = Adam(s_2.parameters(), lr=para.lr, betas=(0.9, 0.99))    # 虚部
lr_scheduer_1 = StepLR(optimizer=optim_1, step_size=para.step_size, gamma=0.9)  # 学习率调整
lr_scheduer_2 = StepLR(optimizer=optim_2, step_size=para.step_size, gamma=0.9)

# 训练过程
error_sum = []  #误差初始化
error_real = []
error_imag = []

for epoch in range(para.iter+1):
    optim_1.zero_grad()
    optim_2.zero_grad()

    l_real = loss.helmholtz_loss(samplerpde=samplerpde, samplerbc=samplerbc, real_net=s_1, imag_net=s_2, flag='real')
    l_imag = loss.helmholtz_loss(samplerpde=samplerpde, samplerbc=samplerbc, real_net=s_1, imag_net=s_2, flag='imag')

    l = 0.5 * (l_real + l_imag)
    l.backward(retain_graph=True)

    optim_1.step()
    lr_scheduer_1.step()
    optim_2.step()
    lr_scheduer_2.step()

    if epoch % 50 == 0:
        error_real_ = s_1.estimate_error(helmholtz.solution_numpy_real, mesh, coordtype='c')
        error_imag_ = s_2.estimate_error(helmholtz.solution_numpy_imag, mesh, coordtype='c')

        error_real.append(error_real_)
        error_imag.append(error_imag_)
        error_sum.append(l.detach().numpy())

        print(f"Epoch: {epoch}, Loss: {l}")
        print(f"Error_real:{error_real_}, Error_imag:{error_imag_}")
        print('\n')

# 结果展示
fig_mesh = plot_mesh(mesh=mesh, solution=helmholtz.solution, s1=s_1, s2=s_2)
fig_error = plot_error(error_real = error_real, error_imag=error_imag, error=error_sum)
plt.show()
