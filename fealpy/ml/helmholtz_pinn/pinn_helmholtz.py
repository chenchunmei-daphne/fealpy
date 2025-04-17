
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from fealpy.mesh import TriangleMesh

from fealpy.ml import Solution
from  fealpy.ml.sampler import ISampler, BoxBoundarySampler

from fealpy.ml.helmholtz_pinn import Parameter, Pinn_loss, plot_error, plot_mesh, Helmholtz2d


para = Parameter(iter=50)  # 超参数
helmholtz =  Helmholtz2d()  # 方程
loss = Pinn_loss(pde_fun=helmholtz.pde_func, bc_fun=helmholtz.robin_func, npde=para.npde, nbc=para.nbc)  # 损失函数，内含采样

domain = helmholtz.domain()    # 网格参数
nx, ny = 64, 64
mesh = TriangleMesh.from_box(domain, nx=nx, ny=ny)

samplerpde = ISampler(domain, requires_grad=True)  # 采样器
samplerbc = BoxBoundarySampler(domain[0::2], domain[1::2], requires_grad=True)



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
loss_sum = []  #误差初始化
loss_real = []
loss_imag = []

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
        loss_real_ = s_1.estimate_error(helmholtz.solution_numpy_real, mesh, coordtype='c')
        loss_imag_ = s_2.estimate_error(helmholtz.solution_numpy_imag, mesh, coordtype='c')

        loss_real.append(loss_real_)
        loss_imag.append(loss_imag_)
        loss_sum.append(l.detach().numpy())

        print(f"Epoch: {epoch}, loss_sum: {l}")
        print(f"loss_real:{loss_real_}, loss_imag:{loss_imag_}")
        print('\n')

# 结果展示
fig_mesh = plot_mesh(mesh=mesh, solution=helmholtz.solution, s1=s_1, s2=s_2)
fig_error = plot_error(loss_real = loss_real, loss_imag=loss_imag, loss_sum=loss_sum)
plt.show()
