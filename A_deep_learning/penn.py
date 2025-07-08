import torch
import torch.nn as nn
import torch.optim as optim
from fealpy.backend import bm
bm.set_backend("pytorch")
import matplotlib.pyplot as plt
from fealpy.ml.data_model.exp_exp_data_2d import ExpExpData2D
from fealpy.ml.data_model.step_data_2d import StepData2D 
from fealpy.mesh import UniformMesh

# 定义 PDE 参数
pde = StepData2D()
domain = pde.domain()  # 获取定义域

# 生成网格点
M, N = 25, 25          # 网格点数 
mesh = UniformMesh(domain, [0, M-1, 0, N-1])  # 创建均匀网格
X = mesh.entity('node').float()   # (M*N, 2)
# print(type(X))
g_true = pde.source(X).reshape(-1, 1)   # (M*N, 1)


# 定义 PENN 架构
class PENN(nn.Module):
    def __init__(self, hidden_size=30, num_layers=2, uf: callable=pde.gaussian_shape_function, bf: callable=pde.boundary_scaling_function):
        super(PENN, self).__init__()
        self.layer1 = nn.Linear(2, hidden_size) # 输入层 (x,y) -> hidden1
        self.layer2 = nn.Linear(hidden_size, hidden_size) 
        self.last_layer = nn.Linear(hidden_size, 1)    # 输出层 -> 系数
        
        self.uf = uf     # 形状函数（确保边界为 0）
        self.bf = bf     # 边界缩放函数(确保满足边界条件)

    def forward(self, X):
        h1 = self.layer1(X)
        h2 = nn.functional.logsigmoid(self.layer2(h1))
        coefficients = self.last_layer(h2)   # 线性激活
        
        u = self.uf(X)[..., None]
        b = self.bf(X)[..., None]
        # 最后一层：嵌入 PDE
        f = coefficients * u 
        
        # 计算 PDE 残差 ∇²f - g
        f = f.reshape(M, N)
        dx = (domain[1] - domain[0]) / (M - 1)
        dy = (domain[3] - domain[2]) / (N - 1)
        
        # 二阶中心差分计算 ∇²f
        laplacian_f = (torch.roll(f, -1, 0) + torch.roll(f, 1, 0) - 2 * f) / dx**2 + \
                      (torch.roll(f, -1, 1) + torch.roll(f, 1, 1) - 2 * f) / dy**2
        residual = laplacian_f.reshape(-1, 1) - g_true
        return f.reshape(-1, 1), residual

# 初始化模型和优化器
model = PENN(hidden_size=30, num_layers=2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 定义RMSE损失函数
def rmse_loss(y_res, y=g_true):
    mse = torch.sum(y_res ** 2) / torch.sum(y ** 2)
    rmse = torch.sqrt(mse)
    return rmse

loss_ = []
mse = nn.MSELoss()
# 训练循环
epochs = 4000
for epoch in range(epochs):
    optimizer.zero_grad()
    f_pred, residual = model(X)
    loss = rmse_loss(residual)  # 最小化残差
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        loss_.append(loss.detach())
        # print(f"Epoch {epoch}, Loss: {loss.item():.4e}")
print("last loss: ", loss_[-1])
# 获取最终解
f_solution = f_pred.reshape(M, N).detach().numpy()

# 绘制结果（子图）
plt.figure(figsize=(12, 5))

# 子图1：PENN求解的 f(x,y) 的三维表面图
# ax1 = plt.subplot(1, 2, 1, projection='3d')
# surf = ax1.plot_surface(X[:, 0], Y, f_solution, cmap='viridis', edgecolor='none')
# plt.colorbar(surf, shrink=0.5, aspect=5, label='f(x,y)')
# ax1.set_title("3D PENN Solution of Poisson Equation")
# ax1.set_xlabel("x")
# ax1.set_ylabel("y")
# ax1.set_zlabel("f(x,y)")

# 子图2：训练损失曲线
plt.subplot(1, 1, 1)
plt.plot(loss_, label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training Loss Curve")

plt.grid(True)
plt.tight_layout()  # 调整子图间距
plt.show()
