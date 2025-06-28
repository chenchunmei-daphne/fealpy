import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 定义 PDE 参数
a_x, a_y, gamma = 1.0, 1.0, 0.0  # 泊松方程参数 (∇²f = g)
domain = [-5, 5, -5, 5]           # 定义域 [x_min, x_max, y_min, y_max]
# M, N = 25, 25                      # 网格点数
M, N = 6, 6                     # 网格点数 (增加分辨率)
# 生成网格点
x = torch.linspace(domain[0], domain[1], M)
y = torch.linspace(domain[2], domain[3], N)
X, Y = torch.meshgrid(x, y, indexing='ij')
X_flat = X.reshape(-1, 1)          # 展平为 (M*N, 1)
Y_flat = Y.reshape(-1, 1)
# print("X", X)
# print("Y", Y)
# print("X_flat shape:", X_flat)  # (M*N, 1)
# print("Y_flat shape:", Y_flat)  # (M*N, 1)
input_grid = torch.cat([X_flat, Y_flat], dim=1)  # 输入层数据 (M*N, 2)
print("input_grid shape:", input_grid)  # (M*N, 2)

exit()
# 定义 PDE 的右端项 g(x,y)
def source_function(x, y):
    term1 = torch.exp(-((x + 2)**2 / 2 + y**2 / 2))
    term2 = 0.5 * torch.exp(-((x - 2)**2 / 2 + y**2 / 2))
    return term1 - term2

g_true = source_function(X_flat, Y_flat).reshape(-1, 1)

# 定义 PENN 架构
class PENN(nn.Module):
    def __init__(self, hidden_size=30, num_layers=2):
        super(PENN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(2, hidden_size))  # 输入层 (x,y) -> hidden
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.last_layer = nn.Linear(hidden_size, 1)    # 输出层 -> 系数
        
        # 边界条件函数 (Dirichlet 边界 f=0)
        self.U = lambda x, y: torch.exp(-x**2 - y**2)  # 形状函数（确保边界为0）
        self.B = lambda x, y: torch.zeros_like(x)       # 边界函数（直接设为0）

    def forward(self, x, y):
        # 前向传播：预测系数
        h = torch.cat([x, y], dim=1)
        for layer in self.layers:
            h = torch.nn.functional.logsigmoid(layer(h))  # logsig 激活函数
        coefficients = self.last_layer(h)   # 线性激活
        
        # 最后一层：嵌入 PDE
        f = coefficients * self.U(x, y) + self.B(x, y)
        
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
optimizer = optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.MSELoss()

# 训练循环
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    f_pred, residual = model(X_flat, Y_flat)
    loss = criterion(residual, torch.zeros_like(residual))  # 最小化残差
    loss.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4e}")

# 可视化结果
f_solution = f_pred.reshape(M, N).detach().numpy()
plt.imshow(f_solution, extent=domain, origin='lower', cmap='viridis')
plt.colorbar(label='f(x,y)')
plt.title("PENN Solution of Poisson Equation")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
