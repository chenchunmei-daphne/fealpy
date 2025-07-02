import torch 
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(123)
X = torch.rand(100, 2, dtype=torch.float64)  # 2D 张量
W0 = torch.tensor([[2], [4]], dtype=torch.float64)
b0 = torch.tensor(4, dtype=torch.float64)
y = X @ W0 + b0  
y += torch.normal(0, 0.1, size=y.shape, dtype=torch.float64)  # 添加噪声


# 定义 MLP（手动初始化参数）
class MLP_Hand(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=10, output_dim=1):
        super(MLP_Hand, self).__init__()
        "初始化权重和偏置（手动管理）"
        self.W1 = torch.rand(input_dim, hidden_dim, dtype=torch.float64, requires_grad=False)
        self.b1 = torch.rand(hidden_dim, dtype=torch.float64, requires_grad=False)
        self.W2 = torch.rand(hidden_dim, output_dim, dtype=torch.float64, requires_grad=False)
        self.b2 = torch.rand(output_dim, dtype=torch.float64, requires_grad=False)

    def forward(self, x):
        self.h = x @ self.W1 + self.b1      # 隐藏层
        self.h_relu = torch.relu(self.h)    # ReLU 激活
        y_pred = self.h_relu @ self.W2 + self.b2    # 输出层
        return y_pred

net = MLP_Hand()
mse = nn.MSELoss(reduction="mean")   # 损失函数
epochs = 100
lr = 0.01   # 学习率
l = []

for i in range(epochs):
    y_pred = net(X)  # 前向传播
    loss = ((y_pred - y) ** 2).mean()  # 计算 MSE 损失
    
    # 手动计算梯度（反向传播）
    grad_y_pred = 2 * (y_pred -y) / len(y)  # 计算损失对 y_pred 的梯度

    # 计算损失对 W2, b2 的梯度
    grad_W2 = net.h_relu.T @ grad_y_pred
    grad_b2 = grad_y_pred.sum(0)

    grad_h_relu = grad_y_pred @ net.W2.T  # 计算损失对 h_relu 的梯度
    grad_h = grad_h_relu * (net.h > 0).float()   # 计算损失对 h 的梯度（ReLU 的梯度）
    
    # 计算损失对 W1, b1 的梯度
    grad_W1 = X.T @ grad_h
    grad_b1 = grad_h.sum(0)
    
    # 4. 手动更新参数（梯度下降）
    net.W1 -= grad_W1 * lr
    net.b1 -= grad_b1 * lr
    net.W2 -= grad_W2 * lr
    net.b2 -= grad_b2 * lr
    
    l.append(loss)  # 记录损失

# 测试
X_test = torch.tensor([[0.5, 0.5], [1.0, 2.0]], dtype=torch.float64)
y_pred = net(X_test)
print("Prediction: \n", y_pred)
print("Truth:\n", X_test @ W0 + b0)

plt.plot(l) # 画损失图像
plt.show()



