import torch 
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt

torch.manual_seed(123)
X = torch.rand(100, 2, dtype=torch.float64)  # 2D 张量
W0 = torch.tensor([[2], [4]], dtype=torch.float64)
b0 = torch.tensor(4, dtype=torch.float64)
y = X @ W0 + b0  
y += torch.normal(0, 0.1, size=y.shape, dtype=torch.float64)  # 添加噪声

# 定义多层感知机模型
class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=10, output_dim=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True, dtype=torch.float64)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=True, dtype=torch.float64)
        self.relu = nn.ReLU()
        
    def forward(self, X):
        h_relu = self.relu(self.fc1(X))
        y_pred = self.fc2(h_relu)
        return y_pred

net = MLP()
mse = nn.MSELoss(reduction="mean")   # 损失函数
epochs = 100
lr = 0.01   # 学习率
optimizer = Adam(net.parameters(), lr=lr)  # 优化器
l = []

for i in range(epochs):
    optimizer.zero_grad()

    y_pred = net(X)
    loss = mse(y_pred, y)
    loss.backward()   # 反向转播
    optimizer.step()  # 调整参数

    l.append(loss.detach())

# 测试
X_test = torch.tensor([[0.5, 0.5], [1.0, 2.0]], dtype=torch.float64)
y_pred = net(X_test)
print("Prediction: \n", y_pred)
print("Truth:\n", X_test @ W0 + b0)

plt.plot(l) # 画损失图像
plt.show()




