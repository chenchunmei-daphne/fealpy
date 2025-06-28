import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import SGD

from fealpy.ml.modules import Solution

torch.manual_seed(123)
X = torch.rand(100, 2, dtype=torch.float64)  # 2D 张量
W0 = torch.tensor([[2], [4]], dtype=torch.float64)
b0 = torch.tensor(4, dtype=torch.float64)
y = X @ W0 + b0  
y += torch.normal(0, 0.1, size=y.shape, dtype=torch.float64)  # 添加噪声

input_dim=2 
hidden_dim=10
output_dim=1
model = nn.Sequential(
        nn.Linear(in_features=input_dim, out_features=hidden_dim, dtype=torch.float64),
        nn.ReLU(),
        nn.Linear(in_features=hidden_dim, out_features=output_dim, dtype=torch.float64))
# 定义多层感知机模型
net = Solution(model)

lr = 0.1
epochs = 100
l = []

# optimizer = Adam(net.parameters(), lr=lr)
optimizer = SGD(net.parameters(), lr=lr)  ## 优化器
mse = nn.MSELoss(reduction="mean")  # 均方误差损失函数

for i in range(epochs):
    optimizer.zero_grad()

    y_pred = net(X)
    loss = mse(y_pred, y)
    loss.backward()   # 反向转播
    optimizer.step()  # 调整参数

    l.append(loss.detach())

X_test = torch.tensor([[0.5,0.5], [1, 2]], dtype=torch.float64)

with torch.no_grad():
    y_hat = net(X_test)
    print("Prediction: \n", y_hat)
    print("Truth: \n", X_test @ W0 + b0)
print("last epochs loss: ", l[-1])
plt.plot(l)
plt.show()





