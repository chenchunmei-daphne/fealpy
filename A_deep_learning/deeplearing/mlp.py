import torch 
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
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
        X = self.relu(self.fc1(X))
        X = self.fc2(X)
        return X

net = MLP()
lr = 0.1
# optimizer = Adam(net.parameters(), lr=lr)
optimizer = SGD(net.parameters(), lr=lr)
scheduler = StepLR(optimizer=optimizer, step_size=50, gamma=0.9)

epochs = 100
l = []
mse = nn.MSELoss(reduction="mean")
for i in range(epochs):
    optimizer.zero_grad()

    y_pred = net(X)
    loss = mse(y_pred, y)
    loss.backward()   # 反向转播
    optimizer.step()  # 调整参数
    # scheduler.step()  # 调整学习率

    l.append(loss.detach())

X_test = torch.tensor([[0.5,0.5], [1, 2]], dtype=torch.float64)

with torch.no_grad():
    y_hat = net(X_test)
    print("Prediction: \n", y_hat)
    print("Truth: \n", X_test @ W0 + b0)
print("last epochs loss: ", l[-1])
plt.plot(l)
plt.show()




