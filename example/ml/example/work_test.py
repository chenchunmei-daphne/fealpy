import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from fealpy.model import PDEModelManager
from fealpy.backend import backend_manager as bm
from fealpy.ml.sampler import ISampler, BoxBoundarySampler
bm.set_backend('pytorch')

class PINNNet(nn.Module):
    def __init__(self, layer_sizes):
        super(PINNNet, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1], dtype=torch.float64))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1], dtype=torch.float64))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
        
def compute_laplace(u, p):
    grad_u = torch.autograd.grad(
        outputs=u, 
        inputs=p, 
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]
    
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    u_xx = torch.autograd.grad(
        outputs=u_x, 
        inputs=p, 
        grad_outputs=torch.ones_like(u_x),
        create_graph=True 
    )[0][:, 0:1]
    u_yy = torch.autograd.grad(
        outputs=u_y, 
        inputs=p, 
        grad_outputs=torch.ones_like(u_y),
        create_graph=True
    )[0][:, 1:2]
    
    return u_xx + u_yy

pde = PDEModelManager("diffusion_reaction").get_example(2)
domain = (-3, 3, -3, 3)

layer_sizes = [2, 40, 40, 1] 
net = PINNNet(layer_sizes)

in_sampler = ISampler(ranges=domain, mode='random', dtype=torch.float64, requires_grad=True)
bc_sampler = BoxBoundarySampler(domain, mode='random', dtype=torch.float64, requires_grad=True)

def loss_function(net, pde, in_sampler, bc_sampler):
    in_point = in_sampler.run(2500)
    bc_point = bc_sampler.run(400)

    u = net(in_point)
    laplace_u = compute_laplace(u, in_point) 
    f = pde.source(in_point)
    c = 0.2
    pde_residual = -laplace_u - c * u - f
    loss_pde = torch.mean(pde_residual**2) 

    u_bc = net(bc_point)
    g_bc = pde.dirichlet(bc_point)
    bc_residual = u_bc - g_bc
    loss_bc = torch.mean(bc_residual**2) 

    total_loss = loss_pde + 10 * loss_bc
    
    return total_loss, loss_pde, loss_bc

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

epochs = 2000
for i in range(epochs + 1):
    optimizer.zero_grad()
    
    loss, loss_pde, loss_bc = loss_function(net, pde, in_sampler, bc_sampler)
    
    loss.backward()
    
    optimizer.step()
    
    if i % 100 == 0:
        print(f"Epoch: {i}, Total Loss: {loss.item():.4e}, "
              f"PDE Loss: {loss_pde.item():.4e}, BC Loss: {loss_bc.item():.4e}")

print("\n训练完成，正在可视化结果...")

nx = 101
ny = 101
x = torch.linspace(domain[0], domain[1], nx, dtype=torch.float64)
y = torch.linspace(domain[2], domain[3], ny, dtype=torch.float64)
X, Y = torch.meshgrid(x, y, indexing='ij')
plot_point = torch.stack([X.flatten(), Y.flatten()], axis=1)

with torch.no_grad():
    u_pred = net(plot_point)

U_pred = u_pred.reshape(nx, ny)

X_np = X.numpy()
Y_np = Y.numpy()
U_pred_np = U_pred.numpy()

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X_np, Y_np, U_pred_np, cmap='viridis')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('u(x, y)')
ax1.set_title('Predicted Solution (3D Surface)')
ax1.view_init(elev=30, azim=-60)

ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contourf(X_np, Y_np, U_pred_np, 50, cmap='viridis')
fig.colorbar(contour)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Predicted Solution (Contour Plot)')
ax2.set_aspect('equal', 'box')

plt.tight_layout()
plt.show()