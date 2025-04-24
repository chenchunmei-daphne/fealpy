from fealpy.mesh import TriangleMesh, TetrahedronMesh
from fealpy.functionspace import LagrangeFESpace
mesh = TriangleMesh.from_box(nx=2, ny=2)
space = LagrangeFESpace(mesh=mesh)

# from fealpy.ml.helmholtz_pinn.pde import HelmholtzData2d as h2d
# from fealpy.pde.helmholtz_2d import HelmholtzData2d
# from fealpy.backend import backend_manager as bm
# bm.set_backend('pytorch')
# import numpy as np
# # p1 = h2d(k=1.0)
# p1 = HelmholtzData2d(k=1.0)
# X = bm.tensor([[-0.5, -0.5], 
#                [0.5, -0.5], 
#                [0., -0.1], 
#                [-0.5, -0.2]])
# print("X.shape", X.shape)
# print('--------------------------')
# solution = p1.solution(X)
# print("solution.shape", solution.shape)
# print('--------------------------')
# source = p1.source(X)
# print("source.shape", source.shape)
# print('--------------------------')
# gradient = p1.gradient(X)
# print("gradient.shape", gradient.shape) 
node = mesh.entity('node')
print("node.shape", node.shape)








# print('--------------------------')
# re = p1.solution_numpy_real(X)
# print("re.shape", re.shape)
# print('--------------------------')
# im = p1.solution_numpy_imag(X)      
# print("im.shape", im.shape)
# print('--------------------------')

# pp = bm.tensor([[-0.5, -0.5], [0.5, -0.5], [0.5, -0.], [-0.5, -0.2]])
# x = pp[..., 0]
# y = pp[..., 1]
# n = bm.zeros_like(pp)  # 法向量 n
# n[x > bm.abs(y), 0] = 1.0
# n[y > bm.abs(x), 1] = 1.0
# n[x < -bm.abs(y), 0] = -1.0
# n[y < -bm.abs(x), 1] = -1.0
# bc = p1.robin(p=pp, n=n)
# print("bc.shape", bc.shape)




# print("domain:")
# print(p1.domain())
# print(p.domain())

# print("solution:")
# print(p1.solution(bm.tensor([[0.1, 0.], [-0.2, 0.3]])))
# print(p.solution(np.array([[0.1, 0.], [-0.2, 0.3]])))

# print("gradient:")
# print(p1.gradient(bm.tensor([[0.1, 0.], [0.2, 0.3]])))
# print(p.gradient(np.array([[0.1, 0.], [0.2, 0.3]])))

# print("source:")
# print(p1.source(bm.tensor([[0.1, 0.1], [-0.2, 0.3]])))
# print(p.source(np.array([[0.1, 0.1], [-0.2, 0.3]])))

# print("robin:")
# pp = bm.tensor([[-0.5, -0.5], [0.5, -0.5], [0.5, -0.], [-0.5, -0.2]])
# x = pp[..., 0]
# y = pp[..., 1]
# n = bm.zeros_like(pp)  # 法向量 n
# n[x > bm.abs(y), 0] = 1.0
# n[y > bm.abs(x), 1] = 1.0
# n[x < -bm.abs(y), 0] = -1.0
# n[y < -bm.abs(x), 1] = -1.0
# # print('w', n)
# print(p1.robin(p=pp,n=n))
# n = np.array(n)
# # print('yy', n)
# pp = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, -0.], [-0.5, -0.2]])
# print(p.robin(p=pp, n=n))
# print(pp.shape)

# bm.set_backend('numpy')
# p1 = HelmholtzData2d(k=1.0)
# X = bm.tensor([[-0.5, -0.5], [0.5, -0.5], [0.5, -0.], [-0.5, -0.2]])
# x = X[..., 0]
# y = X[..., 1]
# n = bm.zeros_like(X)  # 法向量 n
# n[x > bm.abs(y), 0] = 1.0
# n[y > bm.abs(x), 1] = 1.0
# n[x < -bm.abs(y), 0] = -1.0
# n[y < -bm.abs(x), 1] = -1.0
# kappa =  1.0 * 1j
# g = (p1.gradient(X) * n).sum(dim=-1, keepdim=True) + kappa * p1.solution(X)
# print('__g.shape', g.shape)

# r = p1.robin(p=X, n=n)
# print("r.shape", r.shape)
# print(g)
