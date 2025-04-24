from fealpy.mesh import TriangleMesh, TetrahedronMesh
from fealpy.functionspace import LagrangeFESpace
mesh = TriangleMesh.from_box(nx=2, ny=2)
space = LagrangeFESpace(mesh=mesh)
space.number_of_local_dofs()
from fealpy.ml.helmholtz_pinn.pde import HelmholtzData2d as h2d
from fealpy.pde.helmholtz_2d import HelmholtzData2d
from fealpy.backend import backend_manager as bm
bm.set_backend('pytorch')
import numpy as np

p1 = h2d(k=1.0)
p = HelmholtzData2d(k=1.0)
X = bm.tensor([[-0.5, -0.5], 
               [0.5, -0.5], 
               [0., -0.1], 
               [-0.5, -0.2]])
print("X.shape", X.shape)

print("robin:")

pp = bm.tensor([[-0.5, -0.5], [0.5, -0.5], [0.5, -0.], [-0.5, -0.2], [0.5, -0.5], 
               [0.5, -0.], [-0.5, -0.2], [0.5, -0.5], [0.5, -0.], [-0.5, -0.2]])

# x = pp[..., 0]
# y = pp[..., 1]
# n = bm.zeros_like(pp)  # 法向量 n
# n[x > bm.abs(y), 0] = 1.0
# n[y > bm.abs(x), 1] = 1.0
# n[x < -bm.abs(y), 0] = -1.0
# n[y < -bm.abs(x), 1] = -1.0

# print("新的", p1.robin(p=pp,n=n).shape)

# n = np.array(n)
# pp = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, -0.], [-0.5, -0.2], [0.5, -0.5], 
#                [0.5, -0.], [-0.5, -0.2], [0.5, -0.5], [0.5, -0.], [-0.5, -0.2]])
# print("以前的", p.robin(p=pp, n=n).shape)
