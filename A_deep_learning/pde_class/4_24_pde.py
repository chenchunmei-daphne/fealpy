from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')
from fealpy.typing import TensorLike
from typing import Sequence


class SinPDEData1D:
    """
    1D Elliptic problem:

        -u''(x)-π²*u(x) = f(x),  x in (0, 1)
        u(0) = u(1) = 0

    with the exact solution:

        u(x) = sin(πx)

    The corresponding source term is:

        f(x) = 0

    Dirichlet boundary conditions are applied at both ends of the interval.
    """

    def geo_dimension(self) -> int: 
        return 1

    def domain(self) -> Sequence[float]:
        return [0.0, 1.0]

    def solution(self, p: TensorLike) -> TensorLike:
        x = p
        pi = bm.pi
        val = bm.sin(pi * x)
        return val

    def gradient(self, p: TensorLike) -> TensorLike:
        x = p
        pi = bm.pi
        val = pi * bm.cos(pi * x)
        return val

    def source(self, p: TensorLike) -> TensorLike:
        val = self.solution(p) 
        return val
    
    def dirichlet(self, p: TensorLike) -> TensorLike:
        return self.solution(p).reshape(-1)
    

class SinPDEData2D:
    """
    2D Elliptic problem:

        -Δu(x, y)-2π²*u(x) = f(x, y),  (x, y) ∈ (0, 1) x (0, 1)
         u(x, y) = 0,         on ∂Ω

    with the exact solution:

        u(x, y) = sin(πx)·sin(πy)

    The corresponding source term is:

        f(x, y) = 0

    Homogeneous Dirichlet boundary conditions are applied on all edges.
    """
    
    def geo_dimension(self) -> int: 
        return 2

    def domain(self) -> Sequence[float]:
        return [0.0, 1.0, 0.0, 1.0]

    def solution(self, p: TensorLike) -> TensorLike:
        x = p[:, 0]
        y = p[:, 1]
        pi = bm.pi
        val = bm.sin(pi * x) * bm.sin(pi * y)
        return val

    def gradient(self, p: TensorLike) -> TensorLike:
        x = p[:, 0]
        y = p[:, 1]
        pi = bm.pi
        du_dx = pi * bm.cos(pi * x) * bm.sin(pi * y)
        du_dy = pi * bm.sin(pi * x) * bm.cos(pi * y)
        return bm.stack([du_dx, du_dy], axis=-1)

    def source(self, p: TensorLike) -> TensorLike:
        val = self.solution(p) * 1
        return val

    def dirichlet(self, p: TensorLike) -> TensorLike:
        return self.solution(p).reshape(-1)
    

class SinPDEData3D:
    """
    3D Elliptic problem:

        -Δu(x, y, z)-3π²*u(x, z) = f(x, y, z),  (x, y, z) ∈ (0, 1) x (0, 1) x (0, 1)
        u(x, y, z) = 0,         on ∂Ω

    with the exact solution:

        u(x) = sin(πx)·sin(πy)·sin(πz)

    The corresponding source term is:

        f(x) = 0

    Dirichlet boundary conditions are applied at both ends of the interval.
    """

    def geo_dimension(self) -> int: 
        return 3

    def domain(self) -> Sequence[float]:
        return [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

    def solution(self, p: TensorLike) -> TensorLike:
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        pi = bm.pi
        val = bm.sin(pi * x) * bm.sin(pi * y) * bm.sin(pi * z)
        return val

    def gradient(self, p: TensorLike) -> TensorLike:
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]
        pi = bm.pi
        du_dx = pi * bm.cos(pi * x) * bm.sin(pi * y) * bm.sin(pi * z)
        du_dy = pi * bm.sin(pi * x) * bm.cos(pi * y) * bm.sin(pi * z)
        du_dz = pi * bm.sin(pi * x) * bm.sin(pi * y) * bm.cos(pi * z)
        return bm.stack([du_dx, du_dy, du_dz], axis=-1)
    
    def source(self, p: TensorLike) -> TensorLike:
        val = bm.zeros_like(p)
        return val

    def dirichlet(self, p: TensorLike) -> TensorLike:
        return self.solution(p).reshape(-1)
    

import math
from fealpy.mesh import UniformMesh
from fealpy.sparse import csr_matrix, SparseTensor
from fealpy.backend import backend_manager as bm

class LaplaceOperator():
    """
    Laplace operator in 1D, 2D, and 3D.
    """
    def __init__(self, mesh: UniformMesh):
        self.mesh = mesh

    def assembly(self) -> SparseTensor:
        """
        Assemble the global sparse matrix representing the Laplace operator.
        """
        mesh = self.mesh
        ftype = mesh.ftype
        GD = mesh.geo_dimension()

        h = mesh.h
        c = 1.0 / (h ** 2)

        NN = mesh.number_of_nodes()
        K = mesh.linear_index_map('node')

        # 对角元的元素
        diag_value = bm.full(NN, 2 * c.sum(), dtype=ftype)
        I = K.flat
        J = K.flat
        A = csr_matrix((diag_value, (I, J)), shape=(NN, NN))

        # 非对角元的元素
        full_slice = (slice(None),) * GD
        for i in range(GD):
            # 第 i 轴非对角元素缺少的个数 = 其余两个轴的节点数的乘积
            n_shift = math.prod(
                count for dim_idx, count in enumerate(K.shape) if dim_idx != i)
            
            off_diag_value = bm.full(NN - n_shift, -c[i], dtype=ftype)

            # 计算非对角元的行列索引
            s1 = full_slice[:i] + (slice(1, None),) + full_slice[i+1:]
            s2 = full_slice[:i] + (slice(None, -1),) + full_slice[i+1:]
            I = K[s1].flat
            J = K[s2].flat
            
            # 添加非对角元素
            A += csr_matrix((off_diag_value, (I, J)), shape=(NN, NN))
            A += csr_matrix((off_diag_value, (J, I)), shape=(NN, NN))
        return A


class MassOperator():
    """
    Laplace operator in 1D, 2D, and 3D.
    """
    def __init__(self, mesh: UniformMesh, cofe: float):
        self.mesh = mesh
        self.cofe = cofe

    def assembly(self) -> SparseTensor:
        """
        Assemble the global sparse matrix representing the Mass operator.
        """
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        K = mesh.linear_index_map('node')
        diag = bm.full(NN, self.cofe, dtype=mesh.ftype)

        A = csr_matrix((diag, (K.flat, K.flat)), shape=(NN, NN))
        return A
    
from fealpy.sparse import spdiags
# class DirichletBc():
#     def __init__(self, mesh, pde):
#         self.mesh = mesh
#         self.pde = pde

#     def apply(self, A, f):
#         # 处理边界条件
#         NN = self.mesh.number_of_nodes()
#         node = self.mesh.entity('node')
#         bdflag = self.mesh.boundary_node_flag()

#         uh = bm.zeros(NN, dtype=bm.float64)
#         uh[bdflag] = pde.dirichlet(node[bdflag])
#         f = f - A @ uh
#         f[bdflag] = uh[bdflag]

#         bdIdx = bm.zeros(NN, dtype=bm.float64)
#         bdIdx[bdflag] = 1

#         D0 = spdiags(1-bdIdx, 0, NN, NN, format='scr')
#         D1 = spdiags(bdIdx, 0, NN, NN, format='scr')
#         A = D0 @ A @ D0 + D1
#         return A, f


class DirichletBC():

    def __init__(self, mesh, gd, threshold=None):
        self.mesh = mesh
        self.gd = gd
        self.threshold = threshold

    def apply(self, A, f, uh=None):
        """
        """
        if uh is None:
            uh = bm.zeros(A.shape[0], **A.values_context())

        node = self.mesh.entity('node')
        bdFlag = self.mesh.boundary_node_flag()
        uh = bm.set_at(uh, bdFlag, self.gd(node[bdFlag]))
        
        f = f - A @ uh 
        f = bm.set_at(f, bdFlag, uh[bdFlag])

        NN = self.mesh.number_of_nodes()
        K = self.mesh.linear_index_map('node')
        diag_value = bm.full(NN, 1-bdFlag)
        D0 = csr_matrix((diag_value, (K.flat, K.flat)), shape=(NN, NN))
        D1 = csr_matrix((diag_value, (K.flat, K.flat)), shape=(NN, NN))

        A = D0@A@D0 + D1
        return A, f

from fealpy.mesh import UniformMesh2d, UniformMesh1d, UniformMesh3d
# mesh = UniformMesh1d((0, 5), h=0.2, origin=0)
mesh = UniformMesh2d((0, 5, 0, 5), h=(0.2, 0.2), origin=(0, 0))
# mesh = UniformMesh3d((0, 5, 0, 5, 0, 5), h=(0.2, 0.2, 0.2), origin=(0, 0, 0))
pde = SinPDEData2D()
l = LaplaceOperator(mesh)
A = l.assembly()
# print(A.to_dense())
# M = MassOperator(mesh, -bm.pi**2)
# M = M.assembly()
# print(M.to_dense())

# A += M

# 源项

node = mesh.entity("node")
f = pde.source(node)  

# 处理边界条件
bc = DirichletBC(mesh, pde.dirichlet)
A, f = bc.apply(A, f)
print(A.to_dense())
print(f)



