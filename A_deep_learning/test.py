import numpy as np
from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')

from fealpy.mesh import UniformMesh1d
from fealpy.sparse import csr_matrix, spdiags


# [0, 1] 区间均匀剖分 10 段，每段长度 0.1
nx = 10
hx = 1/nx
mesh = UniformMesh1d([0, nx], h=hx, origin=0.0)


def laplace_operator(self):
    """
    @brief 组装 Laplace 算子 ∆u 对应的有限差分离散矩阵

    @note 并未处理边界条件
    """
    h = self.h
    cx = 1/(h**2)
    NN = self.number_of_nodes()
    K = np.arange(NN)

    A = spdiags(bm.ones(NN), diags=0, M=NN, N=NN)
    print(type(A))
    val = np.broadcast_to(-cx, (NN-1, ))
    I = K[1:]
    J = K[0:-1]
    A = A + csr_matrix((val, (I, J)), shape=(NN, NN))
    A = A +  csr_matrix((val, (J, I)), shape=(NN, NN))
    return A

A=laplace_operator(mesh)
print(A)