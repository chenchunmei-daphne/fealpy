import numpy as np
from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')

from fealpy.mesh import UniformMesh1d
from fealpy.sparse import csr_matrix, spdiags
from scipy.special import jv


def solution( p):
    print('solution___')
    k = 1
    x = p[..., 0]
    y = p[..., 1]
    r = np.sqrt(x**2 + y**2)

    val = np.zeros(x.shape, dtype=np.complex128)
    val[:] = np.cos(k*r)/k
    c = complex(np.cos(k), np.sin(k))/complex(jv(0, k), jv(1, k))/k
    val -= c*jv(0, k*r)


    print('c=', c, flush=True)  # 添加 flush=True
    print('jv(0, k*r)', jv(0, k*r), flush=True)
    print('val', val, flush=True)

    return val

from fealpy.pde.helmholtz_2d import HelmholtzData2d
pde = HelmholtzData2d(k=1)
print('122222222222')
p = bm.tensor([[0, 1], [1, 2]])
a = solution(p=p)
print(p)
print(a)
print()
print()
