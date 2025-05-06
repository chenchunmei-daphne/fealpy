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
        val = bm.zeros_like(p)
        return val

    def dirichlet(self, p: TensorLike) -> TensorLike:
        return self.solution(p)
    

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
        val = bm.zeros_like(p)
        return val

    def dirichlet(self, p: TensorLike) -> TensorLike:
        return self.solution(p)
    

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
        return self.solution(p)
    
pde1d = SinPDEData1D()
p = bm.array([[0.0], [1.0]])
print(pde1d.solution(p))
print(pde1d.gradient(p))
print(pde1d.source(p))

pde2d = SinPDEData2D()
p = bm.array([[0.0, 0.0], [1.0, 1.0]])
print(pde2d.solution(p))
print(pde2d.gradient(p))
print(pde2d.source(p))

pde3d = SinPDEData3D()
p = bm.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
print(pde3d.solution(p))
print(pde3d.gradient(p))
print(pde3d.source(p))
print(pde3d.dirichlet(p))
print(pde3d.geo_dimension())