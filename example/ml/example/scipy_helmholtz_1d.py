import sympy as sp

from sympy.utilities.lambdify import lambdify
from typing import Sequence

from ...decorator import cartesian
from ...backend import backend_manager as bm
from ...backend import TensorLike
from ...mesher import IntervalMesher

class Exp0006(IntervalMesher):
    """
    1D Helmholtz problem:
    
        -Δu(x) + k*u(x) = f(x),   x ∈ [0,8] 
          u(x) = g(x),    on ∂Ω
    
    with the exact solution:
    
        u(x) = sin(3πx + 3π/20)*cos(2πx + π/10) + 2

    Homogeneous Dirichlet boundary conditions are applied on all edges.
    """
    def __init__(self, options: dict = {}):
        self.box = [0.0, 8.0] 
        super().__init__(interval=self.box)
        self.k = bm.tensor(options.get('k', 1.0))

        self.x = sp.symbols('x')
        self.u = sp.sin(3*sp.pi*self.x + 3*sp.pi/20) * sp.cos(2*sp.pi*self.x + sp.pi/10) + 2

    def get_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return self.box

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """Compute exact solution"""
        u_func = lambdify((self.x), self.u, bm.backend_name)
        val = u_func(p)
        return val

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """Compute gradient of solution."""
        dx = sp.diff(self.u, self.x)
        print("dx:", dx)
        grad_func = lambdify((self.x), dx, bm.backend_name)
        val = bm.stack(grad_func(p), axis=-1)
        
        return val

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute exact source """
        f = sp.diff(self.u, self.x, 2)
        print("f: ", f)
        val = lambdify((self.x), f, bm.backend_name)(p)
        return -val + self.k * self.solution(p)

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition"""
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is on boundary."""
        x = p[..., 0]
        atol = 1e-12  # 绝对误差容限
        on_boundary = (
            (bm.abs(x - 8.) < atol) | (bm.abs(x) < atol)
        )
        return on_boundary 