from typing import Sequence
from ...decorator import cartesian
from ...backend import TensorLike 
from ...backend import backend_manager as bm


class ExpExpData2D:
    """
    2D Poisson equation:

        -∇·(∇u(x, y)) = f(x, y),  (x, y) ∈ Ω = (-5, 5) x (-5, 5)
                                  u(x, y) = g(x, y),  on ∂Ω

    where:
        f(x, y) = exp{-[(x+2)^2/2 + y^2/2]} - 0.5*exp{-[(x-2)^2/2 + y^2/2]}
        g(x, y) = 0
    """


    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return [-5.0, 5.0,-5.0, 5.0]

    
    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Return the source term f(x, y)"""
        x, y = p[..., 0], p[..., 1]
        term1 = bm.exp(-((x + 2)**2 / 2 + y**2 / 2))
        term2 = 0.5 * bm.exp(-((x - 2)**2 / 2 + y**2 / 2))
        term1 -= term2
        return term1

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition."""
        return bm.zeros_like(p[..., 0])

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is on boundary."""
        x, y = p[..., 0], p[..., 1]
        atol = 1e-8
        return (
            (bm.abs(x - 5) < atol) | (bm.abs(x + 5) < atol) |
            (bm.abs(y - 5) < atol) | (bm.abs(y + 5) < atol)
        )

