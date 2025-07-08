from typing import Sequence
from ...decorator import cartesian
from ...backend import TensorLike 
from ...backend import backend_manager as bm


class StepData2D:
    """
    2D Poisson equation:

        -∇·(∇u(x, y)) = f(x, y),  (x, y) ∈ Ω = (-4, 4) x (-4, 4)
                                  u(x, y) = g(x, y),  on ∂Ω

    where:
        f(x, y) = -1 for √((x + 2)^2 + y^2) ≤ 1.5;
                = 0 otherwise.
        g(x, y) = 0 for x = -4;
                = 1 for x = 4.
    """


    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return [-4.0, 4.0, -4.0, 4.0]

    
    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Return the source term f(x, y)"""
        x, y = p[..., 0], p[..., 1]
        r = bm.sqrt((x + 2)**2 + y**2)
        return -1.0 * (r <= 1.5)

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition:
            - u(-4, y) = 0
            - u(4, y) = 1
        """
        p = p[self.is_dirichlet_boundary(p)]
        x = p[..., 0]
        atol = 1e-8
        cond1 = (bm.abs(x + 4) < atol)  # x = -4
        cond2 = (bm.abs(x - 4) < atol)   # x = 4
        return bm.where(cond1, 0.0, bm.where(cond2, 1.0, 0.0))

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is on boundary."""
        x = p[..., 0]
        atol = 1e-8
        return (
            (bm.abs(x - 4) < atol) | (bm.abs(x + 4) < atol)
        )

    @cartesian
    def boundary_scaling_function(self, p: TensorLike) -> TensorLike:
        """A boundary scaling is done by addition of the function B(x, y) 
        which satisfies the boundary and still satisfies the boundary along the edges."""
        x = p[..., 0]

        return (x + 4.0) / 8.0
    
    @cartesian
    def gaussian_shape_function(self, p: TensorLike) -> TensorLike:
        """gaussian function with zero boundary through shape function U (x, y)."""
        x, y = p[..., 0], p[..., 1]
        return bm.exp(-0.5*((x+1)**2 + y**2))

