from typing import Optional, Sequence
from ...decorator import cartesian
from ...backend import backend_manager as bm
from ...backend import TensorLike


class Exp0013:
    """
    5D Poisson problem:
    
        -Δu(x_1, ..., x_5) = f(x_1, ..., x_5),  (x_1, ..., x_5) ∈ (0, 1)^5
         u(x_1, ..., x_5) = 0,                   on ∂Ω

    with the exact solution:

        u(x_1, ..., x_5) = sin(πx_1)·sin(πx_2)·sin(πx_3)·sin(πx_4)·sin(πx_5)

    The corresponding source term is:

        f(x_1, ..., x_5) = 5·π²·sin(πx_1)·sin(πx_2)·sin(πx_3)·sin(πx_4)·sin(πx_5)

    Homogeneous Dirichlet boundary conditions are applied on all boundaries.
    """
    def __init__(self):
        self.box = [0.0, 1.0] * 5  # [xmin, xmax, ymin, ymax, zmin, zmax, ...]

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 5

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax, zmin, zmax, ...]."""
        return self.box

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """Compute exact solution u(x) = sin(πx_1)·sin(πx_2)·...·sin(πx_5)."""
        pi = bm.pi
        result = bm.ones_like(p[..., 0])
        for i in range(5):
            result = result * bm.sin(pi * p[..., i])
        return result

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """Compute gradient of solution."""
        pi = bm.pi
        grad_components = []
        for i in range(5):
            grad_i = bm.ones_like(p[..., 0])
            for j in range(5):
                if j == i:
                    grad_i = grad_i * pi * bm.cos(pi * p[..., j])
                else:
                    grad_i = grad_i * bm.sin(pi * p[..., j])
            grad_components.append(grad_i)
        return bm.stack(grad_components, axis=-1)

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute exact source term f(x) = 5·π²·sin(πx_1)·...·sin(πx_5)."""
        pi = bm.pi
        result = bm.ones_like(p[..., 0])
        for i in range(5):
            result = result * bm.sin(pi * p[..., i])
        return 5 * pi**2 * result

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition (zero on all boundaries)."""
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is on boundary (any coordinate is 0 or 1)."""
        atol = 1e-12
        on_boundary = bm.zeros(p.shape[:-1], dtype=bool)
        for i in range(5):
            coord = p[..., i]
            on_boundary = on_boundary | (bm.abs(coord) < atol) | (bm.abs(coord - 1.0) < atol)
        return on_boundary

    def scaling_function(self, p: TensorLike) -> TensorLike:
        """Compute scaling function that satisfies the boundary conditions."""
        return bm.zeros_like(p[..., 0])