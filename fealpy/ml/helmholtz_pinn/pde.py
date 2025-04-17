import torch

from torch import Tensor
from numpy.typing import NDArray
from scipy.special import jv

from fealpy.backend import backend_manager as bm
bm.set_backend('pytorch')

from fealpy.decorator import cartesian


class Helmholtz2d():
    """
    A class representing the 2D Helmholtz equation with methods for solution, gradient,
    source term, PDE formulation, and Robin boundary conditions.

    Parameters:
        k: Wave number for the Helmholtz equation.
        domain: The computational domain for the equation.
    """
    def __init__(self, k=1.0):
        self.k = bm.tensor(k)

        c1 = bm.cos(self.k) + bm.sin(self.k)*1j
        c2 = jv(0, self.k) + 1j * jv(1, self.k)
        # c2 = torch.special.bessel_j0(self.k) + torch.special.bessel_j1(self.k)*1j
        self.c = c1 / c2    # 方程中的常数

    def domain(self):
        return (-1.0, 1.0, -1.0, 1.0)

    @cartesian
    def solution(self, p: Tensor) -> Tensor:
        """
        The exact solution of the 2D Helmholtz equation.

        Parameters:
            p: Input tensor representing spatial coordinates (2D).

        Returns:
            Tensor: The exact solution at given points.
        """
        x = p[..., 0:1]
        y = p[..., 1:2]
        r = bm.sqrt(x ** 2 + y ** 2)
        val = bm.zeros(x.shape, dtype=bm.complex128)
        val[:] = (bm.cos(self.k * r) - self.c * torch.special.bessel_j0(self.k * r)) / self.k
        return val

    def solution_numpy_real(self, p: NDArray) -> NDArray:
        """
        The real part of the exact solution, with numpy array input/output.

        Parameters:
            p: Input numpy array representing spatial coordinates.

        Returns:
            NDArray: The real part of the solution at given points.
        """
        sol = self.solution(bm.tensor(p))
        real_ = bm.real(sol)
        return real_.detach().numpy()

    def solution_numpy_imag(self, p: NDArray) -> NDArray:
        """
           The imaginary part of the exact solution, with numpy array input/output.

           Parameters:
               p: Input numpy array representing spatial coordinates.

           Returns:
               NDArray: The imaginary part of the solution at given points.
           """
        sol = self.solution(bm.tensor(p))
        imag_ = bm.imag(sol)
        return imag_.detach().numpy()

    @cartesian
    def source(self, p: Tensor) -> Tensor:
        """
        The source term of the 2D Helmholtz equation.

        Parameters:
            p: Input tensor representing spatial coordinates.

        Returns:
            Tensor: The source term at given points.
        """
        x = p[..., 0]
        y = p[..., 1]
        r = bm.sqrt(x ** 2 + y ** 2)
        f = bm.zeros(x.shape, dtype=bm.complex128)
        f[:] = bm.sin(self.k * r) / r  # 源项
        return f

    @cartesian
    def gradient(self, p: Tensor) -> Tensor:
        """
        The gradient of the exact solution.

        Parameters:
            p: Input tensor representing spatial coordinates.

        Returns:
            Tensor: The gradient of the solution at given points.
        """
        x = p[..., 0:1]
        y = p[..., 1:2]
        r = bm.sqrt(x ** 2 + y ** 2)

        val = bm.zeros(p.shape, dtype=bm.complex128)
        u_r = self.c * torch.special.bessel_j1(self.k * r) - bm.sin(self.k * r)
        val[..., 0:1] = u_r * x / r
        val[..., 1:2] = u_r * y / r
        return val

    @cartesian
    def robin(self, p, n):
        x = p[..., 0]
        y = p[..., 1]
        grad = self.gradient(p) # (NC, NQ, dof_numel)
        val = bm.sum(grad*n[:, bm.newaxis, :], axis=-1)
        kappa = bm.broadcast_to(bm.tensor(1j * self.k), shape=x.shape)
        val += kappa*self.solution(p) 
        return val

    def pde_func(self, p: Tensor, real_net, imag_net) -> Tensor:
        """
        The PDE formulation of the 2D Helmholtz equation.

        Parameters:
            p: Input tensor representing spatial coordinates.
            real_net: Network for real part of the solution.
            imag_net: Network for imaginary part of the solution.

        Returns:
            Tensor: The PDE residual u_xx + u_yy + k**2 * u + f.
        """
        from fealpy.ml import gradient

        u = real_net(p)+ imag_net(p)*1j

        u_x_real, u_y_real = gradient(u.real, p, create_graph=True, split=True)
        u_xx_real, _ = gradient(u_x_real, p, create_graph=True, split=True)
        _, u_yy_real = gradient(u_y_real, p, create_graph=True, split=True)

        u_x_imag, u_y_imag = gradient(u.imag, p, create_graph=True, split=True)
        u_xx_imag, _ = gradient(u_x_imag, p, create_graph=True, split=True)
        _, u_yy_imag = gradient(u_y_imag, p, create_graph=True, split=True)

        u_xx = u_xx_real + u_xx_imag*1j
        u_yy = u_yy_real + u_yy_imag*1j

        f = self.source(p)  # 源项

        return u_xx + u_yy + self.k ** 2 * u + f

    
    def robin_func(self, p: Tensor, real_net, imag_net) -> Tensor:
        """
        The Robin boundary condition for the 2D Helmholtz equation.

        Parameters:
            p: Input tensor representing spatial coordinates.
            real_net: Network for real part of the solution.
            imag_net: Network for imaginary part of the solution.

        Returns:
            Tensor: The Robin boundary condition residual.
        """

        from fealpy.ml import gradient

        u = real_net(p) + imag_net(p) * 1j  # 数值解

        x = p[..., 0]
        y = p[..., 1]
        n = bm.zeros_like(p)  # 法向量 n
        n[x > bm.abs(y), 0] = 1.0
        n[y > bm.abs(x), 1] = 1.0
        n[x < -bm.abs(y), 0] = -1.0
        n[y < -bm.abs(x), 1] = -1.0

        grad_u_real = gradient(u.real, p, create_graph=True, split=False)  # 数值解的梯度
        grad_u_imag = gradient(u.imag, p, create_graph=True, split=False)
        grad_u = grad_u_real + grad_u_imag * 1j

        kappa = bm.tensor(0.0) + self.k * 1j
        g = (self.gradient(p) * n).sum(dim=-1, keepdim=True) + kappa * self.solution(p)

        return (grad_u * n).sum(dim=-1, keepdim=True) + kappa * u - g

