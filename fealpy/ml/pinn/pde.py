# from fealpy.ml.pinn import parameter
# import torch as bm

import torch
from torch import Tensor
from numpy.typing import NDArray
from fealpy.backend import backend_manager as bm
bm.set_backend('pytorch')


class Helmholtz2d():
    def __init__(self, k, domain):
        """
        @param k: wave number.
        """
        self.k = bm.tensor(k)
        self.domain = domain

        c1 = bm.cos(self.k) + bm.sin(self.k)*1j
        c2 = torch.special.bessel_j0(self.k) + torch.special.bessel_j1(self.k)*1j
        self.c = c1 / c2    # 方程中的常数


    def solution(self, p: Tensor) -> Tensor:
        '''
        @brief: 2 维 Helmholtz 方程的真解.

        @para p: 二维张量且类型为 Tensor.
        '''
        x = p[..., 0:1]
        y = p[..., 1:2]
        r = bm.sqrt(x ** 2 + y ** 2)
        val = bm.zeros(x.shape, dtype=bm.complex128)
        val[:] = (bm.cos(self.k * r) - self.c * torch.special.bessel_j0(self.k * r)) / self.k
        return val

    def solution_numpy_real(self, p: NDArray) -> NDArray:
        '''
        @brief: 方程真解的实部.
        @para p: 二维张量且类型为 Numpy.
        @return: 返回的张量的类型是 Numpy.
        '''
        sol = self.solution(bm.tensor(p))
        real_ = bm.real(sol)
        return real_.detach().numpy()

    def solution_numpy_imag(self, p: NDArray) -> NDArray:
        '''
        @brief: 方程真解的虚部.
        @para p: 二维张量且类型为 Numpy.
        @return: 返回的张量的类型是 Numpy.
        '''
        sol = self.solution(bm.tensor(p))
        imag_ = bm.imag(sol)
        return imag_.detach().numpy()

    def source(self, p: Tensor) -> Tensor:
        '''
        @brief: 2 维 Helmholtz 方程的源项.
        @para p: 二维张量且类型为 Tensor.
        @return: 返回源项 f.
        '''

        x = p[..., 0:1]
        y = p[..., 1:2]
        r = bm.sqrt(x ** 2 + y ** 2)
        f = bm.zeros(x.shape, dtype=bm.complex128)
        f[:] = bm.sin(self.k * r) / r  # 源项
        return f

    def grad(self, p: Tensor) -> Tensor:
        '''
        @brief: 方程真解的梯度.
        @para p: 二维张量且类型为 Tensor.
        @return: 返回的张量的类型是Tensor.
        '''

        x = p[..., 0:1]
        y = p[..., 1:2]
        r = bm.sqrt(x ** 2 + y ** 2)

        val = bm.zeros(p.shape, dtype=bm.complex128)
        u_r = self.c * torch.special.bessel_j1(self.k * r) - bm.sin(self.k * r)
        val[..., 0:1] = u_r * x / r
        val[..., 1:2] = u_r * y / r
        return val


    def pde(self, p: Tensor, real_net, imag_net) -> Tensor:
        '''
        @brief: 2 维 Helmholtz 方程的真解的 PDE.

        @para p: 二维张量且类型为 Tensor.

        @return: u_xx + u_yy + k ** 2 * u + f.
        '''

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


    def robin_bc(self, p: Tensor, real_net, imag_net) -> Tensor:
        '''
        @brief: 2 维 Helmholtz 方程的 Robin 边界条件.

        @para p: 二维张量且类型为 Tensor.

        @return: 返回的张量的类型是Tensor.
        '''

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
        g = (self.grad(p) * n).sum(dim=-1, keepdim=True) + kappa * self.solution(p)

        return (grad_u * n).sum(dim=-1, keepdim=True) + kappa * u - g

