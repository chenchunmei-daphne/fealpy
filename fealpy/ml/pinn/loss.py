
from fealpy.backend import backend_manager as bm
bm.set_backend('pytorch')
import torch.nn as nn


class Pinn_loss():
    def __init__(self, pde_fun, bc_fun, npde, nbc, reduction='mean'):
        """
        @param pde_fun: Function representing the PDE equation.
        @param bc_fun: Function representing the boundary conditions.
        @param npde: int. Number of sample points inside the domain. Defaults to 200.
        @param nbc: int. Number of sample points on the boundary. Defaults to 100.
        @param reduction: 'none' | 'mean' | 'sum'.
            - If 'none': no reduction will be applied,
            - If 'mean': the sum of the output will be divided by the number of elements in the output,
            - If 'sum': the output will be summed.
            Defaults to 'mean'.
        """

        self.pde = pde_fun
        self.bc = bc_fun
        self.npde = npde
        self.nbc = nbc
        self.reduction = reduction

    def helmholtz_loss(self, samplerpde, samplerbc, real_net, imag_net, flag=None):
        '''
        @brief: Compute the mean squared error (squared L2 norm) function for the PINN model solving the Helmholtz equation.

        @param samplerpde: Sampler for generating points inside the domain, generated using ISampler from fealpy.ml.sampler.
        @param samplerbc: Sampler for generating points on the boundary, generated using BoxBoundarySampler from fealpy.ml.sampler.
        @param real_net: The network trained for the real part of the Helmholtz equation.
        @param imag_net: The network trained for the imaginary part of the Helmholtz equation.
        @param flag: str. Determines the type of error to return:
            - If `None`, returns the total error (real + imaginary parts).
            - If 'real', returns the error for the real part.
            - If 'imag', returns the error for the imaginary part.
            Defaults to `None`.

        @return: Tensor. The computed loss value (0.5 times the error).
        '''
        mse = nn.MSELoss(reduction=self.reduction)
        spde = samplerpde.run(self.npde)
        sbc = samplerbc.run(self.nbc, self.nbc)

        out_pde = self.pde(spde, real_net, imag_net)
        out_bc = self.bc(sbc, real_net, imag_net)

        if flag is None:

            out_pde_real = bm.real(out_pde)
            out_bc_real = bm.real(out_bc)

            out_pde_imag = bm.imag(out_pde)
            out_bc_imag = bm.imag(out_bc)

            mse_pde_real = mse(out_pde_real, bm.zeros_like(out_pde_real))
            mse_bc_real = mse(out_bc_real, bm.zeros_like(out_bc_real))

            mse_pde_imag = mse(out_pde_imag, bm.zeros_like(out_pde_imag))
            mse_bc_imag = mse(out_bc_imag, bm.zeros_like(out_bc_imag))

            sum_mse = 0.5 * (mse_pde_real + mse_pde_imag) + 0.5 * (mse_bc_real + mse_bc_imag)

        elif flag == 'real':

            out_pde_real = bm.real(out_pde)
            out_bc_real = bm.real(out_bc)

            mse_pde_real = mse(out_pde_real, bm.zeros_like(out_pde_real))
            mse_bc_real = mse(out_bc_real, bm.zeros_like(out_bc_real))

            sum_mse = 0.5 * (mse_pde_real + mse_bc_real)

        elif flag == 'imag':

            out_pde_imag = bm.imag(out_pde)
            out_bc_imag = bm.imag(out_bc)

            mse_pde_imag = mse(out_pde_imag, bm.zeros_like(out_pde_imag))
            mse_bc_imag = mse(out_bc_imag, bm.zeros_like(out_bc_imag))

            sum_mse = 0.5 * (mse_pde_imag + mse_bc_imag)

        return sum_mse
