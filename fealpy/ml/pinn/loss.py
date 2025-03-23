
from fealpy.backend import backend_manager as bm
bm.set_backend('pytorch')
import torch.nn as nn


class Pinn_loss():
    def __init__(self, pde_fun, bc_fun, npde, nbc,reduction='mean'):
        self.pde = pde_fun
        self.bc = bc_fun
        self.npde = npde
        self.nbc = nbc
        self.reduction = reduction

    def helmholtz_loss(self, samplerpde, samplerbc, real_net, imag_net, flag=None):

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




# def pinn_loss(samplerpde, samplerbc, pde, bc, real_net, imag_net, npde=200, nbc=100, flag=None):

    """
    @brief: Compute the loss function for the PINN model solving the Helmholtz equation.

    @param samplerpde: Sampler for generating points inside the domain, generated using ISampler from fealpy.ml.sampler.
    @param samplerbc: Sampler for generating points on the boundary, generated using BoxBoundarySampler from fealpy.ml.sampler.
    @param pde: Function representing the PDE equation.
    @param bc: Function representing the boundary conditions.
    @param npde: int. Number of sample points inside the domain. Defaults to 200.
    @param nbc: int. Number of sample points on the boundary. Defaults to 100.
    @param flag: str. Determines the type of error to return:
           - If `None`, returns the total error (real + imaginary parts).
           - If 'real', returns the error for the real part.
           - If 'imag', returns the error for the imaginary part.
           Defaults to `None`.

    @return: Tensor. The computed loss value (0.5 times the error).
    """

# def loss(samplerpde, samplerbc, pde, bc, real_net, imag_net, npde=200, nbc=100, flag=None):
def loss(samplerpde, samplerbc, pde, bc, npde=200, nbc=100, flag=None):
    """
    @brief: Compute the loss function for the PINN model solving the Helmholtz equation.

    @param samplerpde: Sampler for generating points inside the domain, generated using ISampler from fealpy.ml.sampler.
    @param samplerbc: Sampler for generating points on the boundary, generated using BoxBoundarySampler from fealpy.ml.sampler.
    @param pde: Function representing the PDE equation.
    @param bc: Function representing the boundary conditions.
    @param npde: int. Number of sample points inside the domain. Defaults to 200.
    @param nbc: int. Number of sample points on the boundary. Defaults to 100.
    @param flag: str. Determines the type of error to return:
           - If `None`, returns the total error (real + imaginary parts).
           - If 'real', returns the error for the real part.
           - If 'imag', returns the error for the imaginary part.
           Defaults to `None`.

    @return: Tensor. The computed loss value (0.5 times the error).
    """
    import torch
    mse = nn.MSELoss(reduction='mean')
    spde = samplerpde.run(npde)
    sbc = samplerbc.run(nbc, nbc)

    # out_pde = pde(spde, real_net, imag_net)
    # out_bc = bc(sbc, real_net, imag_net)

    out_pde = pde(spde)
    out_bc = bc(sbc)
    if flag == None:

        out_pde_real = torch.real(out_pde)
        out_bc_real = torch.real(out_bc)

        out_pde_imag = torch.imag(out_pde)
        out_bc_imag = torch.imag(out_bc)

        mse_pde_real = mse(out_pde_real, torch.zeros_like(out_pde_real))
        mse_bc_real = mse(out_bc_real, torch.zeros_like(out_bc_real))

        mse_pde_imag = mse(out_pde_imag, torch.zeros_like(out_pde_imag))
        mse_bc_imag = mse(out_bc_imag, torch.zeros_like(out_bc_imag))

        sum_mse = 0.5 * (mse_pde_real + mse_pde_imag) + 0.5 * (mse_bc_real + mse_bc_imag)

    elif flag == 'real':

        out_pde_real = torch.real(out_pde)
        out_bc_real = torch.real(out_bc)

        mse_pde_real = mse(out_pde_real, torch.zeros_like(out_pde_real))
        mse_bc_real = mse(out_bc_real, torch.zeros_like(out_bc_real))

        sum_mse = 0.5 * (mse_pde_real + mse_bc_real)

    elif flag == 'imag':

        out_pde_imag = torch.imag(out_pde)
        out_bc_imag = torch.imag(out_bc)

        mse_pde_imag = mse(out_pde_imag, torch.zeros_like(out_pde_imag))
        mse_bc_imag = mse(out_bc_imag, torch.zeros_like(out_bc_imag))

        sum_mse = 0.5 * (mse_pde_imag + mse_bc_imag)

    return sum_mse