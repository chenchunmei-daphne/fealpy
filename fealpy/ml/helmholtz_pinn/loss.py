import torch.nn as nn

from fealpy.backend import backend_manager as bm
bm.set_backend('pytorch')


class Pinn_loss():
    """
    A class for computing the loss function of Physics-Informed Neural Networks (PINN)
    for solving partial differential equations, particularly the Helmholtz equation.

    Parameters:
        pde_fun: Function representing the PDE equation.
        bc_fun: Function representing the boundary conditions.
        npde: Number of sample points inside the domain. Defaults to 200.
        nbc: Number of sample points on the boundary. Defaults to 100.
        reduction: Reduction method for the loss calculation.
            - 'none': No reduction will be applied.
            - 'mean': The sum of the output will be divided by the number of elements.
            - 'sum': The output will be summed.
            Defaults to 'mean'.
    """
    def __init__(self, pde_fun, bc_fun, npde, nbc, reduction='mean'):
        self.pde = pde_fun
        self.bc = bc_fun
        self.npde = npde
        self.nbc = nbc
        self.reduction = reduction

    def helmholtz_loss(self, samplerpde, samplerbc, real_net, imag_net, flag=None):
        """
               Compute the mean squared error (squared L2 norm) function for the PINN model
               solving the Helmholtz equation.

               Parameters:
                   samplerpde: Sampler for generating points inside the domain, generated
                              using ISampler from fealpy.ml.sampler.
                   samplerbc: Sampler for generating points on the boundary, generated
                             using BoxBoundarySampler from fealpy.ml.sampler.
                   real_net: The network trained for the real part of the Helmholtz equation.
                   imag_net: The network trained for the imaginary part of the Helmholtz equation.
                   flag: Determines the type of error to return:
                       - None: Returns the total error (real + imaginary parts).
                       - 'real': Returns the error for the real part.
                       - 'imag': Returns the error for the imaginary part.
                       Defaults to None.

               Returns:
                   Tensor: The computed loss value (0.5 times the error).
               """
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
