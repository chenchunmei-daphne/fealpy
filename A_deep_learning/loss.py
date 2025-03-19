import torch
import torch.nn as nn
from fealpy.ml.sampler import BoxBoundarySampler, ISampler

# mse_cost_func = nn.MSELoss(reduction='mean')

def loss(npde, nbc, pde, bc, pde_domain=None, flag=None):
    '''
    param npde: 区域内部的采样点个数
    param nbc: 边界上的采样点个数
    param pde: 关于PDE方程的函数
    param bc: 边界条件的函数
    param pde_domain：区域。An object that can be converted to a `numpy.ndarray`,\
               representing the ranges in each sampling axis.\
               For example, if sampling x in [0, 1] and y in [4, 5],\
               use `ranges=[[0, 1], [4, 5]]`.
    param flag: 确定返回的是全部的误差（flag=None），还是实部（flag=real）或者虚部（flag=imag）的误差
    return:
    '''

    if pde_domain == None:
        pde_domain = [[-0.5, 0.5], [-0.5, 0.5]]

    mse = nn.MSELoss(reduction='mean')
    samplerpde = ISampler(pde_domain, requires_grad=True)
    samplerbc = BoxBoundarySampler(pde_domain[0], pde_domain[1], requires_grad=True)

    spde = samplerpde.run(npde)
    sbc = samplerbc.run(nbc, nbc)

    out_pde = pde(spde)
    out_bc = bc(sbc)

    out_pde_real = torch.real(out_pde)
    out_bc_real = torch.real(out_bc)

    out_pde_imag = torch.imag(out_pde)
    out_bc_imag = torch.imag(out_bc)

    mse_pde_real = mse(out_pde_real, torch.zeros_like(out_pde_real))
    mse_bc_real = mse(out_bc_real, torch.zeros_like(out_bc_real))

    mse_pde_imag = mse(out_pde_imag, torch.zeros_like(out_pde_imag))
    mse_bc_imag = mse(out_bc_imag, torch.zeros_like(out_bc_imag))

    if flag==None:

        sum_mse = mse_pde_real + mse_pde_imag + mse_bc_real + mse_bc_imag

    elif flag == 'real':

        sum_mse = mse_pde_real + mse_bc_real

    elif flag == 'imag':

        sum_mse =  mse_pde_imag + mse_bc_imag

    return sum_mse


def pde(x):
    return 0 * x + 1 + 1 * 1j

def bc(x):
    return 0 * x + 1 - 1 * 1j


a= loss(10, 10, pde, bc, flag='imag')
print(a)

a= loss(10, 10, pde, bc, flag='real')
print(a)


a= loss(10, 10, pde, bc)
print(a)
