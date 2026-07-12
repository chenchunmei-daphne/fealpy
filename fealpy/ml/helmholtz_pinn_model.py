import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from typing import Union

from fealpy.backend import bm
from fealpy.utils import timer
from fealpy.typing import TensorLike
from fealpy.model import ComputationalModel, PDEModelManager
from fealpy.model.helmholtz import HelmholtzPDEDataT
from fealpy.mesh import MeshDS

from fealpy.ml import gradient, optimizers, activations

from fealpy.ml.modules import Solution
from fealpy.ml.sampler import BoxBoundarySampler, ISampler

import time
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import BilinearForm, LinearForm
from fealpy.fem import ScalarDiffusionIntegrator, ScalarMassIntegrator
from fealpy.fem import ScalarRobinBCIntegrator, ScalarSourceIntegrator, ScalarRobinSourceIntegrator
from fealpy.solver import cg


class HelmholtzPINNModel(ComputationalModel):
    """Physics-Informed Neural Network (PINN) model for solving Helmholtz equations.
    
    Implements a PINN framework to solve Helmholtz PDE problems using neural networks.
    Handles PDE residual calculation, boundary condition enforcement, and training process.
    Supports both uniform and random sampling strategies for collocation points.
    Specialized for complex-valued solutions (real + imaginary components).
    
    Parameters:
        options(dict): If None, default parameters from get_options() will be used.
            Configuration dictionary containing:
            - pde(int or HelmholtzPDEDataT): PDE definition;
            - lr(float): Learning rate;
            - epochs(int): Number of training epochs;
            - weights(tuple): Weight for the equation loss and boundary loss;
            - hidden_size(tuple): Tuple of hidden layer sizes;
            - npde(int): Number of PDE collocation points;
            - nbc(int): Number of boundary collocation points;
            - activation(str): Activation function, can choose from 'Tanh', 'ReLU', 'LeakyReLU', 'Sigmoid', 'LogSigmoid', 'Softmax', 'LogSoftmax';
            - optimizer(str): Optimization algorithm, can choose from 'Adam', 'SGD';
            - sampling_mode(str): Sampling strategy, can choose from 'linspace' or 'random';
            - complex(bool): Boolean flag for complex-valued solutions, if True, the solutions is complex-valued;
            - wave(float): Wave number k for Helmholtz equation.
            - step_size(int): Period of learning rate decay;
            - gamma(float): Multiplicative factor of learning rate decay.
            - pbar_log(bool): Whether to use progress bar for logging;
            - log_level(str): Logging level, can choose from 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
        
    Attributes:
        pde(PoissonPDEDataT): Helmholtz PDE problem definition.

        gd(int): Geometric dimension.

        domain(list): Computational domain boundaries.

        mesh(TriangleMesh or UniformMesh): Discretization mesh for error estimation.

        net(torch.nn.Module): Neural network model.
        
        optimizer(torch.optim.Optimizer): Training optimizer.
        
        Loss(list): Training loss history.
        
        error_real(list): Real part error history (vs FEM solution).
        
        error_imag(list): Imaginary part error history (vs FEM solution, only when complex=True).
        
        options(dict): Initial configuration dictionary.
        
        complex(bool): Flag indicating complex-valued solutions.
        
        k(float): Wave number for Helmholtz equation.

        tmr(timer): Timer for measuring training time.
    
    Methods:
        set_pde(): Initialize PDE problem.

        set_network(): Configure neural network architecture.

        set_mesh(): Initialize computational mesh.

        set_n(): Compute normal vectors at boundary points.

        pde_residual(): Compute PDE residual (Δu + k²u + f).

        bc_residual(): Compute boundary condition residual.

        run(): Execute training process.

        predict(): Make predictions at given points.

        show(): Visualize results.  
    
    Reference:
        https://wnesm678i4.feishu.cn/wiki/U219wwT18iH4v7kNTOacxl8cnXb?from=from_copylink.
        
    Examples:
        >>> from fealpy.backend import bm  
        >>> bm.set_backend('pytorch')  # Set the backend to PyTorch  
        >>> from fealpy.ml import HelmholtzPINNModel  
        >>> options = HelmholtzPINNModel.get_options()  # Get the default options of the network  
        >>> model = HelmholtzPINNModel(options=options)  
        >>> model.run()   # Train the network  
        >>> model.show()   # Show the results of the network training  
    """
    def __init__(self, options: dict = {}):
        self.options = self.get_options()
        self.options.update(options)
        
        self.pbar_log = self.options['pbar_log']
        self.log_level = self.options['log_level']
        super().__init__(pbar_log=self.pbar_log, log_level=self.log_level)

        self.k = self.options['wave']  
        self.lr = self.options['lr']   
        self.epochs = self.options['epochs'] 
        self.hidden_size = self.options['hidden_size']    
        self.activation = activations[self.options['activation']]   
        self.npde = self.options['npde']   
        self.nbc = self.options['nbc']   
        self.weights = self.options['weights']  
        self.complex = self.options['complex'] 
        self.tmr = timer() 
 
        self.set_pde(self.options['pde'])  # PDE 
        self.set_mesh(self.options['mesh_size']) 
        self.set_network()

        self.fem_func = None
        self.fem_solution = None
        self.fem_space = None
        self.fem_solve_time = None
        self.train_total_time = None
        self.train_avg_time = None

    @classmethod
    def get_options(cls):
        """Get default configuration parameters for the model.
        
        Defines and returns default configurations for the model through a command-line argument parser,
        including PDE problem number, grid size, network structure, and optimizer parameters.
        
        Returns:
            options(dict): Dictionary containing all configuration parameters with parameter names as keys and default values
        """

        import argparse
        parser = argparse.ArgumentParser(description="Helmholtz equation solver using PINN.")

        parser.add_argument('--pde',default=1, type=int,
                            help="Built-in PDE example ID for different Helmholtz problems, default is 1.")
        
        parser.add_argument('--mesh_size', default=30, type=int,
                            help='Number of grid points along each dimension, default is 30.')

        parser.add_argument('--complex',  default=True, type=bool,
                            help="Enable complex-valued solution modeling, default is True")

        parser.add_argument('--wave', default=1.0, type=float,
                            help="Wave number k for Helmholtz equation (Δu + k²u + f = 0), default is 1.0")

        parser.add_argument('--sampling_mode', default='random', type=str,
                            help="Sampling method for collocation points: 'random' or 'linspace', default is 'random'")

        parser.add_argument('--npde', default=400, type=int,
                            help='Number of PDE samples, default is 400.')

        parser.add_argument('--nbc', default=100, type=int,
                            help='Number of boundary condition samples, default is 100.')
        
        parser.add_argument('--weights', default=(1, 30), type=tuple,
                            help='The first value is the weight for the equation loss, and the second ' \
                            'value is the weight for the boundary loss., default is (1, 30).')

        parser.add_argument('--hidden_size', default=(50, 50, 50, 50), type=tuple,
                            help='Default hidden sizes, default is (50, 50, 50, 50).')

        parser.add_argument('--optimizer', default="Adam",  type=str,
                            help='Optimizer to use for training, default is Adam')

        parser.add_argument('--activation', default="Tanh", type=str,
                            help='Activation function, default is Tanh')

        parser.add_argument('--lr', default=0.001, type=float,
                            help='Learning rate for the optimizer, default is 0.001.')

        parser.add_argument('--step_size', default=0, type=int,
                            help='Period of learning rate decay, default is 0.')

        parser.add_argument('--gamma', default=0.99, type=float,
                            help='Multiplicative factor of learning rate decay. Default: 0.99.')

        parser.add_argument('--epochs', default=3000, type=int,
                            help='Number of training epochs, default is 3000.')

        parser.add_argument('--pbar_log', default=True, type=bool,
                            help='Whether to show progress bar, default is True')

        parser.add_argument('--log_level', default='INFO', type=str,
                            help='Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL')
        options = vars(parser.parse_args())
        return options
        
    def set_pde(self, pde: Union[HelmholtzPDEDataT, int]=1):
        """Initialize the PDE problem definition and boundary condition sampler, internal sampler.
        
        Parameters:
            pde(Union[HelmholtzPDEDataT, int]): Either a Helmholtz equation problem object or the ID (integer) of a predefined example. 
                If an integer, the corresponding predefined Helmholtz equation problem is retrieved from the PDE model manager.
        """
        if isinstance(pde, int):
            self.pde = PDEModelManager('helmholtz').get_example(pde, k=self.k)
        else:
            self.pde = pde 
        
        self.gd = self.pde.geo_dimension() 
        self.domain = self.pde.domain()

    def set_network(self, net=None):
        """Configure the neural network architecture and optimizer, learning rate scheduler.
        
        Parameters:
            net(torch.nn.Module): Custom network architecture. If None, creates default MLP.
        """
        if net == None:
            layers = []
            sizes = (self.gd,) + self.hidden_size
            for i in range(len(sizes)-1):
                layers.append(nn.Linear(sizes[i], sizes[i+1], dtype=bm.float64))
                if i < len(sizes)-1:  
                    layers.append(self.activation())
                    
            if self.complex:
                layers.append(nn.Linear(sizes[-1], 2, dtype=bm.float64))
            else:
                layers.append(nn.Linear(sizes[-1], 1, dtype=bm.float64))
            net = nn.Sequential(*layers)
            
        self.net = Solution(net, self.complex)

        # optimizer
        opt = optimizers[self.options['optimizer']]
        self.optimizer = opt(params=self.net.parameters(), lr=self.lr)

        # scheduler
        step_size = self.options['step_size']
        gamma = self.options['gamma']
        self.set_steplr(step_size, gamma)

    def set_mesh(self, mesh_size: int=30, mesh=None):
        """Create computational mesh.
        
        Creates a computational mesh over the domain defined by the PDE based on the specified mesh size.
        
        Parameters:
            mesh_size(int): Number of nodes in each dimension.

            mesh: Mesh object. If None, creates a default mesh based on the PDE domain and mesh size.
        """
        if mesh == None:
            gd = self.gd
            self.mesh_size = (mesh_size, ) * gd
            cell_size = tuple(x - 1 for x in self.mesh_size)
            self.mesh = self.pde.init_mesh['uniform_tri'](*cell_size)
        else:
            self.mesh = mesh

    def set_steplr(self, step_size: int=0, gamma: float=0.9):
        """Create learning rate scheduler
        
        Initializes a learning rate scheduler for decaying the learning rate periodically during training.
        
        Parameters:
            step_size(int): Default is 0. Period for learning rate decay, i.e., decay every step_size epochs. No scheduler is used if step_size = 0.
            
            gamma(float): default is 0.9. Multiplicative factor for learning rate decay, new_lr = current_lr * gamma.
        """
        if step_size == 0:
            self.steplr = None
        else:
            self.steplr = StepLR(self.optimizer, step_size, gamma)

    def set_n(self, p: TensorLike) -> TensorLike:
        """Compute normal vectors at boundary points (for rectangular domains)
        
        Parameters:
            p(TensorLike): Boundary point coordinates.
                
        Returns:
            TensorLike: Unit normal vectors.
        """
        n = bm.zeros_like(p)
        tol = 1e-4
        dim = self.gd
        coords = [p[..., i] for i in range(dim)] 
        
        for axis in range(dim):
            min_val, max_val = self.domain[2*axis], self.domain[2*axis+1]
            min_mask = bm.abs(coords[axis] - min_val) <= tol
            max_mask = bm.abs(coords[axis] - max_val) <= tol
            if axis == 0:
                active_mask = bm.ones_like(min_mask, dtype=bool) 
            else:
                active_mask = ~bm.any(n != 0, axis=-1) 
            
            n[min_mask & active_mask, axis] = -1.0 
            n[max_mask & active_mask, axis] = 1.0 
        
        return n

    def pde_residual(self, p: TensorLike) -> TensorLike:
        """Compute PDE residual (Δu + k²u + f)
        
        Parameters:
            p(TensorLike): Collocation point coordinates.
                
        Returns:
            TensorLike: PDE residual values.

        Notes:
            Helmholtz equation form: Δu + k²u + f = 0.
            Uses automatic differentiation to compute Laplacian.
        """
        u = self.net(p)
        f = self.pde.source(p).flatten()
        grad_u = gradient(u.real, p, create_graph=True)  ## (npde, dim)
        laplacian = bm.zeros(u.shape[0])    
        
        for i in range(p.shape[-1]):
            u_ii = gradient(grad_u[..., i], p, create_graph=True, split=True)[i]   
            laplacian += u_ii.flatten()
        
        if self.complex:
            grad_u_imag = gradient(u.imag, p, create_graph=True)
            laplacian_imag = bm.zeros(u.shape[0])
            for i in range(p.shape[-1]):
                u_ii = gradient(grad_u_imag[..., i], p, create_graph=True, split=True)[i]
                laplacian_imag += u_ii.flatten()
            laplacian = laplacian + 1j * laplacian_imag
        assert f.shape == laplacian.shape, f"Shape mismatch: f.shape={f.shape}, laplacian.shape={laplacian.shape}."
        val = laplacian + self.k**2 * u.flatten() + f
        return val

    def bc_residual(self, p: TensorLike) -> TensorLike:
        """Compute boundary condition residual
        
        Parameters:
            p(TensorLike): Boundary point coordinates.

        Returns:
            TensorLike: Boundary condition residual values.

        Notes:
            Supported boundary conditions:
            1. Dirichlet: u - g = 0
            2. Robin: i*k*u + ∂u/∂n - g = 0, i serves as the imaginary unit.
        """
        u = self.net(p).flatten()
        if hasattr(self.pde, 'dirichlet'):
            g = self.pde.dirichlet(p).flatten()
            assert u.shape == g.shape, f"Shape mismatch: u.shape={u.shape}, g.shape={g.shape}."
            val = u - g
        elif hasattr(self.pde, 'robin'):
            n = self.set_n(p)
            g = self.pde.robin(p, n)
            grad_u_real = gradient(u.real, p, create_graph=True, split=False)
            grad_u_imag = gradient(u.imag, p, create_graph=True, split=False)
            grad_u = grad_u_real + 1j * grad_u_imag
            kappa = bm.tensor(0.0 + 1j * self.k)
            g_hat = (grad_u * n).sum(dim=-1) + kappa * u
           
            assert g_hat.shape == g.shape, f"Shape mismatch: g_hat.shape={g_hat.shape}, g.shape={g.shape}."
            val = g_hat - g

        return val
    
    def fem(self):
        """Solve Helmholtz equation using standard FEM (Lagrange P1) for comparison.
        
        Uses HelmholtzLFEMModel with 'standard' method to solve the PDE.
        """
        import time
        
        # 创建 HelmholtzLFEMModel 的配置选项
        fem_options = {
            'pbar_log': self.pbar_log,
            'log_level': self.log_level,
            'pde': self.pde,  # 直接使用已经设置好的 pde
            'init_mesh': 'uniform_tri',  # 使用相同的网格
            'nx':  self.options['mesh_size']-1,
            'ny':  self.options['mesh_size']-1,
            'space_degree': 1,
            'wave_number': self.k,
            'gamma': 0.0,  # standard method 不使用 penalty
            'solver': 'direct',
            'method': 'standard'
        }
        
        # 导入 HelmholtzLFEMModel
        from fealpy.fem import HelmholtzLFEMModel
        
        # 创建 FEM 模型并运行
        fem_model = HelmholtzLFEMModel(fem_options)
        
        t0 = time.time()
        uh, _ = fem_model.run(plot=False)
        self.fem_solve_time = time.time() - t0
        
        # # 保存 FEM 结果
        # self.fem_solution = uh
        # self.fem_space = fem_model.space
        # self.fem_func = fem_model.space.function(uh)
        
        return uh
    

    def run(self):
        """Execute training process.
        
        Notes:
            Training workflow:
            1. Sample collocation points (domain + boundary)
            2. Compute PDE and boundary residuals
            3. Separate real/imaginary loss computation (when complex=True)
            4. Backpropagation and parameter update
            5. Periodic error evaluation
            
            Complex-valued solution handling:
            - Separate computation of real/imaginary PDE and boundary residuals
            - Total loss = real_PDE_loss + imag_PDE_loss + real_BC_loss + imag_BC_loss
        """
        tmr = timer()
        next(tmr)
        # sampler
        sampler_pde = ISampler(self.domain, requires_grad=True, mode=self.options['sampling_mode'])
        sampler_bc = BoxBoundarySampler(self.domain, requires_grad=True, mode=self.options['sampling_mode'])
        mse = nn.MSELoss(reduction='mean')

        self.Loss = []
        self.error_real= []
        self.error_imag = []
        w = self.weights
        mesh = self.mesh

        train_start = time.time()

        for epoch in range(self.epochs+1):
            train_start_epoch = time.time()
            self.optimizer.zero_grad()

            # 采样点
            if (self.options['sampling_mode'] == 'linspace') :
                if epoch == 0:
                    ''' 均匀采样只采一次 '''
                    spde = sampler_pde.run(self.npde)
                    sbc = sampler_bc.run(self.nbc)
            else:
                spde = sampler_pde.run(self.npde)
                sbc = sampler_bc.run(self.nbc)

            # 计算残差与损失
            pde_res = self.pde_residual(spde)
            bc_res = self.bc_residual(sbc)
            pde_r = bm.real(pde_res)
            bc_r = bm.real(bc_res)
            mse_pde_r = mse(pde_r, bm.zeros_like(pde_r))
            mse_bc_r = mse(bc_r, bm.zeros_like(bc_r))

            if self.complex:
                pde_i =  bm.imag(pde_res)
                bc_i = bm.imag(bc_res)
                mse_pde_i = mse(pde_i, bm.zeros_like(pde_i))
                mse_bc_i = mse(bc_i, bm.zeros_like(bc_i))
                loss = w[0]* (mse_pde_r + mse_pde_i) + w[1] * (mse_bc_r + mse_bc_i)
            else:
                loss = w[0] * mse_pde_r + w[1]* mse_bc_r

            loss.backward()            
            self.optimizer.step()  
            if self.steplr is not None:
                self.steplr.step()
            train_end_epoch = time.time()
            if epoch % 100 == 0:
                error = self.net.estimate_error(self.pde.solution, mesh, coordtype='c', compare='real')
                self.error_real.append(error.detach().numpy())
                self.Loss.append(loss.item())
                time_epoch = train_end_epoch - train_start_epoch
                self.logger.info(f"epoch: {epoch}, Loss: {loss.item():.6e}, Time: {time_epoch:.2f}s")  

                if self.complex:
                    error_i = self.net.estimate_error(self.pde.solution, mesh, coordtype='c', compare='imag')
                    self.error_imag.append(error_i.detach().numpy()) 
        train_end = time.time()
        self.train_total_time = train_end - train_start
        self.train_avg_time = self.train_total_time / (self.epochs + 1)
        # self.logger.info(f"PINN Total training time: {self.train_total_time:.2f}s, Average time per epoch: {self.train_avg_time:.2f}s")
        tmr.send(f'PINN training time')
        next(tmr)

    def predict(self, p: TensorLike) -> TensorLike:
        """Make predictions using trained network.
        
        Parameters:
            p(TensorLike): Input point coordinates.

        Returns:
            TensorLike: Network predictions (complex tensor when complex=True).
        """
        return self.net(p)
    
    def show(self):
        """Visualize training loss, error curves, and compare PINN with FEM and exact solution.
        
        Saves figures and result statistics to a subfolder named with timestamp
        in the current working directory.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from datetime import datetime
        import numpy as np
        import os

        base_dir = r"D:\chen\fealpy\example\ml\Helmholtz_result"
        time_str = datetime.now().strftime("%m%d%H%M")
        folder_name = os.path.join(base_dir, f"helmholtz_result_{time_str}")
        os.makedirs(folder_name, exist_ok=True)

        # 所有文件路径加上子文件夹前缀
        result_file = os.path.join(folder_name, f"helmholtz_result_{time_str}.txt")
        loss_file   = os.path.join(folder_name, f"helmholtz_loss_{time_str}.png")
        comp_file   = os.path.join(folder_name, f"helmholtz_comparison_{time_str}.png")
        err_file    = os.path.join(folder_name, f"helmholtz_error_{time_str}.png")

        # ================== 1. 训练损失与误差曲线 ==================
        fig1, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))
        Loss = bm.log10(bm.tensor(self.Loss)).numpy()

        axes[0].plot(Loss, 'r-', linewidth=2)
        axes[0].set_title('Training Loss', fontsize=12)
        axes[0].set_xlabel('training epochs*100', fontsize=10)
        axes[0].set_ylabel('log10(Loss)', fontsize=10)
        axes[0].grid(True)

        error_real = bm.log10(bm.tensor(self.error_real)).numpy()
        error_imag = bm.log10(bm.tensor(self.error_imag)).numpy() if self.error_imag else None
        axes[1].plot(error_real, 'b-', linewidth=2, label='Real Part Error')
        if error_imag is not None:
            axes[1].plot(error_imag, 'g--', linewidth=2, label='Imag Part Error')
        axes[1].set_title('L2 Error between PINN and Exact', fontsize=12)
        axes[1].set_ylabel('log10(Error)', fontsize=10)
        axes[1].set_xlabel('training epochs*100', fontsize=10)
        axes[1].grid(True)
        axes[1].legend()
        fig1.tight_layout()
        fig1.savefig(loss_file)
        plt.close(fig1)


        # ================== 在网格节点上计算各解 ==================
        node = self.mesh.entity('node')
        t0 = time.time()
        self.net.eval()
        u_pinn = self.net(node).detach()          # 形状 (N, ) 或 (N, 1)
        if u_pinn.ndim > 1:
            u_pinn = u_pinn.flatten()
        pinn_predict_time = time.time() - t0

        u_exact = self.pde.solution(node)         # 真解
        if u_exact.ndim > 1:
            u_exact = u_exact.flatten()
        u_fem = self.fem()                 # FEM 解（节点值）
        if u_fem.ndim > 1:
            u_fem = u_fem.flatten()

        # 计算三种误差（复数值）
        error_pinn_fem = u_pinn - u_fem
        error_pinn_exact = u_pinn - u_exact
        error_fem_exact = u_fem - u_exact

        # 计算误差统计量（分别对实部和虚部）
        def compute_stats(err):
            err = err.detach().numpy()
            real = err.real
            imag = err.imag
            stats = {
                'real_mae': np.mean(np.abs(real)),
                'real_max': np.max(np.abs(real)),
                'real_rmse': np.sqrt(np.mean(real**2)),
                'imag_mae': np.mean(np.abs(imag)),
                'imag_max': np.max(np.abs(imag)),
                'imag_rmse': np.sqrt(np.mean(imag**2)),
                'complex_mae': np.mean(np.abs(err)),
                'complex_max': np.max(np.abs(err)),
                'complex_rmse': np.sqrt(np.mean(np.abs(err)**2)),
            }
            return stats

        stats_pinn_fem = compute_stats(error_pinn_fem)
        stats_pinn_exact = compute_stats(error_pinn_exact)
        stats_fem_exact = compute_stats(error_fem_exact)
        para = sum(p.numel() for p in self.net.parameters())

        # ================== 4. 打印时间信息 ==================
        print("\n========== Timing ==========")
        print(f"FEM solve time          : {self.fem_solve_time:.6f} s")
        print(f"PINN prediction time    : {pinn_predict_time:.6f} s")
        print(f"PINN training total time: {self.train_total_time:.2f} s")
        print(f"PINN training avg/step  : {self.train_avg_time:.6f} s")
        print(f"PINN parameters: {para}")
        print(f"Mesh Node:{node.shape[0]}")
        print("============================\n")

        # ================== 5. 绘制解对比图 ==================
        node_np = node.detach().numpy()
        u_pinn_np = u_pinn.detach().numpy()
        u_fem_np = u_fem.detach().numpy()
        u_exact_np = u_exact.detach().numpy()

        if self.gd == 2:
            x = node_np[:, 0]
            y = node_np[:, 1]
            fig2 = plt.figure(figsize=(15, 10))
            if self.complex:
                nrows, ncols = 2, 3
            else:
                nrows, ncols = 1, 3

            # 实部
            ax1 = fig2.add_subplot(nrows, ncols, 1, projection='3d')
            ax1.plot_trisurf(x, y, u_pinn_np.real, cmap='viridis', linewidth=0.2)
            ax1.set_title('PINN (Real)')
            ax1.set_xlabel('x'); ax1.set_ylabel('y')

            ax2 = fig2.add_subplot(nrows, ncols, 2, projection='3d')
            ax2.plot_trisurf(x, y, u_fem_np.real, cmap='plasma', linewidth=0.2)
            ax2.set_title('FEM (Real)')
            ax2.set_xlabel('x'); ax2.set_ylabel('y')

            ax3 = fig2.add_subplot(nrows, ncols, 3, projection='3d')
            ax3.plot_trisurf(x, y, u_exact_np.real, cmap='coolwarm', linewidth=0.2)
            ax3.set_title('Exact (Real)')
            ax3.set_xlabel('x'); ax3.set_ylabel('y')

            if self.complex:
                ax4 = fig2.add_subplot(nrows, ncols, 4, projection='3d')
                ax4.plot_trisurf(x, y, u_pinn_np.imag, cmap='viridis', linewidth=0.2)
                ax4.set_title('PINN (Imag)')
                ax4.set_xlabel('x'); ax4.set_ylabel('y')

                ax5 = fig2.add_subplot(nrows, ncols, 5, projection='3d')
                ax5.plot_trisurf(x, y, u_fem_np.imag, cmap='plasma', linewidth=0.2)
                ax5.set_title('FEM (Imag)')
                ax5.set_xlabel('x'); ax5.set_ylabel('y')

                ax6 = fig2.add_subplot(nrows, ncols, 6, projection='3d')
                ax6.plot_trisurf(x, y, u_exact_np.imag, cmap='coolwarm', linewidth=0.2)
                ax6.set_title('Exact (Imag)')
                ax6.set_xlabel('x'); ax6.set_ylabel('y')

            fig2.tight_layout()
            fig2.savefig(comp_file)
            plt.close(fig2)

        elif self.gd == 1:
            fig2 = plt.figure(figsize=(10, 6))
            plt.plot(node_np, u_pinn_np.real, 'b-', label='PINN Real')
            plt.plot(node_np, u_fem_np.real, 'g--', label='FEM Real')
            plt.plot(node_np, u_exact_np.real, 'r-.', label='Exact Real')
            if self.complex:
                plt.plot(node_np, u_pinn_np.imag, 'c-', label='PINN Imag')
                plt.plot(node_np, u_fem_np.imag, 'm--', label='FEM Imag')
                plt.plot(node_np, u_exact_np.imag, 'y-.', label='Exact Imag')
            plt.xlabel('x')
            plt.ylabel('u(x)')
            plt.title('1D Solution Comparison')
            plt.legend()
            plt.grid(True)
            fig2.tight_layout()
            fig2.savefig(comp_file)
            plt.close(fig2)

        # ================== 6. 绘制误差分布图 ==================
        fig3 = plt.figure(figsize=(15, 10))
        if self.gd == 2:
            if self.complex:
                # 实部误差：PINN - Exact
                ax1 = fig3.add_subplot(2, 3, 1, projection='3d')
                ax1.plot_trisurf(x, y, error_pinn_exact.real, cmap='RdBu_r', linewidth=0.2)
                ax1.set_title('PINN-Exact (Real)')
                # 虚部
                ax2 = fig3.add_subplot(2, 3, 2, projection='3d')
                ax2.plot_trisurf(x, y, error_pinn_exact.imag, cmap='RdBu_r', linewidth=0.2)
                ax2.set_title('PINN-Exact (Imag)')
                # 模
                ax3 = fig3.add_subplot(2, 3, 3, projection='3d')
                ax3.plot_trisurf(x, y, np.abs(error_pinn_exact), cmap='hot', linewidth=0.2)
                ax3.set_title('|PINN-Exact|')

                # FEM-Exact
                ax4 = fig3.add_subplot(2, 3, 4, projection='3d')
                ax4.plot_trisurf(x, y, error_fem_exact.real, cmap='RdBu_r', linewidth=0.2)
                ax4.set_title('FEM-Exact (Real)')
                ax5 = fig3.add_subplot(2, 3, 5, projection='3d')
                ax5.plot_trisurf(x, y, error_fem_exact.imag, cmap='RdBu_r', linewidth=0.2)
                ax5.set_title('FEM-Exact (Imag)')
                ax6 = fig3.add_subplot(2, 3, 6, projection='3d')
                ax6.plot_trisurf(x, y, np.abs(error_fem_exact), cmap='hot', linewidth=0.2)
                ax6.set_title('|FEM-Exact|')
            else:
                ax1 = fig3.add_subplot(1, 2, 1, projection='3d')
                ax1.plot_trisurf(x, y, error_pinn_exact, cmap='RdBu_r', linewidth=0.2)
                ax1.set_title('PINN-Exact')
                ax2 = fig3.add_subplot(1, 2, 2, projection='3d')
                ax2.plot_trisurf(x, y, error_fem_exact, cmap='RdBu_r', linewidth=0.2)
                ax2.set_title('FEM-Exact')
        else:
            # 1D 误差曲线
            plt.plot(node_np, error_pinn_exact.real, 'b-', label='PINN-Exact Real')
            if self.complex:
                plt.plot(node_np, error_pinn_exact.imag, 'b--', label='PINN-Exact Imag')
            plt.plot(node_np, error_fem_exact.real, 'r-', label='FEM-Exact Real')
            if self.complex:
                plt.plot(node_np, error_fem_exact.imag, 'r--', label='FEM-Exact Imag')
            plt.xlabel('x')
            plt.ylabel('Error')
            plt.legend()
            plt.grid(True)
        fig3.tight_layout()
        fig3.savefig(err_file)
        plt.close(fig3)

        # ================== 7. 保存结果到文本文件（包含 options 和误差统计） ==================
        with open(result_file, 'w') as f:
            f.write("========== Helmholtz PINN Results ==========\n")
            f.write("Options:\n")
            for key, val in self.options.items():
                f.write(f"  {key}: {val}\n")
            f.write("\n--- Error Statistics (based on nodal values) ---\n")
            f.write("\nError: PINN - FEM\n")
            f.write(f"  Real MAE  : {stats_pinn_fem['real_mae']:.6e}\n")
            f.write(f"  Real Max  : {stats_pinn_fem['real_max']:.6e}\n")
            f.write(f"  Real RMSE : {stats_pinn_fem['real_rmse']:.6e}\n")
            if self.complex:
                f.write(f"  Imag MAE  : {stats_pinn_fem['imag_mae']:.6e}\n")
                f.write(f"  Imag Max  : {stats_pinn_fem['imag_max']:.6e}\n")
                f.write(f"  Imag RMSE : {stats_pinn_fem['imag_rmse']:.6e}\n")
                f.write(f"  Complex MAE : {stats_pinn_fem['complex_mae']:.6e}\n")
                f.write(f"  Complex Max : {stats_pinn_fem['complex_max']:.6e}\n")
                f.write(f"  Complex RMSE: {stats_pinn_fem['complex_rmse']:.6e}\n")

            f.write("\nError: PINN - Exact\n")
            f.write(f"  Real MAE  : {stats_pinn_exact['real_mae']:.6e}\n")
            f.write(f"  Real Max  : {stats_pinn_exact['real_max']:.6e}\n")
            f.write(f"  Real RMSE : {stats_pinn_exact['real_rmse']:.6e}\n")
            if self.complex:
                f.write(f"  Imag MAE  : {stats_pinn_exact['imag_mae']:.6e}\n")
                f.write(f"  Imag Max  : {stats_pinn_exact['imag_max']:.6e}\n")
                f.write(f"  Imag RMSE : {stats_pinn_exact['imag_rmse']:.6e}\n")
                f.write(f"  Complex MAE : {stats_pinn_exact['complex_mae']:.6e}\n")
                f.write(f"  Complex Max : {stats_pinn_exact['complex_max']:.6e}\n")
                f.write(f"  Complex RMSE: {stats_pinn_exact['complex_rmse']:.6e}\n")

            f.write("\nError: FEM - Exact\n")
            f.write(f"  Real MAE  : {stats_fem_exact['real_mae']:.6e}\n")
            f.write(f"  Real Max  : {stats_fem_exact['real_max']:.6e}\n")
            f.write(f"  Real RMSE : {stats_fem_exact['real_rmse']:.6e}\n")
            if self.complex:
                f.write(f"  Imag MAE  : {stats_fem_exact['imag_mae']:.6e}\n")
                f.write(f"  Imag Max  : {stats_fem_exact['imag_max']:.6e}\n")
                f.write(f"  Imag RMSE : {stats_fem_exact['imag_rmse']:.6e}\n")
                f.write(f"  Complex MAE : {stats_fem_exact['complex_mae']:.6e}\n")
                f.write(f"  Complex Max : {stats_fem_exact['complex_max']:.6e}\n")
                f.write(f"  Complex RMSE: {stats_fem_exact['complex_rmse']:.6e}\n")

            f.write("\n--- Timing ---\n")
            f.write(f"FEM solve time           : {self.fem_solve_time:.6f} s\n")
            f.write(f"PINN prediction time     : {pinn_predict_time:.6f} s\n")
            f.write(f"PINN training total time : {self.train_total_time:.2f} s\n")
            f.write(f"PINN training avg/step   : {self.train_avg_time:.6f} s\n")
            f.write(f"PINN parameters: {para} \n")
            f.write(f"Number mesh Node:{node.shape[0]}\n")
            f.write("============================================\n")

        print(f"Results saved to folder: {folder_name}")
        print(f"  - {result_file}")
        print(f"  - {loss_file}")
        print(f"  - {comp_file}")
        print(f"  - {err_file}")
