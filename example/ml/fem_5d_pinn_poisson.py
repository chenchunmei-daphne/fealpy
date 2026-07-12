"""
5D Poisson方程 PINN 与 FEM 对比求解器
=====================================
核心设计：所有比较（PINN vs 真解、FEM vs 真解、PINN vs FEM）
都使用完全相同的测试点集（8^5 = 32768 个点）
"""

import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from typing import Union, Optional
import time
import os
import sys
from datetime import datetime
import numpy as np
from scipy.sparse import csr_matrix, kron, eye
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from fealpy.backend import bm
from fealpy.utils import timer
from fealpy.typing import TensorLike
from fealpy.model import ComputationalModel, PDEModelManager
from fealpy.model.poisson import PoissonPDEDataT

from fealpy.ml import gradient, optimizers, activations

from fealpy.ml.modules import Solution
from fealpy.ml.sampler import ISampler, BoxBoundarySampler


class Poisson5DFEMSolver:
    """
    5D Poisson方程求解器（使用张量积方法）
    支持在任意测试点上计算误差（通过插值）
    """
    
    def __init__(self, pde, mesh_size=10):
        """
        参数:
            pde: PDE对象
            mesh_size: 每维的节点数，默认10
        """
        self.pde = pde
        self.n = mesh_size
        self.gd = pde.geo_dimension()
        self.domain = pde.domain()
        
        print(f"5D FEM Solver initialized with mesh_size={mesh_size}")
        print(f"Total nodes: {self.n}^{self.gd} = {self.n**self.gd}")
        print(f"Total elements: {self.n-1}^{self.gd} = {(self.n-1)**self.gd}")
        
        # 计算步长
        self.h = [(self.domain[2*i+1] - self.domain[2*i]) / (self.n - 1) 
                  for i in range(self.gd)]
        
        # 构建1D刚度矩阵和质量矩阵
        self.K1, self.M1 = self._build_1d_matrices()
        
        # 构建节点坐标
        self.nodes = self._build_nodes()
        
        # 标记边界节点
        self.boundary_nodes = self._mark_boundary_nodes()
        
        # 构建KDTree用于插值
        self._build_tree()
        
    def _build_1d_matrices(self):
        """构建1D刚度矩阵和质量矩阵（线性元）"""
        n = self.n
        h = self.h[0]  # 假设均匀网格
        
        # 刚度矩阵（三对角）
        K = csr_matrix((n, n))
        K.setdiag(2/h * np.ones(n))
        K.setdiag(-1/h * np.ones(n-1), 1)
        K.setdiag(-1/h * np.ones(n-1), -1)
        
        # 质量矩阵（三对角）
        M = csr_matrix((n, n))
        M.setdiag(2*h/6 * np.ones(n))
        M.setdiag(h/6 * np.ones(n-1), 1)
        M.setdiag(h/6 * np.ones(n-1), -1)
        
        return K, M
    
    def _build_nodes(self):
        """构建5D节点坐标"""
        points_per_dim = [bm.linspace(self.domain[2*i], self.domain[2*i+1], self.n) 
                         for i in range(self.gd)]
        
        meshgrid = bm.meshgrid(*points_per_dim, indexing='ij')
        nodes = bm.stack([grid.flatten() for grid in meshgrid], axis=-1)
        
        return nodes
    
    def _mark_boundary_nodes(self):
        """标记边界节点"""
        atol = 1e-12
        boundary_nodes = bm.zeros(self.nodes.shape[0], dtype=bool)
        
        for i in range(self.gd):
            coord = self.nodes[:, i]
            boundary_nodes |= (bm.abs(coord - self.domain[2*i]) < atol)
            boundary_nodes |= (bm.abs(coord - self.domain[2*i+1]) < atol)
        
        return boundary_nodes
    
    def _build_tree(self):
        """构建KDTree用于快速插值"""
        if hasattr(self.nodes, 'numpy'):
            nodes_np = self.nodes.numpy()
        else:
            nodes_np = np.array(self.nodes)
        self.tree = cKDTree(nodes_np)
    
    def _build_tensor_product_matrix(self):
        """构建5D张量积矩阵 A = K⊗M⊗...⊗M + M⊗K⊗M⊗...⊗M + ..."""
        print("Building 5D tensor product matrix...")
        start_time = time.time()
        
        A = csr_matrix((self.n**self.gd, self.n**self.gd))
        
        for d in range(self.gd):
            term = self.K1
            for i in range(self.gd):
                if i < d:
                    term = kron(self.M1, term)
                elif i > d:
                    term = kron(term, self.M1)
            A += term
        
        print(f"Matrix assembly time: {time.time() - start_time:.4f}s")
        print(f"Matrix nonzeros: {A.nnz}")
        print(f"Matrix density: {A.nnz / (A.shape[0]**2):.2e}")
        
        return A
    
    def _build_mass_matrix(self):
        """构建5D质量矩阵 M = M1⊗M1⊗...⊗M1"""
        M_total = self.M1
        for _ in range(self.gd - 1):
            M_total = kron(M_total, self.M1)
        return M_total
    
    def interpolate(self, p):
        """
        在任意点插值FEM解（使用最近邻插值）
        
        参数:
            p: 查询点，形状为 (n_points, gd)
            
        返回:
            插值后的解，形状为 (n_points,)
        """
        if hasattr(p, 'numpy'):
            p_np = p.numpy()
        else:
            p_np = np.array(p)
        
        if p_np.ndim == 1:
            p_np = p_np.reshape(1, -1)
        
        distances, indices = self.tree.query(p_np, k=1)
        return self.uh[indices]
    
    def solve(self, test_points: Optional[TensorLike] = None):
        """
        求解5D泊松方程
        
        参数:
            test_points: 测试点集，用于计算FEM误差（通过插值）
        """
        print("\n" + "="*60)
        print("5D FEM SOLVER")
        print("="*60)
        
        # 构建张量积矩阵
        A = self._build_tensor_product_matrix()
        
        # 构建右端项
        print("Building RHS...")
        start_time = time.time()
        
        f = self.pde.source(self.nodes).flatten()
        M_total = self._build_mass_matrix()
        F = M_total @ f
        
        print(f"RHS assembly time: {time.time() - start_time:.4f}s")
        
        # 应用边界条件
        print("Applying boundary conditions...")
        A_mod = A.copy()
        F_mod = F.copy()
        
        boundary_indices = np.where(self.boundary_nodes)[0]
        g = self.pde.dirichlet(self.nodes[self.boundary_nodes]).flatten()
        
        for idx, bidx in enumerate(boundary_indices):
            A_mod[bidx, :] = 0
            A_mod[bidx, bidx] = 1
            F_mod[bidx] = g[idx]
        
        print(f"Boundary nodes: {len(boundary_indices)}")
        print(f"Interior nodes: {self.nodes.shape[0] - len(boundary_indices)}")
        
        # 求解线性系统
        print("Solving linear system...")
        solve_start = time.time()
        uh = spsolve(A_mod, F_mod)
        solve_time = time.time() - solve_start
        print(f"Solve time: {solve_time:.4f}s")
        
        self.uh = uh
        
        # ===== 计算FEM误差（使用测试点） =====
        self.error_stats = {'solve_time': solve_time}
        
        if hasattr(self.pde, 'solution') and test_points is not None:
            # 在测试点上插值FEM解
            uh_at_test = self.interpolate(test_points)
            u_true = self.pde.solution(test_points).flatten()
            
            diff = bm.tensor(uh_at_test, dtype=bm.float64) - u_true
            n_points = test_points.shape[0]
            
            # 计算L2误差（RMSE）
            l2_error_rmse = bm.sqrt(bm.mean(diff**2))
            
            # 计算体积加权的L2误差
            volume = 1.0
            for i in range(self.gd):
                volume *= (self.domain[2*i+1] - self.domain[2*i])
            dV = volume / n_points
            l2_error_integral = bm.sqrt(bm.sum(diff**2) * dV)
            
            # 相对误差
            u_true_norm = bm.sqrt(bm.sum(u_true**2) * dV)
            relative_l2_error = l2_error_integral / (u_true_norm + 1e-12)
            
            print(f"\nFEM Error Statistics (on {n_points} test points):")
            print(f"  - L2 Error (Integral): {float(l2_error_integral):.6e}")
            print(f"  - L2 Error (RMSE):     {float(l2_error_rmse):.6e}")
            print(f"  - Relative L2 Error:   {float(relative_l2_error):.6e}")
            
            self.error_stats.update({
                'l2_error': float(l2_error_integral),
                'l2_error_rmse': float(l2_error_rmse),
                'relative_l2_error': float(relative_l2_error),
                'n_test_points': n_points
            })
        
        print("="*60 + "\n")
        return uh
    
    def get_nodes(self):
        return self.nodes
    
    def get_solution(self):
        return self.uh


class PoissonPINNModel(ComputationalModel):
    """
    物理信息神经网络 (PINN) 模型
    所有比较使用完全相同的测试点集
    """
    
    def __init__(self, options: dict = {}):
        self.options = self.get_options()
        self.options.update(options)
        
        self.pbar_log = self.options['pbar_log']
        self.log_level = self.options['log_level']
        super().__init__(pbar_log=self.pbar_log, log_level=self.log_level)
    
        self.lr = self.options['lr']
        self.epochs = self.options['epochs']
        self.hidden_size = self.options['hidden_size']
        self.activation = activations[self.options['activation']]
        self.npde = self.options['npde']
        self.nbc = self.options['nbc']
        self.weights = self.options['weights']
        self.mesh_size = self.options.get('mesh_size', 10)
        self.tmr = timer()

        self.set_pde(self.options['pde'])
        self.set_network()
        
        self.epoch_times = []
        self.total_training_time = 0
        
        # 统计信息
        self.training_stats = {
            'pde_points': 0,
            'bc_points': 0,
            'total_points': 0,
        }
        # 测试点集（所有比较共用）
        self.test_points = None
        self.test_stats = {
            'total': 0,
            'interior': 0,
            'boundary': 0,
            'points_per_dim': 0
        }
        # 误差统计
        self.pinn_error = {}      # PINN vs 真解
        self.fem_error = {}       # FEM vs 真解
        self.pinn_vs_fem_error = {}  # PINN vs FEM
        
        # 结果保存路径
        self.result_dir = None
        self.log_file = None
        self.original_stdout = sys.stdout

    @classmethod
    def get_options(cls):
        import argparse

        parser = argparse.ArgumentParser(description="Poisson equation solver using PINN.")

        parser.add_argument('--pde', default=1, type=int,
                            help="Built-in PDE example ID, default is 1.")
        parser.add_argument('--mesh_size', default=10, type=int,
                            help='Mesh size for FEM solver, default is 10.')
        parser.add_argument('--sampling_mode', default='random', type=str,
                            help="Sampling method: 'random' or 'linspace', default is 'random'")
        parser.add_argument('--npde', default=400, type=int,
                            help='Number of PDE samples, default is 400.')
        parser.add_argument('--nbc', default=100, type=int,
                            help='Number of boundary samples, default is 100.')
        parser.add_argument('--weights', default=(1, 30), type=tuple,
                            help='Weights for PDE and BC loss, default is (1, 30).')
        parser.add_argument('--hidden_size', default=(64, 64, 32), type=tuple,
                            help='Hidden layer sizes, default is (64, 64, 32).')
        parser.add_argument('--optimizer', default="Adam", type=str,
                            help="Optimizer: 'Adam' or 'SGD', default is 'Adam'.")
        parser.add_argument('--activation', default="Tanh", type=str,
                            help="Activation function, default is Tanh.")
        parser.add_argument('--lr', default=0.001, type=float,
                            help='Learning rate, default is 0.001.')
        parser.add_argument('--step_size', default=0, type=int,
                            help='Learning rate decay step size, default is 0.')
        parser.add_argument('--gamma', default=0.99, type=float,
                            help='Learning rate decay factor, default is 0.99.')
        parser.add_argument('--epochs', default=2000, type=int,
                            help='Number of training epochs, default is 2000.')
        parser.add_argument('--pbar_log', default=True, type=bool,
                            help='Whether to show progress bar, default is True')
        parser.add_argument('--log_level', default='INFO', type=str,
                            help='Log level, default is INFO.')
        parser.add_argument('--test_points_per_dim', default=8, type=int,
                            help='Points per dimension for test set (total = per_dim^d), default is 8.')
        
        options = vars(parser.parse_args())
        return options
    
    def set_pde(self, pde: Union[PoissonPDEDataT, int]=1):
        if isinstance(pde, int):
            self.pde = PDEModelManager('poisson').get_example(pde)
        else:
            self.pde = pde 
        self.gd = self.pde.geo_dimension()
        self.domain = self.pde.domain()

    def set_network(self, net=None):
        if net is None:
            layers = []
            sizes = (self.gd,) + self.hidden_size + (1,)
            for i in range(len(sizes)-1):
                layers.append(nn.Linear(sizes[i], sizes[i+1], dtype=bm.float64))
                if i < len(sizes)-2:  
                    layers.append(self.activation())
            net = nn.Sequential(*layers)
        self.net = Solution(net)

        opt = optimizers[self.options.get('optimizer', 'Adam')]
        self.optimizer = opt(params=self.net.parameters(), lr=self.lr)

        step_size = self.options.get('step_size', 0)
        gamma = self.options.get('gamma', 0.99)
        self.set_steplr(step_size, gamma)

    def set_steplr(self, step_size: int=0, gamma: float=0.9):
        if step_size == 0:
            self.steplr = None
        else:
            self.steplr = StepLR(self.optimizer, step_size, gamma)

    def _ensure_dtype(self, tensor: TensorLike) -> TensorLike:
        if bm.backend_name == 'pytorch':
            import torch
            if tensor.dtype != torch.float64:
                tensor = tensor.to(dtype=torch.float64)
        return tensor

    def _is_on_boundary(self, p: TensorLike, atol: float = 1e-12) -> TensorLike:
        if hasattr(self.pde, 'is_dirichlet_boundary'):
            return self.pde.is_dirichlet_boundary(p)
        else:
            on_boundary = bm.zeros(p.shape[0], dtype=bool)
            for i in range(self.gd):
                coord = p[:, i]
                on_boundary = on_boundary | (bm.abs(coord - self.domain[2*i]) < atol)
                on_boundary = on_boundary | (bm.abs(coord - self.domain[2*i+1]) < atol)
            return on_boundary

    def generate_test_points(self, n_points_per_dim: int = None) -> TensorLike:
        """
        生成统一的测试点集（完整网格）
        所有比较（PINN vs 真解、FEM vs 真解、PINN vs FEM）
        都使用这个点集
        """
        if n_points_per_dim is None:
            n_points_per_dim = self.options.get('test_points_per_dim', 8)
        
        points_per_dim = [
            bm.linspace(self.domain[2*i], self.domain[2*i+1], n_points_per_dim, dtype=bm.float64)
            for i in range(self.gd)
        ]
        meshgrid = bm.meshgrid(*points_per_dim, indexing='ij')
        p = bm.stack([grid.flatten() for grid in meshgrid], axis=-1)
        
        # 统计信息
        on_boundary = self._is_on_boundary(p)
        self.test_stats = {
            'total': p.shape[0],
            'interior': int(bm.sum(~on_boundary).item()),
            'boundary': int(bm.sum(on_boundary).item()),
            'points_per_dim': n_points_per_dim
        }
        
        self.test_points = p
        return p

    def pde_residual(self, p: TensorLike) -> TensorLike:
        p = self._ensure_dtype(p)
        u = self.net(p)
        f = self.pde.source(p)
        f = self._ensure_dtype(f)

        grad_u = gradient(u, p, create_graph=True)
        laplacian = bm.zeros(u.shape[0], dtype=bm.float64)    
        
        for i in range(p.shape[-1]):
            u_ii = gradient(grad_u[..., i], p, create_graph=True, split=True)[i]
            laplacian += u_ii.flatten()

        return laplacian + f

    def bc_residual(self, p: TensorLike) -> TensorLike:
        p = self._ensure_dtype(p)
        u = self.net(p).flatten()
        bc = self.pde.dirichlet(p)
        bc = self._ensure_dtype(bc)
        return u - bc

    def _random_uniform(self, low: float, high: float, size: tuple) -> TensorLike:
        if bm.backend_name == 'pytorch':
            import torch
            return torch.rand(size, dtype=torch.float64) * (high - low) + low
        elif bm.backend_name == 'jax':
            return bm.random.uniform(low, high, size, dtype=bm.float64)
        else:
            return bm.random.uniform(low, high, size).astype(bm.float64)

    def _random_sample_domain(self, n: int) -> TensorLike:
        p = bm.zeros((n, self.gd), dtype=bm.float64)
        for i in range(self.gd):
            p[:, i] = self._random_uniform(self.domain[2*i], self.domain[2*i+1], (n,))
        if bm.backend_name == 'pytorch':
            p.requires_grad_(True)
        return p

    def _random_sample_boundary(self, n: int) -> TensorLike:
        n_per_boundary = max(1, n // (2 * self.gd))
        points_list = []
        
        for dim in range(self.gd):
            for side in [0, 1]:
                p = bm.zeros((n_per_boundary, self.gd), dtype=bm.float64)
                for i in range(self.gd):
                    if i == dim:
                        p[:, i] = self.domain[2*i + side]
                    else:
                        p[:, i] = self._random_uniform(self.domain[2*i], self.domain[2*i+1], (n_per_boundary,))
                points_list.append(p)
        
        all_points = bm.concat(points_list, axis=0)
        if all_points.shape[0] > n:
            if bm.backend_name == 'pytorch':
                import torch
                indices = torch.randperm(all_points.shape[0])[:n]
            else:
                indices = bm.random.choice(all_points.shape[0], n, replace=False)
            all_points = all_points[indices]
        
        if bm.backend_name == 'pytorch':
            all_points.requires_grad_(True)
        return all_points

    def run_fem(self):
        """运行FEM求解器，使用统一的测试点计算误差"""
        print("\n" + "="*60)
        print("Running 5D FEM Solver...")
        print("="*60)
        
        fem_solver = Poisson5DFEMSolver(self.pde, mesh_size=self.mesh_size)
        uh = fem_solver.solve(test_points=self.test_points)
        
        self.fem_solver = fem_solver
        self.fem_solution = uh
        self.fem_error = fem_solver.error_stats if hasattr(fem_solver, 'error_stats') else {}
        
        return uh

    def compute_all_errors(self):
        """
        在统一的测试点上计算所有误差：
        1. PINN vs 真解
        2. FEM vs 真解（已由FEM求解器计算）
        3. PINN vs FEM
        """
        print("\n" + "="*60)
        print("Computing All Errors on Unified Test Points...")
        print("="*60)
        
        if self.test_points is None:
            raise ValueError("Test points not generated. Call generate_test_points() first.")
        
        p = self.test_points
        n_points = p.shape[0]
        
        # 计算体积
        volume = 1.0
        for i in range(self.gd):
            volume *= (self.domain[2*i+1] - self.domain[2*i])
        dV = volume / n_points
        
        # ===== 1. PINN vs 真解 =====
        u_pinn = self.net(p).flatten()
        u_true = self.pde.solution(p).flatten()
        
        diff_pinn = u_pinn - u_true
        l2_pinn = bm.sqrt(bm.mean(diff_pinn**2))
        l2_pinn_integral = bm.sqrt(bm.sum(diff_pinn**2) * dV)
        
        self.pinn_error = {
            'l2_error_rmse': float(l2_pinn),
            'l2_error_integral': float(l2_pinn_integral),
            'max_error': float(bm.max(bm.abs(diff_pinn))),
            'mean_abs_error': float(bm.mean(bm.abs(diff_pinn)))
        }
        
        print(f"\n[1] PINN vs Exact Solution (on {n_points} test points):")
        print(f"  - L2 Error (RMSE):     {l2_pinn:.6e}")
        print(f"  - L2 Error (Integral): {l2_pinn_integral:.6e}")
        print(f"  - Max Error:           {bm.max(bm.abs(diff_pinn)):.6e}")
        print(f"  - Mean Abs Error:      {bm.mean(bm.abs(diff_pinn)):.6e}")
        
        # ===== 2. FEM vs 真解 =====
        # 已由FEM求解器计算并存储在 self.fem_error 中
        print(f"\n[2] FEM vs Exact Solution (on {n_points} test points):")
        if 'l2_error_rmse' in self.fem_error:
            print(f"  - L2 Error (RMSE):     {self.fem_error['l2_error_rmse']:.6e}")
            print(f"  - L2 Error (Integral): {self.fem_error['l2_error']:.6e}")
            print(f"  - Relative L2 Error:   {self.fem_error.get('relative_l2_error', 0):.6e}")
        else:
            print("  - FEM error statistics not available.")
        
        # ===== 3. PINN vs FEM =====
        # 在测试点上插值FEM解
        uh_fem = self.fem_solver.interpolate(p)
        diff_pinn_fem = u_pinn - bm.tensor(uh_fem, dtype=bm.float64)
        
        l2_pinn_fem = bm.sqrt(bm.mean(diff_pinn_fem**2))
        l2_pinn_fem_integral = bm.sqrt(bm.sum(diff_pinn_fem**2) * dV)
        
        self.pinn_vs_fem_error = {
            'l2_error_rmse': float(l2_pinn_fem),
            'l2_error_integral': float(l2_pinn_fem_integral),
            'max_error': float(bm.max(bm.abs(diff_pinn_fem))),
            'mean_abs_error': float(bm.mean(bm.abs(diff_pinn_fem)))
        }
        
        print(f"\n[3] PINN vs FEM (on {n_points} test points):")
        print(f"  - L2 Error (RMSE):     {l2_pinn_fem:.6e}")
        print(f"  - L2 Error (Integral): {l2_pinn_fem_integral:.6e}")
        print(f"  - Max Error:           {bm.max(bm.abs(diff_pinn_fem)):.6e}")
        print(f"  - Mean Abs Error:      {bm.mean(bm.abs(diff_pinn_fem)):.6e}")
        
        print("="*60 + "\n")
        
        return {
            'pinn_vs_exact': self.pinn_error,
            'fem_vs_exact': self.fem_error,
            'pinn_vs_fem': self.pinn_vs_fem_error
        }

    def _setup_result_dir(self):
        base_dir = r"D:\chen\fealpy\example\ml\Possion_result"
        timestamp = datetime.now().strftime("%m%d%H%M")
        pde_id = self.options.get('pde', 1)
        self.result_dir = os.path.join(base_dir, f"pde{pde_id}_{timestamp}")
        
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        
        self.log_file = os.path.join(self.result_dir, f"poisson_result_{timestamp}.txt")
        self.file_handle = open(self.log_file, 'w', encoding='utf-8')
        sys.stdout = self.file_handle

    def _restore_stdout(self):
        if hasattr(self, 'file_handle') and self.file_handle:
            sys.stdout = self.original_stdout
            self.file_handle.close()

    def _save_figures(self):
        import matplotlib.pyplot as plt
        
        if self.Loss:
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
            Loss = bm.log10(bm.tensor(self.Loss)).numpy()
            axes.plot(Loss, 'r-', linewidth=2)
            axes.set_title('PINN Training Loss', fontsize=12)
            axes.set_xlabel('training epochs (x100)', fontsize=10)
            axes.set_ylabel('log10(Loss)', fontsize=10)
            axes.grid(True)
            fig.savefig(os.path.join(self.result_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)

    def _print_stats(self):
        print(f"\n{'='*60}")
        para = sum(p.numel() for p in self.net.parameters())
        print(f'Model parameters: {para}')
        print(f"{'='*60}")
        
        print(f"TRAINING STATISTICS")
        print(f"{'='*60}")
        print(f"  - PDE points:          {self.training_stats['pde_points']}")
        print(f"  - BC points:           {self.training_stats['bc_points']}")
        print(f"  - Total training:      {self.training_stats['total_points']}")
        print(f"  - Total time:          {self.total_training_time:.4f}s")
        print(f"  - Time per epoch:      {self.total_training_time / max(1, len(self.epoch_times)):.4f}s")
        print(f"  - Epochs:              {len(self.epoch_times)}")
        
        print(f"\n{'='*60}")
        print(f"UNIFIED TEST POINTS (used for all comparisons)")
        print(f"{'='*60}")
        print(f"  - Points per dim:      {self.test_stats['points_per_dim']}")
        print(f"  - Total points:        {self.test_stats['total']} = {self.test_stats['points_per_dim']}^{self.gd}")
        print(f"  - Interior:            {self.test_stats['interior']}")
        print(f"  - Boundary:            {self.test_stats['boundary']}")
        print(f"  - Interior ratio:      {self.test_stats['interior'] / self.test_stats['total']:.2%}")
        print(f"  - Boundary ratio:      {self.test_stats['boundary'] / self.test_stats['total']:.2%}")
        
        print(f"\n{'='*60}")
        print(f"ERROR COMPARISON (same {self.test_stats['total']} test points)")
        print(f"{'='*60}")
        
        print(f"\n[1] PINN vs Exact Solution:")
        print(f"  - L2 Error (RMSE):     {self.pinn_error.get('l2_error_rmse', 0):.6e}")
        print(f"  - L2 Error (Integral): {self.pinn_error.get('l2_error_integral', 0):.6e}")
        print(f"  - Max Error:           {self.pinn_error.get('max_error', 0):.6e}")
        print(f"  - Mean Abs Error:      {self.pinn_error.get('mean_abs_error', 0):.6e}")
        
        print(f"\n[2] FEM vs Exact Solution:")
        print(f"  - L2 Error (RMSE):     {self.fem_error.get('l2_error_rmse', 0):.6e}")
        print(f"  - L2 Error (Integral): {self.fem_error.get('l2_error', 0):.6e}")
        print(f"  - Relative L2 Error:   {self.fem_error.get('relative_l2_error', 0):.6e}")
        
        print(f"\n[3] PINN vs FEM:")
        print(f"  - L2 Error (RMSE):     {self.pinn_vs_fem_error.get('l2_error_rmse', 0):.6e}")
        print(f"  - L2 Error (Integral): {self.pinn_vs_fem_error.get('l2_error_integral', 0):.6e}")
        print(f"  - Max Error:           {self.pinn_vs_fem_error.get('max_error', 0):.6e}")
        print(f"  - Mean Abs Error:      {self.pinn_vs_fem_error.get('mean_abs_error', 0):.6e}")
        
        if self.Loss:
            print(f"\nLoss:")
            print(f"  - Final loss:          {self.Loss[-1]:.6f}")
        print(f"{'='*60}\n")

    def run(self):
        """执行完整的训练和评估流程"""
        self._setup_result_dir()
        
        print(f"{'='*60}")
        print(f"Poisson PINN Results")
        print(f"{'='*60}")
        print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nOptions:")
        for key, value in self.options.items():
            print(f"  {key}: {value}")
        
        try:
            # ===== 1. 生成统一的测试点集 =====
            n_test = self.options.get('test_points_per_dim', 8)
            self.generate_test_points(n_test)
            print(f"\nUnified test points generated:")
            print(f"  - Points per dimension: {self.test_stats['points_per_dim']}")
            print(f"  - Total:                {self.test_stats['total']}")
            print(f"  - Interior:             {self.test_stats['interior']}")
            print(f"  - Boundary:             {self.test_stats['boundary']}")
            
            # ===== 2. 运行FEM求解器 =====
            self.run_fem()
            
            next(self.tmr)
            
            # ===== 3. 设置采样器 =====
            try:
                sampler_pde = ISampler(self.domain, requires_grad=True, mode=self.options['sampling_mode'])
            except Exception as e:
                self.logger.warning(f"ISampler failed: {e}. Using fallback.")
                sampler_pde = None
                
            try:
                sampler_bc = BoxBoundarySampler(self.domain, requires_grad=True, mode=self.options['sampling_mode'])
            except Exception as e:
                self.logger.warning(f"BoxBoundarySampler failed: {e}. Using fallback.")
                sampler_bc = None
            
            mse = nn.MSELoss(reduction='mean')
            self.Loss = []
            self.epoch_times = []
            w = self.weights

            training_start_time = time.time()

            # ===== 4. 训练PINN =====
            for epoch in range(self.epochs + 1):
                epoch_start_time = time.time()
                self.optimizer.zero_grad()

                # 采样
                if sampler_pde is not None:
                    try:
                        spde = sampler_pde.run(self.npde) if self.options['sampling_mode'] == 'random' else self._random_sample_domain(self.npde)
                    except:
                        spde = self._random_sample_domain(self.npde)
                else:
                    spde = self._random_sample_domain(self.npde)
                
                if sampler_bc is not None:
                    try:
                        sbc = sampler_bc.run(self.nbc) if self.options['sampling_mode'] == 'random' else self._random_sample_boundary(self.nbc)
                    except:
                        sbc = self._random_sample_boundary(self.nbc)
                else:
                    sbc = self._random_sample_boundary(self.nbc)

                if epoch == 0:
                    self.training_stats['pde_points'] = spde.shape[0]
                    self.training_stats['bc_points'] = sbc.shape[0]
                    self.training_stats['total_points'] = spde.shape[0] + sbc.shape[0]

                spde = self._ensure_dtype(spde)
                sbc = self._ensure_dtype(sbc)

                pde_res = self.pde_residual(spde)
                bc_res = self.bc_residual(sbc)

                loss = w[0] * mse(pde_res, bm.zeros_like(pde_res)) + w[1] * mse(bc_res, bm.zeros_like(bc_res))
                loss.backward()
                self.optimizer.step()
                
                epoch_time = time.time() - epoch_start_time
                self.epoch_times.append(epoch_time)

                if epoch % 100 == 0:
                    self.Loss.append(loss.item())
                    print(f"epoch: {epoch}, Loss: {loss.item():.6f}, Epoch time: {epoch_time:.4f}s")  
                    
                if self.steplr is not None:
                    self.steplr.step()
            
            self.total_training_time = time.time() - training_start_time
            
            # ===== 5. 计算所有误差（使用统一测试点） =====
            self.compute_all_errors()
            
            # ===== 6. 打印统计信息 =====
            self._print_stats()
            
            # ===== 7. 保存图表 =====
            self._save_figures()
            
            print(f"\nResults saved to: {self.result_dir}")
            print(f"Log file: {self.log_file}")
            print(f"{'='*60}")
            
            self.tmr.send(f'PINN training time: {self.total_training_time:.4f}s')
            next(self.tmr)
                
        finally:
            self._restore_stdout()

    def predict(self, p: TensorLike) -> TensorLike:
        p = self._ensure_dtype(p)
        return self.net(p)

    def show(self):
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        
        if self.result_dir is None:
            print("No results to show.")
            return
        
        image_files = [f for f in os.listdir(self.result_dir) if f.endswith('.png')]
        for img_file in image_files:
            img_path = os.path.join(self.result_dir, img_file)
            img = mpimg.imread(img_path)
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title(img_file)
            plt.show()


if __name__ == "__main__":
    from fealpy.backend import bm
    bm.set_backend('pytorch')  

    options = PoissonPINNModel.get_options()
    options.update({
        'pde': 13,
        'epochs': 3000,
        'mesh_size': 6,  # 6^5 = 7776
        'npde': 3000,
        'nbc': 1000,
        'weights': (1, 30),
        'hidden_size': (128, 64, 64, 32),
        'lr': 0.0005,
        'step_size': 1000,
        'gamma': 0.6,
        'sampling_mode': 'random',
        'activation': 'Tanh',
        'optimizer': 'Adam',
        'test_points_per_dim': 6,   # 统一测试点: 7^5 = 16807
    })

    model = PoissonPINNModel(options=options)
    model.run()
    model.show()