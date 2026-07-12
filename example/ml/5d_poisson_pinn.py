import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from typing import Union, Optional
import time
import os
import sys
from datetime import datetime

from fealpy.backend import bm
from fealpy.utils import timer
from fealpy.typing import TensorLike
from fealpy.model import ComputationalModel, PDEModelManager
from fealpy.model.poisson import PoissonPDEDataT

from fealpy.ml import gradient, optimizers, activations

from fealpy.ml.modules import Solution
from fealpy.ml.sampler import ISampler, BoxBoundarySampler


class PoissonPINNModel(ComputationalModel):
    """A Physics-Informed Neural Network (PINN) model for solving Poisson equations."""
    
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
            'interior_points': 0,
            'boundary_points': 0,
        }
        self.testing_stats = {
            'test_points': 0,
            'test_time': 0.0,
            'l2_error': 0.0,
        }
        
        # 结果保存路径
        self.result_dir = None
        self.log_file = None
        self.original_stdout = sys.stdout

    @classmethod
    def get_options(cls):
        """Get default configuration parameters for the model."""
        import argparse

        parser = argparse.ArgumentParser(description="Poisson equation solver using PINN.")

        parser.add_argument('--pde', default=1, type=int,
                            help="Built-in PDE example ID for different Poisson problems, default is 1.")
        
        parser.add_argument('--sampling_mode', 
                            default='random', type=str,
                            help="Sampling method for collocation points: 'random' or 'linspace', default is 'random'")

        parser.add_argument('--npde',
                            default=400, type=int,
                            help='Number of PDE samples, default is 400.')

        parser.add_argument('--nbc',
                            default=100, type=int,
                            help='Number of boundary condition samples, default is 100.')
    
        parser.add_argument('--weights',
                            default=(1, 30), type=tuple,
                            help='The first value is the weight for the equation loss, and the second ' \
                            'value is the weight for the boundary loss., default is (1, 30).')
        
        parser.add_argument('--hidden_size',
                            default=(64, 64, 32), type=tuple,
                            help='Default hidden sizes, default is (64, 64, 32).')

        parser.add_argument('--optimizer', 
                            default="Adam",  type=str,
                            help="Optimizer to use for training, default is Adam, options are 'Adam' , 'SGD'.")

        parser.add_argument('--activation',
                            default="Tanh", type=str,
                            help="Activation function, default is Tanh.")

        parser.add_argument('--lr',
                            default=0.001, type=float,
                            help='Learning rate for the optimizer, default is 0.001.')
        
        parser.add_argument('--step_size',
                            default=0, type=int,
                            help='Period of learning rate decay, default is 0.')

        parser.add_argument('--gamma',
                            default=0.99, type=float,
                            help='Multiplicative factor of learning rate decay. Default: 0.99.')

        parser.add_argument('--epochs',
                            default=2000, type=int,
                            help='Number of training epochs, default is 2000.')
        
        parser.add_argument('--pbar_log',
                            default=True, type=bool,
                            help='Whether to show progress bar, default is True')

        parser.add_argument('--log_level',
                            default='INFO', type=str,
                            help='Log level, default is INFO.')
        
        parser.add_argument('--test_points_per_dim',
                            default=10, type=int,
                            help='Number of test points per dimension for error estimation, default is 10.')
        
        options = vars(parser.parse_args())
        return options
    
    def set_pde(self, pde: Union[PoissonPDEDataT, int]=1):
        """Initialize the PDE problem definition."""
        if isinstance(pde, int):
            self.pde = PDEModelManager('poisson').get_example(pde)
        else:
            self.pde = pde 
        self.gd = self.pde.geo_dimension()
        self.domain = self.pde.domain()

    def set_network(self, net=None):
        """Configure the neural network architecture and optimizer."""
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
        """Create learning rate scheduler."""
        if step_size == 0:
            self.steplr = None
        else:
            self.steplr = StepLR(self.optimizer, step_size, gamma)

    def _ensure_dtype(self, tensor: TensorLike) -> TensorLike:
        """Ensure tensor has the correct dtype (float64)."""
        if bm.backend_name == 'pytorch':
            import torch
            if tensor.dtype != torch.float64:
                tensor = tensor.to(dtype=torch.float64)
        return tensor

    def _is_on_boundary(self, p: TensorLike, atol: float = 1e-12) -> TensorLike:
        """Check if points are on the boundary."""
        if hasattr(self.pde, 'is_dirichlet_boundary'):
            return self.pde.is_dirichlet_boundary(p)
        else:
            on_boundary = bm.zeros(p.shape[0], dtype=bool)
            for i in range(self.gd):
                coord = p[:, i]
                on_boundary = on_boundary | (bm.abs(coord - self.domain[2*i]) < atol)
                on_boundary = on_boundary | (bm.abs(coord - self.domain[2*i+1]) < atol)
            return on_boundary

    def pde_residual(self, p: TensorLike) -> TensorLike:
        """Compute PDE residual (Laplacian(u) + f)."""
        p = self._ensure_dtype(p)
        u = self.net(p)
        f = self.pde.source(p)
        f = self._ensure_dtype(f)

        grad_u = gradient(u, p, create_graph=True)
        laplacian = bm.zeros(u.shape[0], dtype=bm.float64)    
        
        for i in range(p.shape[-1]):
            u_ii = gradient(grad_u[..., i], p, create_graph=True, split=True)[i]
            laplacian += u_ii.flatten()

        val = laplacian + f
        return val

    def bc_residual(self, p: TensorLike) -> TensorLike:
        """Compute boundary condition residual (u - g)."""
        p = self._ensure_dtype(p)
        u = self.net(p).flatten()
        bc = self.pde.dirichlet(p)
        bc = self._ensure_dtype(bc)
        val = u - bc
        return val

    def _random_uniform(self, low: float, high: float, size: tuple) -> TensorLike:
        """Generate uniform random numbers with float64 dtype."""
        if bm.backend_name == 'pytorch':
            import torch
            return torch.rand(size, dtype=torch.float64) * (high - low) + low
        elif bm.backend_name == 'jax':
            return bm.random.uniform(low, high, size, dtype=bm.float64)
        else:
            return bm.random.uniform(low, high, size).astype(bm.float64)

    def _random_sample_domain(self, n: int) -> TensorLike:
        """Fallback method for random sampling in the domain."""
        p = bm.zeros((n, self.gd), dtype=bm.float64)
        for i in range(self.gd):
            p[:, i] = self._random_uniform(self.domain[2*i], self.domain[2*i+1], (n,))
        if bm.backend_name == 'pytorch':
            p.requires_grad_(True)
        return p

    def _random_sample_boundary(self, n: int) -> TensorLike:
        """Fallback method for random sampling on the boundary."""
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

    def compute_l2_error(self, n_points_per_dim: int = 10) -> float:
        """Compute L2 error on a uniform grid."""
        if not hasattr(self.pde, 'solution'):
            return 0.0
            
        start_time = time.time()
            
        total_points = n_points_per_dim ** self.gd
        if total_points > 100000:
            n_samples = min(10000, total_points)
            p = bm.zeros((n_samples, self.gd), dtype=bm.float64)
            for i in range(self.gd):
                p[:, i] = self._random_uniform(self.domain[2*i], self.domain[2*i+1], (n_samples,))
            actual_points = n_samples
        else:
            points_per_dim = [bm.linspace(self.domain[2*i], self.domain[2*i+1], n_points_per_dim, dtype=bm.float64) 
                             for i in range(self.gd)]
            meshgrid = bm.meshgrid(*points_per_dim, indexing='ij')
            p = bm.stack([grid.flatten() for grid in meshgrid], axis=-1)
            actual_points = p.shape[0]
        
        on_boundary = self._is_on_boundary(p)
        interior_count = int(bm.sum(~on_boundary).item())
        boundary_count = int(bm.sum(on_boundary).item())
        
        u_pred = self.net(p).flatten()
        u_true = self.pde.solution(p).flatten()
        
        diff = u_pred - u_true
        l2_error = bm.sqrt(bm.mean(diff**2))
        
        self.testing_stats['test_points'] = actual_points
        self.testing_stats['interior_points'] = interior_count
        self.testing_stats['boundary_points'] = boundary_count
        self.testing_stats['test_time'] = time.time() - start_time
        self.testing_stats['l2_error'] = l2_error.item()
        
        return l2_error.item()

    def _setup_result_dir(self):
        """Create result directory with timestamp."""
        base_dir = r"D:\chen\fealpy\example\ml\Possion_result"
        timestamp = datetime.now().strftime("%m%d%H%M")
        
        # 获取PDE编号
        pde_id = self.options.get('pde', 1)
        self.result_dir = os.path.join(base_dir, f"pde{pde_id}_{timestamp}")
        
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        
        # 创建日志文件
        self.log_file = os.path.join(self.result_dir, f"poisson_result_{timestamp}.txt")
        
        # 重定向输出到文件
        self.file_handle = open(self.log_file, 'w', encoding='utf-8')
        sys.stdout = self.file_handle

    def _restore_stdout(self):
        """Restore stdout and close file."""
        if hasattr(self, 'file_handle') and self.file_handle:
            sys.stdout = self.original_stdout
            self.file_handle.close()

    def _save_figures(self):
        """Save figures to result directory."""
        import matplotlib.pyplot as plt
        
        # Loss curve
        fig1, axes1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        if self.Loss:
            Loss = bm.log10(bm.tensor(self.Loss)).numpy()
            axes1.plot(Loss, 'r-', linewidth=2)
            axes1.set_title('PINN Training Loss', fontsize=12)
            axes1.set_xlabel('training epochs (x100)', fontsize=10)
            axes1.set_ylabel('log10(Loss)', fontsize=10)
            axes1.grid(True)
        fig1.savefig(os.path.join(self.result_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
        plt.close(fig1)

        # Error curve
        if hasattr(self, 'error') and self.error and self.solution_flag:
            fig2, axes2 = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
            error = bm.log10(bm.tensor(self.error)).numpy()
            axes2.plot(error, 'b--', linewidth=2)
            axes2.set_title('L2 Error between PINN Solution and Exact Solution', fontsize=12)
            axes2.set_ylabel('log10(Error)', fontsize=10)
            axes2.set_xlabel('training epochs (x100)', fontsize=10)
            axes2.grid(True)
            fig2.savefig(os.path.join(self.result_dir, 'error_curve.png'), dpi=300, bbox_inches='tight')
            plt.close(fig2)

        # For 1D and 2D problems, plot solution comparison
        if self.gd <= 2 and self.solution_flag:
            n_points = min(self.options.get('test_points_per_dim', 20), 50)
            
            if self.gd == 1:
                x = bm.linspace(self.domain[0], self.domain[1], n_points, dtype=bm.float64).reshape(-1, 1)
                u_pred = self.net(x).detach().numpy().flatten()
                u_true = self.pde.solution(x).detach().numpy().flatten()
                x_np = x.detach().numpy().flatten()
                
                fig3 = plt.figure(figsize=(10, 6))
                plt.plot(x_np, u_true, 'b-', linewidth=2, label='Exact Solution')
                plt.plot(x_np, u_pred, 'g--', linewidth=2, label='PINN Prediction')
                plt.plot(x_np, u_pred - u_true, 'r-', linewidth=2, label='Error')
                plt.xlabel('x', fontsize=12)
                plt.ylabel('u(x)', fontsize=12)
                plt.title('Comparison between PINN and Exact Solution', fontsize=14)
                plt.legend(fontsize=12)
                plt.grid(True, linestyle=':')
                fig3.savefig(os.path.join(self.result_dir, 'solution_comparison.png'), dpi=300, bbox_inches='tight')
                plt.close(fig3)
                
            elif self.gd == 2:
                x = bm.linspace(self.domain[0], self.domain[1], n_points, dtype=bm.float64)
                y = bm.linspace(self.domain[2], self.domain[3], n_points, dtype=bm.float64)
                X, Y = bm.meshgrid(x, y, indexing='ij')
                points = bm.stack([X.flatten(), Y.flatten()], axis=-1)
                
                u_pred = self.net(points).detach().numpy().reshape(n_points, n_points)
                u_true = self.pde.solution(points).detach().numpy().reshape(n_points, n_points)
                X_np = X.detach().numpy()
                Y_np = Y.detach().numpy()
                
                fig3 = plt.figure(figsize=(15, 5))
                
                ax1 = fig3.add_subplot(131, projection='3d')
                surf1 = ax1.plot_surface(X_np, Y_np, u_pred, cmap='viridis', alpha=0.8)
                ax1.set_title('PINN Solution')
                ax1.set_xlabel('X')
                ax1.set_ylabel('Y')
                ax1.set_zlabel('u(x,y)')
                fig3.colorbar(surf1, ax=ax1, shrink=0.5)
                
                ax2 = fig3.add_subplot(132, projection='3d')
                surf2 = ax2.plot_surface(X_np, Y_np, u_true, cmap='plasma', alpha=0.8)
                ax2.set_title('Exact Solution')
                ax2.set_xlabel('X')
                ax2.set_ylabel('Y')
                ax2.set_zlabel('u(x,y)')
                fig3.colorbar(surf2, ax=ax2, shrink=0.5)
                
                ax3 = fig3.add_subplot(133, projection='3d')
                surf3 = ax3.plot_surface(X_np, Y_np, u_pred - u_true, cmap='coolwarm', alpha=0.8)
                ax3.set_title('Error: PINN - Exact')
                ax3.set_xlabel('X')
                ax3.set_ylabel('Y')
                ax3.set_zlabel('Error')
                fig3.colorbar(surf3, ax=ax3, shrink=0.5)
                
                plt.suptitle('Comparison between PINN and Exact Solution', fontsize=14)
                fig3.savefig(os.path.join(self.result_dir, 'solution_comparison.png'), dpi=300, bbox_inches='tight')
                plt.close(fig3)

    def _print_training_stats(self):
        """Print training and testing statistics."""
        print(f"\n{'='*60}")
        para = sum(p.numel() for p in self.net.parameters())
        print(f'Model parameters: {para}')
        print(f"{'='*60}")
        print(f"TRAINING STATISTICS")
        print(f"{'='*60}")
        print(f"Training Points:")
        print(f"  - PDE (interior) points: {self.training_stats['pde_points']}")
        print(f"  - BC (boundary) points:  {self.training_stats['bc_points']}")
        print(f"  - Total training points: {self.training_stats['total_points']}")
        print(f"\nTraining Time:")
        print(f"  - Total training time:   {self.total_training_time:.4f} seconds")
        print(f"  - Average per epoch:     {self.total_training_time / max(1, len(self.epoch_times)):.4f} seconds")
        print(f"  - Number of epochs:      {len(self.epoch_times)}")
        
        if self.solution_flag:
            print(f"\n{'='*60}")
            print(f"TESTING STATISTICS")
            print(f"{'='*60}")
            print(f"Test Points:")
            print(f"  - Total test points:   {self.testing_stats['test_points']}")
            print(f"  - Interior points:     {self.testing_stats['interior_points']}")
            print(f"  - Boundary points:     {self.testing_stats['boundary_points']}")
            print(f"  - Interior ratio:      {self.testing_stats['interior_points'] / max(1, self.testing_stats['test_points']):.2%}")
            print(f"  - Boundary ratio:      {self.testing_stats['boundary_points'] / max(1, self.testing_stats['test_points']):.2%}")
            print(f"\nTest Results:")
            print(f"  - Test time:           {self.testing_stats['test_time']:.6f} seconds")
            print(f"  - L2 error:            {self.testing_stats['l2_error']:.6e}")
        
        if self.Loss:
            print(f"\nLoss:")
            print(f"  - Final loss:          {self.Loss[-1]:.6f}")
        print(f"{'='*60}\n")

    def run(self):
        """Execute the training process for the PINN model."""
        # 设置结果保存目录
        self._setup_result_dir()
        
        # 打印运行信息到文件
        print(f"{'='*60}")
        print(f"Poisson PINN Results")
        print(f"{'='*60}")
        print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nOptions:")
        for key, value in self.options.items():
            print(f"  {key}: {value}")
        
        try:
            next(self.tmr)
            
            try:
                sampler_pde = ISampler(self.domain, requires_grad=True, mode=self.options['sampling_mode'])
            except Exception as e:
                self.logger.warning(f"ISampler initialization failed: {e}. Using fallback sampling.")
                sampler_pde = None
                
            try:
                sampler_bc = BoxBoundarySampler(self.domain, requires_grad=True, mode=self.options['sampling_mode'])
            except Exception as e:
                self.logger.warning(f"BoxBoundarySampler initialization failed: {e}. Using fallback sampling.")
                sampler_bc = None
            
            mse = nn.MSELoss(reduction='mean')

            self.Loss = []
            self.error = []
            self.epoch_times = []
            w = self.weights
            
            n_test = self.options.get('test_points_per_dim', 10)

            training_start_time = time.time()

            for epoch in range(self.epochs + 1):
                epoch_start_time = time.time()
                
                self.optimizer.zero_grad()

                if sampler_pde is not None:
                    try:
                        if self.options['sampling_mode'] == 'linspace' and epoch == 0:
                            spde = sampler_pde.run(self.npde)
                        elif self.options['sampling_mode'] == 'random':
                            spde = sampler_pde.run(self.npde)
                        else:
                            spde = self._random_sample_domain(self.npde)
                    except Exception as e:
                        self.logger.warning(f"PDE sampling failed: {e}. Using fallback.")
                        spde = self._random_sample_domain(self.npde)
                else:
                    spde = self._random_sample_domain(self.npde)
                
                if sampler_bc is not None:
                    try:
                        if self.options['sampling_mode'] == 'linspace' and epoch == 0:
                            sbc = sampler_bc.run(self.nbc)
                        elif self.options['sampling_mode'] == 'random':
                            sbc = sampler_bc.run(self.nbc)
                        else:
                            sbc = self._random_sample_boundary(self.nbc)
                    except Exception as e:
                        self.logger.warning(f"Boundary sampling failed: {e}. Using fallback.")
                        sbc = self._random_sample_boundary(self.nbc)
                else:
                    sbc = self._random_sample_boundary(self.nbc)

                if epoch == 0:
                    self.training_stats['pde_points'] = spde.shape[0]
                    self.training_stats['bc_points'] = sbc.shape[0]
                    self.training_stats['total_points'] = spde.shape[0] + sbc.shape[0]
                    self.training_stats['interior_points'] = spde.shape[0]
                    self.training_stats['boundary_points'] = sbc.shape[0]

                spde = self._ensure_dtype(spde)
                sbc = self._ensure_dtype(sbc)

                pde_res = self.pde_residual(spde)
                bc_res = self.bc_residual(sbc)

                mse_pde = mse(pde_res, bm.zeros_like(pde_res))
                mse_bc = mse(bc_res, bm.zeros_like(bc_res))

                loss = w[0] * mse_pde + w[1] * mse_bc
                loss.backward()
                self.optimizer.step()
                
                epoch_time = time.time() - epoch_start_time
                self.epoch_times.append(epoch_time)

                if epoch % 100 == 0:
                    try:
                        if hasattr(self.pde, 'solution'):
                            self.solution_flag = True
                            error = self.compute_l2_error(n_test)
                            self.error.append(error)
                    except (NotImplementedError, AttributeError):
                        self.solution_flag = False
                    self.Loss.append(loss.item())
                    print(f"epoch: {epoch}, Loss: {loss.item():.6f}, Epoch time: {epoch_time:.4f}s")  
                    
                if self.steplr is not None:
                    self.steplr.step()
            
            self.total_training_time = time.time() - training_start_time
            
            # 打印统计信息
            self._print_training_stats()
            
            # 保存图表
            self._save_figures()
            
            print(f"\nResults saved to: {self.result_dir}")
            print(f"Log file: {self.log_file}")
            print(f"{'='*60}")
            
            self.tmr.send(f'PINN training time: {self.total_training_time:.4f}s')
            if self.solution_flag:
                next(self.tmr)
                
        finally:
            # 恢复stdout
            self._restore_stdout()

    def predict(self, p: TensorLike) -> TensorLike:
        """Make predictions using the trained network."""
        p = self._ensure_dtype(p)
        return self.net(p)

    def show(self):
        """Display the saved figures."""
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        
        if self.result_dir is None:
            print("No results to show. Please run the model first.")
            return
        
        # 显示保存的图片
        image_files = [f for f in os.listdir(self.result_dir) if f.endswith('.png')]
        
        if not image_files:
            print("No image files found.")
            return
        
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
        'pde': 13,                    # 5D Poisson问题
        'epochs': 3000,               # 训练轮数
        'npde': 5000,                 # 内部点 (PDE残差点)
        'nbc': 2500,                   # 边界点
        'weights': (1, 30),           # 损失权重 (PDE损失, BC损失)
        'hidden_size': (50, 50), # 网络结构
        'lr': 0.001,                  # 学习率
        'sampling_mode': 'random',    # 采样方式
        'activation': 'Tanh',         # 激活函数
        'optimizer': 'Adam',          # 优化器
        'test_points_per_dim': 7,     # 测试点每维数量
    })

    model = PoissonPINNModel(options=options)
    model.run()
    model.show()  


