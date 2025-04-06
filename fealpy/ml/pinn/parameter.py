import torch.nn as nn
from fealpy.mesh import TriangleMesh
from  fealpy.ml.sampler import ISampler, BoxBoundarySampler
from fealpy.backend import backend_manager as bm
bm.set_backend('pytorch')

class Parameter():
    def __init__(self, NN=32, npde=200, nbc=100, iter=400, lr=0.01, k=1.0,
                 domain=(-0.5, 0.5, -0.5, 0.5), nx=64, ny=64,
                 step_size = 50):
        '''
        @brief: Module's hyperparameters.

        @param NN:Number of neurons in the first hidden layer.
        @param npde: number of points for Partial Differential Equations (PDE).
        @param nbc: number of points for boundary conditions (BC).
        @param iter: number of iterations for model training.
        @param lr: learning rate.
        @param k: wave number.
        @param domain: The domain of the partial differential equation.
        @param nx:Number of divisions along the x-axis on the parameter domain.
        @param ny:Number of divisions along the y-axis on the parameter domain.
        @para step_size:Adjust the learning rate lr every step_size steps.
        '''

        self.NN = self._check(NN,'NN')
        self.npde = self._check(npde, "npde")
        self.nbc = self._check(nbc, "nbc")
        self.iter = self._check(iter, "iter")
        self.lr = lr
        self.k = k

        self.domain = domain    # 网格参数
        self.nx = self._check(nx, 'nx')
        self.ny = self._check(ny, 'ny')
        self.mesh = TriangleMesh.from_box(self.domain, nx=self.nx, ny=self.ny)

        self.step_size = self._check(step_size,'step_size')

        self.samplerpde = ISampler(self.domain, requires_grad=True)
        self.samplerbc = BoxBoundarySampler(self._bc(self.domain, 0), self._bc(self.domain, 1), requires_grad=True)

    @staticmethod
    def _check(value, name):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
        return value

    @staticmethod
    def _bc(domain, location):
        '''
        @para domain: The domain of the partial differential equation.
        @param location: 0 or 1, if location=0, return domain[0::2], which is the first parameter of BoxBoundarySampler.
        '''
        return domain[location::2]
    def net(self):
        d = len(self.domain) // 2  # 数据的维数
        net = nn.Sequential(nn.Linear(d, self.NN, dtype=bm.float64), nn.Tanh(),
                              nn.Linear(self.NN, self.NN // 2, dtype=bm.float64), nn.Tanh(),
                              nn.Linear(self.NN // 2, self.NN // 4, dtype=bm.float64), nn.Tanh(),
                              nn.Linear(self.NN // 4, 1, dtype=bm.float64))    # 默认 nn.Linear 的数据类型为 torch.float32

        return net

test_200 = Parameter(iter=200)

## 集中配置区
# 在 parameter.py 文件末尾添加
configs = {
    'default': Parameter(),
    'test_200': Parameter(iter=200)
}