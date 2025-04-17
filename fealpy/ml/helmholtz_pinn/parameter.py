import torch.nn as nn

from fealpy.backend import backend_manager as bm
bm.set_backend('pytorch')

class Parameter():
    """
    A class for managing hyperparameters of the neural network model.

    Parameters:
        NL: Number of neurons in the first hidden layer. Defaults to 64.
        npde: Number of sample points for Partial Differential Equations (PDE). 
              Defaults to 200.
        nbc: Number of sample points for boundary conditions (BC). 
             Defaults to 100.
        iter: Number of iterations for model training. Defaults to 400.
        lr: Learning rate for optimization. Defaults to 0.01.
        step_size: Interval for adjusting learning rate (in steps). 
                  Defaults to 50.
    """

    def __init__(self, NL=64, npde=200, nbc=100, iter=400, lr=0.01, step_size = 50):
        self.NN = self._check(NL,'NL')
        self.npde = self._check(npde, "npde")
        self.nbc = self._check(nbc, "nbc")
        self.iter = self._check(iter, "iter")
        self.lr = lr
        self.step_size = self._check(step_size, 'step_size')

    @staticmethod
    def _check(value, name):
        """
        Check if the hyperparameter is a positive integer.

        Parameters:
            value: The value to be checked.
            name: Name of the parameter for error message.

        Returns:
            int: The validated integer value.

        Raises:
            ValueError: If the value is not a positive integer.
        """
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
        return value

    
    def net(self):
        ''' 网络结构 '''
        d = 2  # 方程的维数
        net = nn.Sequential(nn.Linear(d, self.NN, dtype=bm.float64),
                            nn.Tanh(),
                            nn.Linear(self.NN, self.NN // 2, dtype=bm.float64),
                            nn.Tanh(),
                            nn.Linear(self.NN // 2, self.NN // 4, dtype=bm.float64),
                            nn.Tanh(),
                            nn.Linear(self.NN // 4, 1, dtype=bm.float64))    # 默认 nn.Linear 的数据类型为 torch.float32
        return net


# 实例化
test_200 = Parameter(iter=200)

## 集中配置区
configs = {
    'default': Parameter(),
    'test_200': Parameter(iter=200)
}