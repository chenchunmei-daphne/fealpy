

class Parameter():
    def __init__(self, npde=200, nbc=100, iter=400, lr=0.01):
        '''
        @brief: Module's hyperparameters.
        @param npde: number of points for Partial Differential Equations (PDE).
        @param nbc: number of points for boundary conditions (BC).
        @param iter: number of iterations for model training.
        @param lr: learning rate.
        @return: none.
        '''
        self.npde = self._check(npde, "npde")
        self.nbc = self._check(nbc, "nbc")
        self.iter = self._check(iter, "iter")
        self.lr = lr

    @staticmethod
    def _check(value, name):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
        return value


