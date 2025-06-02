from fealpy.model import PDEDataManager
pde = PDEDataManager('poisson').get_example('coscos')
## 或者
# manager = PDEDataManager('poisson')
# pde = manager.get_example('coscos')
help(pde)