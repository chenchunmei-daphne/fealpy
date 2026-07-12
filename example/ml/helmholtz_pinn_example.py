from fealpy.backend import bm
bm.set_backend('pytorch')  # Set the backend to PyTorch
from fealpy.ml import HelmholtzPINNModel

options = HelmholtzPINNModel.get_options()  # Get the default options of the network
options['pde'] = 1
options['wave'] = 20  # Set the number of PDE points
options['mesh_size'] = 60
options['hidden_size'] = (50, 50, 50, 50, 32, 18)
options['npde'] = 800
options['nbc'] = 200
options['epochs'] = 3000  # Set the number of training epochs
model = HelmholtzPINNModel(options=options)
model.run()   # Train the network
model.show()   # Show the results of the network training

