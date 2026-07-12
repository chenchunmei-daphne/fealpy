from fealpy.backend import bm
bm.set_backend('pytorch')  
from fealpy.ml import PoissonPINNModel

options = PoissonPINNModel.get_options()  
options['pde'] = 13
options['epochs'] = 3
model = PoissonPINNModel(options=options)
model.run()   
model.show()  

     