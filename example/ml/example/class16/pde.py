from fealpy.backend import bm
bm.set_backend("pytorch")
a = [1, 5, 0]
c = (0, 9, 2)
print(max(a), max(c))

exit()
a = bm.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = bm.tensor([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
c = bm.concat([a, b], axis=-1).reshape(-1, 3)
print(c)



import numpy as np

# a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# b = np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
# print(np.stack([a, b], axis=0).reshape(-1, 3))

import torch
# bm.set_backend("numpy")
a = bm.tensor([-1., -1.])
b = bm.tensor([1., 1.])
data = bm.stack([a, b]).T
r1 = bm.copy(data)
r1[0, 1] = data[0, 0]
print(data)
print(r1)