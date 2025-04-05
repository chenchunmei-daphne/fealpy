from fealpy.solver import  cg, gmres, spsolve
from fealpy.backend import backend_manager as bm



# A = bm.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
A = bm.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
b = bm.tensor([1, 1, 1])
u = cg(A, b)
print(u)

# u1 = gmres(A, b)
# print(u1)
#
# u2 =spsolve(A, b)
# print(u2)