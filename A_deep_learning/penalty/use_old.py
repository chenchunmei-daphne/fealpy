from fealpy.old.functionspace import  LagrangeFiniteElementSpace
from fealpy.old.mesh import TriangleMesh
mesh = TriangleMesh.from_box()
space = LagrangeFiniteElementSpace(mesh)
J = space.penalty_matrix(q=1)

print('内罚项矩阵：\n', J)

