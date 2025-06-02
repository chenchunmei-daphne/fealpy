import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')

from fealpy.model import PDEDataManager
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import ScalarDiffusionIntegrator
from fealpy.fem import ScalarSourceIntegrator
from fealpy.fem import BilinearForm, LinearForm
from fealpy.fem import DirichletBC
from fealpy.solver import spsolve

pde = PDEDataManager('poisson').get_example('coscos')
domain = pde.domain()
mesh = TriangleMesh.from_box(domain, nx=2, ny=2) 

p = 1
q = 3
NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()
NE = mesh.number_of_edges()
ipoints = mesh.interpolation_points(p)

print('number of nodes =', NN)
print('number of cells =', NC)
print('number of edges =', NE)  

space = LagrangeFESpace(mesh, p)
print('gdof =', space.number_of_global_dofs())

cell2dof = space.cell_to_dof()

bform = BilinearForm(space)
bform.add_integrator(ScalarDiffusionIntegrator(q=3))
A = bform.assembly()

lform = LinearForm(space)
lform.add_integrator(ScalarSourceIntegrator(pde.source, q=3))
F = lform.assembly()

bc = DirichletBC(space, pde.dirichlet)
A, F = bc.apply(A, F)

uh = space.function()
uh[:] = spsolve(A, F, solver='scipy')

e = mesh.error(pde.solution, uh)
print('error =', e)
e1 = mesh.error(pde.gradient, uh.grad_value, q=q)
print('gradient error =', e1)


# qf = mesh.quadrature_formula(q, etype='cell')
# bcs, ws = qf.get_quadrature_points_and_weights()
# print("NQ:", ws.shape)

# ps = mesh.bc_to_point(bcs)
# print("ps.shape", ps.shape)

# g0 = pde.gradient(p=ps)
# g1 = uh.grad_value(bcs)
# print("g0.shape:", g0.shape, "g1.shape:", g1.shape)

# fig = plt.figure()
# axes = fig.add_subplot(111)
# mesh.add_plot(axes)
# mesh.find_node(axes, node=ipoints, 
#                showindex=True, color='r', fontsize='30')
# mesh.find_cell(axes, showindex=True, fontsize='35')
# plt.show()