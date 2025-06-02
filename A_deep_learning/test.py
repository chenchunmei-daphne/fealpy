import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')

from fealpy.model import PDEDataManager
from fealpy.mesh import TriangleMesh, IntervalMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import ScalarDiffusionIntegrator
from fealpy.fem import ScalarSourceIntegrator
from fealpy.fem import BilinearForm, LinearForm
from fealpy.fem import DirichletBC
from fealpy.solver import spsolve


pde = PDEDataManager('poisson').get_example('sin')

domain = pde.domain()
mesh = IntervalMesh.from_interval_domain(interval=domain, nx=10)

p = 3
q = 3
space = LagrangeFESpace(mesh, p)
print("gdof:", space.number_of_global_dofs())

bform = BilinearForm(space)
bform.add_integrator(ScalarDiffusionIntegrator(q=q))
A = bform.assembly()

lform = LinearForm(space)
lform.add_integrator(ScalarSourceIntegrator(pde.source, q=q))
F = lform.assembly()

bc = DirichletBC(space, pde.dirichlet)
A, F = bc.apply(A, F)

uh = space.function()
uh[:] = spsolve(A, F, solver='scipy')

e = mesh.error(pde.solution, uh)
print('solution error =', e)

qf = mesh.quadrature_formula(q, etype='cell')
bcs, ws = qf.get_quadrature_points_and_weights()
print("NQ:", ws.shape)
ps = mesh.bc_to_point(bcs)
# print(bcs.shape, ps.shape)
g0 = pde.gradient(p=ps)
g1 = uh.grad_value(bcs)
print("g0.shape:", g0.shape, "g1.shape:", g1.shape)

e1 = mesh.error(pde.gradient, uh.grad_value, q=q)
print('gradient error =', e1)

# fig = plt.figure()
# axes = fig.add_subplot()
# mesh.add_plot(axes)
# mesh.find_node(axes, node=ipoints, 
#                showindex=True, color='r', fontsize='30')
# mesh.find_cell(axes, showindex=True, fontsize='35')
# plt.show()



exit()

NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()
NE = mesh.number_of_edges()
ipoints = mesh.interpolation_points(p)

print('number of nodes =', NN)
print('number of cells =', NC)
print('number of edges =', NE)  

cell2dof = space.cell_to_dof()
print(cell2dof)

mesh.uniform_refine()
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