import numpy as np
import matplotlib.pyplot as plt

from fealpy.functionspace import ConformingScalarVESpace2d 
from fealpy.functionspace import ConformingVirtualElementSpace2d
import ipdb

from fealpy.vem.conforming_scalar_vem_h1_projector import ConformingScalarVEMH1Projector2d
from fealpy.mesh import MeshFactory as MF
from fealpy.mesh.polygon_mesh import PolygonMesh
def test_assembly_cell_righthand_side(p,plot=False):
    nx = 2
    ny = 2
    dim = 2
    domain = np.array([0, 1, 0, 1])
    mesh = MF.boxmesh2d(domain, nx, ny, meshtype ='poly')
    space = ConformingVirtualElementSpace2d(mesh, p=p)
    realB = space.matrix_B()
    if plot:
        fig ,axes = plt.subplots()
        mesh.add_plot(axes)
        mesh.find_node(axes, showindex=True)
        mesh.find_cell(axes, showindex=True)
        mesh.find_edge(axes, showindex=True)
        plt.show()


    mesh = MF.boxmesh2d(domain, nx, ny, meshtype ='tri')
    mesh = PolygonMesh.from_triangle_mesh_by_dual(mesh)
    space =  ConformingScalarVESpace2d(mesh, p=p)
    b = ConformingScalarVEMH1Projector2d()
    B = b.assembly_cell_righthand_side(space)

    ldof = mesh.number_of_local_ipoints(p)
    a = np.add.accumulate(ldof)
    NC = mesh.number_of_cells()
    location = np.zeros(NC+1, dtype=np.int_)
    location[1:] = a
    for i in range(NC):
        np.testing.assert_equal(realB[:,location[i]:location[i+1]], B[i])
        i = i+1

    if plot:
        fig ,axes = plt.subplots()
        mesh.add_plot(axes)
        mesh.find_node(axes, showindex=True)
        mesh.find_cell(axes, showindex=True)
        mesh.find_edge(axes, showindex=True)
        plt.show()
if __name__ == "__main__":
    test_assembly_cell_righthand_side(3)
