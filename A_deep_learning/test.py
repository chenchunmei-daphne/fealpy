from fealpy.mesh import TriangleMesh, TetrahedronMesh
from fealpy.functionspace import LagrangeFESpace
mesh = TriangleMesh.from_box(nx=2, ny=2)
space = LagrangeFESpace(mesh=mesh)
print(mesh.boundary_face_index())


from fealpy.mesh import TriangleMesh



# print(mesh.entity('node'))

# # space 
# print(space.face_to_dof().shape, space.face_to_dof())

# # mesh
# print(mesh.quadrature_formula(q=3, etype='face'))
# print(mesh.number_of_faces())
# print(mesh.face_unit_normal())