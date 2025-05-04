from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
import matplotlib.pyplot as plt
mesh = TriangleMesh.from_box(nx=2, ny=2)
space = LagrangeFESpace(mesh, p=1)
p = space.p
TD = mesh.top_dimension()
NF = mesh.number_of_faces()
NC = mesh.number_of_cells()

isFaceDof = (mesh.multi_index_matrix(p,TD) == 0) 
cell2face = mesh.cell_to_face()
cell2facesign = mesh.cell_to_face_sign()

ldof = space.number_of_local_dofs() 
fdof = space.number_of_local_dofs('face') 
ndof = ldof - fdof
face2dof = bm.zeros((NF, fdof + 2*ndof), dtype=bm.int32)
cell2dof = space.cell_to_dof()


# print('--'*20)
print('TD: ', TD)
# print('--'*20)
# print('NF: ', NF)
# print('--'*20)
# print('NC: ', NC)
# print('--'*20)

print('isFaceDof: ', isFaceDof.shape, isFaceDof, sep='\n')
print('--'*20)
# print('cell2face: ', cell2face)
# print('--'*20)
print('cell2facesign: ', cell2facesign.shape, cell2facesign, sep='\n')
print('--'*20)

# print('ldof: ', ldof)
# print('--'*20)
# print('fdof: ', fdof)
# print('--'*20)
# print('ndof: ', ndof)
# print('--'*20)
# print('face2dof: ', face2dof)
# print('--'*20)
# print('cell2dof: ', cell2dof)
# print('--'*20)
i = 0
lidx, = bm.nonzero( cell2facesign[:, i]) # 单元是全局面的左边单元
ridx, = bm.nonzero(~cell2facesign[:, i]) # 单元是全局面的右边单元
idx0, = bm.nonzero( isFaceDof[:, i]) # 在面上的自由度
idx1, = bm.nonzero(~isFaceDof[:, i]) # 不在面上的自由度

fidx = cell2face[:, i] # 第 i 个面的全局编号
face2dof[fidx[lidx, None], bm.arange(fdof,      fdof+  ndof)] = cell2dof[lidx[:, None], idx1] 
face2dof[fidx[ridx, None], bm.arange(fdof+ndof, fdof+2*ndof)] = cell2dof[ridx[:, None], idx1]

print('lidx (左侧单元):', lidx)
print('--' * 20)
print('ridx (右侧单元):', ridx)
print('--' * 20)

print('idx0 (面自由度):', idx0)
print('--' * 20)
print('idx1 (内部自由度):', idx1)
print('--' * 20)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()
mesh.add_plot(axes=ax)
mesh.find_cell(ax, showindex=True)
mesh.find_face(ax, showindex=True)
mesh.find_node(ax, showindex=True)
plt.show()
