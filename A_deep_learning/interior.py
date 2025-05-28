from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
import matplotlib.pyplot as plt
from fealpy.sparse import csr_matrix

domain = [0, 1, 0, 1]
space = LagrangeFESpace(domain, p=1)

def penalty_matrix():
    """
    @brief 组装罚项矩阵 
    """

    # 空间次数
    p = space.p

    mesh = TriangleMesh.from_box(domain, nx=3, ny=3)
    GD = mesh.geo_dimension()
    TD = mesh.top_dimension()

    assert TD > 1   # 目前仅能处理 2D 和 3D 的问题
    assert GD == TD # 仅适用于网格拓扑维数和几何维数相同的情形

    NC = mesh.number_of_cells()
    NF = mesh.number_of_faces()

    isFaceDof = (mesh.multi_index_matrix(p,TD) == 0) 
    cell2face = mesh.cell_to_face()
    cell2facesign = mesh.cell_to_face_sign()

    ldof = space.number_of_local_dofs() # 单元上的所有的自由度的个数
    fdof = space.number_of_local_dofs('face') # 每个单元面上的自由度
    ndof = ldof - fdof
    face2dof = bm.zeros((NF, fdof + 2*ndof))

    if TD == 2: # 处理 2D 情形

        qf = mesh.quadrature_formula(p+3, 'face') # 面上的积分公式
        bcs, ws = qf.get_quadrature_points_and_weights()
        NQ = len(ws)

        n = mesh.face_unit_normal()
        cell2dof = space.cell_to_dof()
        # 每个积分点、在每个面上、每个基函数法向导数的取值
        val = bm.zeros((NF, NQ, fdof + 2*ndof)) 

        for i in range(TD+1): # 循环单元每个面

            lidx, = bm.nonzero( cell2facesign[:, i]) # 单元是全局面的左边单元
            ridx, = bm.nonzero(~cell2facesign[:, i]) # 单元是全局面的右边单元
            idx0, = bm.nonzero( isFaceDof[:, i]) # 在面上的自由度
            idx1, = bm.nonzero(~isFaceDof[:, i]) # 不在面上的自由度

            fidx = cell2face[:, i] # 第 i 个面的全局编号
            face2dof[fidx[lidx, None], bm.arange(fdof,      fdof+  ndof)] = cell2dof[lidx[:, None], idx1] 
            face2dof[fidx[ridx, None], bm.arange(fdof+ndof, fdof+2*ndof)] = cell2dof[ridx[:, None], idx1]

            # 面上的自由度按编号大小进行排序
            idx = bm.argsort(cell2dof[:, isFaceDof[:, i]], axis=1) 
            face2dof[fidx, 0:fdof] = cell2dof[:, isFaceDof[:, i]][bm.arange(NC)[:, None], idx] 


            # 面上的积分点转化为体上的积分点
            b = bm.insert(bcs, i, 0, axis=1)
            # (NC, NQ, cdof)
            cval = bm.einsum('cqlm, cm->cql', space.grad_basis(b), n[cell2face[:, i]])
            val[fidx[ridx, None],:, bm.arange(fdof+ndof, fdof+2*ndof)] = +cval[ridx[:, None],:, idx1]
            val[fidx[lidx, None],:, bm.arange(fdof,      fdof+  ndof)] = -cval[lidx[:, None],:, idx1]

            val[fidx[ridx, None],:, bm.arange(0, fdof)] += cval[ridx[:, None],:, idx0[idx[ridx, :]]]
            val[fidx[lidx, None],:, bm.arange(0, fdof)] -= cval[lidx[:, None],:, idx0[idx[lidx, :]]] 

        face2cell = mesh.face_to_cell()
        isInFace = face2cell[:, 0] != face2cell[:, 1]
        #  


        h = mesh.entity_measure('face', index=isInFace)
        f2d = face2dof[isInFace]

        P = bm.einsum('q, fqi, fqj, f->fij', ws, val[isInFace,:], val[isInFace,:], h*h)
        I = bm.broadcast_to(f2d[:, :, None], shape=P.shape)
        J = bm.broadcast_to(f2d[:, None, :], shape=P.shape)

        gdof = space.number_of_global_dofs()
        P = csr_matrix((P.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return P
a = penalty_matrix()
