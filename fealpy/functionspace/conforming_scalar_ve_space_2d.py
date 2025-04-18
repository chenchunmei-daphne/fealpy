
from .lagrange_fe_space import LagrangeFESpace

from typing import Optional, TypeVar, Union, Generic, Callable
from ..typing import TensorLike, Index, _S, Threshold

from ..backend import TensorLike
from ..backend import backend_manager as bm
from ..mesh.mesh_base import Mesh
from .space import FunctionSpace
from .dofs import LinearMeshCFEDof, LinearMeshDFEDof
from .function import Function
from fealpy.decorator import barycentric, cartesian

from .scaled_monomial_space_2d import ScaledMonomialSpace2d

class CSVEDof2d():
    def __init__(self, mesh, p):
        self.p = p
        self.mesh = mesh
        self.cell2dof = self.cell_to_dof() # 初始化的时候就构建出 cell2dof 数组
        
        NC = mesh.number_of_cells()
        ldof = self.number_of_local_dofs(doftype='all')
        self.cell2dofLocation = bm.zeros(NC+1, dtype=mesh.itype)
        self.cell2dofLocation[1:] = bm.add.accumulate(ldof)

    def is_boundary_dof(self, threshold=None):
        idx = self.mesh.ds.boundary_edge_index()
        if threshold is not None:
            bc = self.mesh.entity_barycenter('edge', index=idx)
            idx = threshold(bc)
            #idx  = idx[flag]
        gdof = self.number_of_global_dofs()
        isBdDof = bm.zeros(gdof, dtype=bm.bool_)
        edge2dof = self.edge_to_dof()
        isBdDof[edge2dof[idx]] = True
        return isBdDof

    def edge_to_dof(self, index=_S):
        return self.mesh.edge_to_ipoint(self.p, index=index)

    face_to_dof = edge_to_dof

    def cell_to_dof(self):
        return self.mesh.cell_to_ipoint(self.p)

    def number_of_global_dofs(self):
        return self.mesh.number_of_global_ipoints(self.p)

    def number_of_local_dofs(self, doftype='all'):
        return self.mesh.number_of_local_ipoints(self.p, iptype=doftype)

    def interpolation_points(self, index=_S):
        return self.mesh.interpolation_points(self.p, scale=0.3)


class ConformingScalarVESpace2d():
    def __init__(self, mesh, p=1):
        """
        p: the space order
        q: the index of integral formular
        bc: user can give a barycenter for every mesh cell
        """
        self.mesh = mesh
        self.p = p
        self.smspace = ScaledMonomialSpace2d(mesh, p)
        self.cellmeasure = self.smspace.cellmeasure
        self.dof = CSVEDof2d(mesh, p)

        self.itype = self.mesh.itype
        self.ftype = self.mesh.ftype
        self.stype = 'csvem' # 空间类型

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self, doftype='all'):
        return self.dof.number_of_local_dofs(doftype=doftype)

    def cell_to_dof(self, index=_S):
        return self.dof.cell2dof[index]

    def interpolation_points(self, index=_S):
        return self.dof.interpolation_points()

    def array(self, dim=None, dtype=bm.float64):
        gdof = self.number_of_global_dofs()
        if dim is None:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return bm.zeros(shape, dtype=dtype)

    def function(self, dim=None, array=None, dtype=bm.float64):
        return Function(self, dim=dim, array=array, coordtype='cartesian', dtype=dtype)

    def set_dirichlet_bc(self, gD, uh, threshold=None):
        """
        初始化解 uh  的第一类边界条件。
        """
        p = self.p
        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        end = NN + (p - 1)*NE
        ipoints = self.interpolation_points()
        isDDof = self.dof.is_boundary_dof(threshold=threshold)
        uh[isDDof] = gD(ipoints[:end][isDDof[:end]])
        return isDDof

    def project_to_smspace(self, uh, PI1):
        """
        Project a conforming vem function uh into polynomial space.
        """
        cell2dof = self.cell_to_dof()
        dim = len(uh.shape)
        p = self.p
        g = lambda x: x[0]@uh[x[1]]
        S = self.smspace.function(dim=dim)
        S[:] = bm.concatenate(list(map(g, zip(PI1, cell2dof))))
        return S

    def grad_recovery(self, uh):

        p = self.p
        smldof = self.smspace.number_of_local_dofs()
        NC = self.mesh.number_of_cells()
        h = self.smspace.cellsize

        s = self.project_to_smspace(uh,uh.PI1).reshape(-1, smldof)
        sx = bm.zeros((NC, smldof), dtype=self.ftype)
        sy = bm.zeros((NC, smldof), dtype=self.ftype)

        start = 1
        r = bm.arange(1, p+1)
        for i in range(p):
            sx[:, start-i-1:start] = r[i::-1]*s[:, start:start+i+1]
            sy[:, start-i-1:start] = r[0:i+1]*s[:, start+1:start+i+2]
            start += i+2

        sx /= h.reshape(-1, 1)
        sy /= h.reshape(-1, 1)

        cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation
        NC = len(cell2dofLocation) - 1
        #cd = bm.hsplit(cell2dof, cell2dofLocation[1:-1])
        cd = cell2dof
        #DD = bm.vsplit(uh.D, cell2dofLocation[1:-1])
        DD = uh.D

        f1 = lambda x: x[0]@x[1]
        sx = bm.concatenate(list(map(f1, zip(DD, sx))))
        sy = bm.concatenate(list(map(f1, zip(DD, sy))))

        ldof = self.number_of_local_dofs()
        w = bm.repeat(1/self.smspace.cellsize, ldof)
        sx *= w
        sy *= w

        uh = self.function(dim=2)
        ws = bm.zeros(uh.shape[0], dtype=self.ftype)
        bm.add.at(uh[:, 0], bm.concatenate(cell2dof), sx)
        bm.add.at(uh[:, 1], bm.concatenate(cell2dof), sy)
        bm.add.at(ws, bm.concatenate(cell2dof), w)
        uh /=ws.reshape(-1, 1)
        return uh

    def interpolation(self, u, HB=None):
        """
        u: 可以是一个连续函数， 也可以是一个缩放单项式函数
        """
        if HB is None:
            mesh = self.mesh
            NN = mesh.number_of_nodes()
            NE = mesh.number_of_edges()
            p = self.p
            ipoint = self.dof.interpolation_points()
            uI = self.function()
            uI[:NN+(p-1)*NE] = u(ipoint[:NN+(p-1)*NE])
            if p > 1:
                phi = self.smspace.basis
                def f(x, index):
                    return bm.einsum(
                            'ij, ij...->ij...',
                            u(x), phi(x, index=index, p=p-2))
                bb = self.mesh.integral(f, celltype=True)/self.smspace.cellmeasure[..., bm.newaxis]
                uI[NN+(p-1)*NE:] = bb.reshape(-1)
            return uI
        else:
            uh = self.smspace.interpolation(u, HB)

            cell2dof, cell2dofLocation = self.cell_to_dof()
            NC = len(cell2dofLocation) - 1
            cd = bm.hsplit(cell2dof, cell2dofLocation[1:-1])
            DD = bm.vsplit(self.D, cell2dofLocation[1:-1])

            smldof = self.smspace.number_of_local_dofs()
            f1 = lambda x: x[0]@x[1]
            uh = bm.concatenate(list(map(f1, zip(DD, uh.reshape(-1, smldof)))))

            ldof = self.number_of_local_dofs()
            w = bm.repeat(1/self.smspace.cellmeasure, ldof)
            uh *= w

            uI = self.function()
            ws = bm.zeros(uI.shape[0], dtype=self.ftype)
            bm.add.at(uI, cell2dof, uh)
            bm.add.at(ws, cell2dof, w)
            uI /=ws
            return uI
