from typing import Optional, Literal
from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Threshold
from fealpy.mesh import HomogeneousMesh
from fealpy.functionspace.space import FunctionSpace as _FS
from fealpy.utils import process_coef_func
from fealpy.functional import bilinear_integral
from fealpy.fem.integrator import (
    LinearInt, OpInt, FaceInt,
    enable_cache,
    assemblymethod,
    CoefLike
)
class _PenaltyMassIntegrator(LinearInt, OpInt, FaceInt):
    def __init__(self, coef: Optional[CoefLike] = None, q: Optional[int] = None, *,
                 region: Optional[TensorLike] = None,
                 batched: bool = False,
                 method: Literal['fast', None] = None) -> None:
        super().__init__(method=method if method else 'assembly')
        self.coef = coef
        self.q = q
        self.set_region(region)
        self.batched = batched

    @enable_cache
    def to_global_dof(self, space: _FS, /, indices: Optional[Index] = None) -> TensorLike:
        index = self.entity_selection(indices)
        mesh = getattr(space, 'mesh', None)
        p = getattr(space, 'p', None)

        TD = mesh.top_dimension()
        NF = mesh.number_of_faces()
        NC = mesh.number_of_cells()

        isFaceDof = (mesh.multi_index_matrix(p, TD)) == 0
        cell2face = mesh.cell_to_face()
        cell2facesign = mesh.cell_to_face_sign()

        ldof = space.number_of_local_dofs() 
        fdof = space.number_of_local_dofs('face') 
        ndof = ldof - fdof
        face2dof = bm.zeros((NF, fdof + 2*ndof), dtype=bm.int32)
        cell2dof = space.cell_to_dof()

        for i in range(TD+1):
            lidx, = bm.nonzero(cell2facesign[:, i]) 
            ridx, = bm.nonzero(~cell2facesign[:, i]) 
            idx0, = bm.nonzero(isFaceDof[:, i]) 
            idx1, = bm.nonzero(~isFaceDof[:, i]) 

            fidx = cell2face[:, i] 
            face2dof[fidx[lidx, None], bm.arange(fdof, fdof+ndof)] = cell2dof[lidx[:, None], idx1] 
            face2dof[fidx[ridx, None], bm.arange(fdof+ndof, fdof+2*ndof)] = cell2dof[ridx[:, None], idx1]

            idx = bm.argsort(cell2dof[:, isFaceDof[:, i]], axis=1) 
            face2dof[fidx, 0:fdof] = cell2dof[:, isFaceDof[:, i]][bm.arange(NC)[:, None], idx] 

        return face2dof[index]

    @enable_cache
    def fetch_qf(self, space: _FS, /, indices: Optional[Index] = None):
        q = space.p+3 if self.q is None else self.q
        mesh = space.mesh
        qf = mesh.quadrature_formula(q, 'face')
        return qf.get_quadrature_points_and_weights()

    @enable_cache
    def fetch_measure(self, space: _FS, /, indices: Optional[Index] = None):
        mesh = space.mesh
        return mesh.entity_measure('face', index=self.entity_selection(indices))

    @enable_cache
    def fetch_basis(self, space: _FS, /, indices: Optional[Index] = None):
        bcs, ws = self.fetch_qf(space, indices)
        mesh = space.mesh
        p = space.p
        index = self.entity_selection(indices)

        TD = mesh.top_dimension()
        NF = mesh.number_of_faces()
        NQ = len(ws)

        ldof = space.number_of_local_dofs() 
        fdof = space.number_of_local_dofs('face') 
        ndof = ldof - fdof
        cell2face = mesh.cell_to_face()
        cell2dof = space.cell_to_dof()
        isFaceDof = (mesh.multi_index_matrix(p, TD) == 0)
        cell2facesign = mesh.cell_to_face_sign()
        n = mesh.face_unit_normal()

        phi = bm.zeros((NF, NQ, fdof + 2*ndof), dtype=bm.float64)
        for i in range(TD+1):
            lidx, = bm.nonzero(cell2facesign[:, i]) 
            ridx, = bm.nonzero(~cell2facesign[:, i]) 
            idx0, = bm.nonzero(isFaceDof[:, i]) 
            idx1, = bm.nonzero(~isFaceDof[:, i]) 

            fidx = cell2face[:, i] 
            idx = bm.argsort(cell2dof[:, isFaceDof[:, i]], axis=1) 

            b = bm.insert(bcs, i, 0, axis=1)

            cval = bm.einsum('cqlm, cm->cql', space.grad_basis(b), n[cell2face[:, i]])
            phi[fidx[ridx, None], :, bm.arange(fdof+ndof, fdof+2*ndof)] = +cval[ridx[:, None], :, idx1]
            phi[fidx[lidx, None], :, bm.arange(fdof, fdof+ndof)] = -cval[lidx[:, None], :, idx1]

            phi[fidx[ridx, None], :, bm.arange(0, fdof)] += cval[ridx[:, None], :, idx0[idx[ridx, :]]]
            phi[fidx[lidx, None], :, bm.arange(0, fdof)] -= cval[lidx[:, None], :, idx0[idx[lidx, :]] 
        
        return phi[index]

    def assembly(self, space: _FS, /, indices: Optional[Index] = None) -> TensorLike:
        coef = self.coef
        mesh = space.mesh
        bcs, ws = self.fetch_qf(space, indices)
        cm = self.fetch_measure(space, indices)
        phi = self.fetch_basis(space, indices)
        index = self.entity_selection(indices)

        val = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='face', index=index)
        return bilinear_integral(phi, phi, ws, cm*cm, val, batched=self.batched)

class InnerPenaltyMassIntegrator(_PenaltyMassIntegrator):
    def entity_selection(self, indices: Optional[Index] = None) -> TensorLike:
        if indices is not None:
            return super().entity_selection(indices)
        
        mesh = self.mesh
        face2cell = mesh.face_to_cell()
        index = face2cell[:, 0] != face2cell[:, 1]
        
        if callable(self.region):
            bc = mesh.entity_barycenter('face', index=index)
            index = index[self.region(bc)]
        
        return index

