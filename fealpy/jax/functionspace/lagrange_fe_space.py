import numpy as np
import jax
import jax.numpy as jnp

from .. import logger

class LinearMeshCFEDof():
    def __init__(self, mesh, p):
        TD = mesh.top_dimension()
        self.mesh = mesh
        self.p = p
        self.multiIndex = mesh.multi_index_matrix(p, TD) 
        self.cell2dof = self.cell_to_dof()

    def is_boundary_dof(self, threshold=None):
        TD = self.mesh.top_dimension()
        gdof = self.number_of_global_dofs()
        if type(threshold) is np.ndarray:
            index = threshold
            if (index.dtype == np.bool_) and (len(index) == gdof):
                return index
        else:
            index = self.mesh.ds.boundary_face_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter(TD-1, index=index)
                flag = threshold(bc)
                index = index[flag]

        edge2dof = self.edge_to_dof(index=index) # 只获取指定的面的自由度信息
        face2dof = self.edge_to_dof(index=index) # 只获取指定的面的自由度信息
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        isBdDof[face2dof] = True
        return isBdDof

    def edge_to_dof(self, index=np.s_[:]):
        return self.mesh.edge_to_ipoint(self.p, index=index)

    def face_to_dof(self, index=np.s_[:]):
        return self.mesh.edge_to_ipoint(self.p, index=index)

    def cell_to_dof(self, index=np.s_[:]):
        return self.mesh.cell_to_ipoint(self.p, index=index)

    def interpolation_points(self, index=np.s_[:]):
        return self.mesh.interpolation_points(self.p, index=index)

    def number_of_global_dofs(self):
        return self.mesh.number_of_global_ipoints(self.p)

    def number_of_local_dofs(self, doftype='cell'):
        return self.mesh.number_of_local_ipoints(self.p, iptype=doftype)

class LagrangeFESpace():

    def __init__(self, mesh, p=1, ctype='C'):
        self.mesh = mesh
        self.p = p

        assert ctype in {'C', 'D'}
        self.ctype = ctype # 空间连续性类型
        if ctype == 'C':
            self.dof = LinearMeshCFEDof(mesh, p)

        logger.info(f"Initialize space with {self.dof.number_of_global_dofs()} global dofs")

        self.ftype = mesh.ftype
        self.itype = mesh.itype
        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()

    def number_of_local_dofs(self, doftype='cell'):
        return self.dof.number_of_local_dofs(doftype=doftype)

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def cell_to_dof(self):
        return self.dof.cell2dof

    def face_to_dof(self):
        return self.dof.face_to_dof()

    def basis(self, bc, index=jnp.s_[:], variable='u'):
        return self.mesh.shape_function(bc, p=self.p, variable=variable)

    def grad_basis(self, bc, index=jnp.s_[:], variable='u'):
        """
        @brief
        """
        return self.mesh.grad_shape_function(bc, p=self.p, index=index, variable=variable)

    def hess_basis(self, bc, index=jnp.s_[:], variable='u'):
        """
        @brief
        """
        return self.mesh.hess_shape_function(bc, p=self.p, index=index, variable=variable)


    def value(self, uh, bc, index=jnp.s_[:]):
        """
        @brief
        """
        pass
