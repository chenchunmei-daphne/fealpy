import math

from typing import Optional

from ..backend import backend_manager as bm
from ..backend import TensorLike
from ..sparse import csr_matrix, spdiags, SparseTensor
from ..mesh import UniformMesh

from .operator_base import OpteratorBase, assemblymethod

class EllipticOperator(OpteratorBase):
    """
    """
    def __init__(self, mesh: UniformMesh, 
                 diffusion_coef,
                 convection_coef,
                 reaction_coef,
                 method: Optional[str]=None):
        method = 'assembly' if (method is None) else method
        super().__init__(method=method)

        self.mesh = mesh  # Store the mesh for later assembly
        self.diffusion_coef = diffusion_coef 
        self.convection_coef = convection_coef
        self.reaction_coef = reaction_coef

    def assembly(self) -> SparseTensor:
        """
        """
        pde = self.pde
        A = self.assembly_diffusion()
        if hasattr(pde, 'convection_coef'):
            A += self.assembly_convection()
        if hasattr(pde, 'reaction_coef'): 
            A += self.assembly_reaction()
        return A

    def assembly_diffusion(self) -> SparseTensor:
        """
        Assemble the global sparse matrix representing the diffusion operator.

        Returns:
            csr_matrix: Sparse matrix of size (NN, NN), where NN is number of nodes.
        """
        mesh = self.mesh
        ftype = mesh.ftype  # Floating point data type for matrix entries
        itype = mesh.itype  # Integer data type for indexing (not used directly)
        device = mesh.device  # Device context (e.g., CPU, GPU)
        GD = mesh.geo_dimension()  # Geometric dimension of the mesh

        node = self.mesh.entity('node')
        D = self.diffusion_coef(node) # shape == (GD, GD)

        # spacing of the mesh in each dimension
        h = mesh.h
        # coefficient c = 1/h^2 per dimension
        c = 1.0 / (h ** 2)
        c = D @ c

        NN = mesh.number_of_nodes()  # Total number of grid nodes
        K = mesh.linear_index_map('node')  # Multi-dimensional to linear index map
        shape = K.shape  # Shape of the index map array

        # Create diagonal entries with sum of c over dimensions times 2
        diag_value = bm.full(NN, 2 * c.sum(), dtype=ftype)
        I = K.flat  # Row indices for diagonal entries
        J = K.flat  # Column indices for diagonal entries
        A = csr_matrix((diag_value, (I, J)), shape=(NN, NN))

        # Slices tuple for indexing all dimensions
        full_slice = (slice(None),) * GD

        # Off-diagonal contributions for each dimension
        for i in range(GD):
            # Number of nodes shifted along dimension i
            n_shift = math.prod(
                count for dim_idx, count in enumerate(shape) if dim_idx != i
            )
            # Off-diagonal value for neighbor entries
            off_value = bm.full(NN - n_shift, -c[i], dtype=ftype)
            # Create slice objects to select neighbor index arrays
            s1 = full_slice[:i] + (slice(1, None),) + full_slice[i+1:]
            s2 = full_slice[:i] + (slice(None, -1),) + full_slice[i+1:]
            # Row indices for off-diagonal
            I = K[s1].flat
            J = K[s2].flat
            # Add entries for coupling in both directions
            A += csr_matrix((off_value, (I, J)), shape=(NN, NN))
            A += csr_matrix((off_value, (J, I)), shape=(NN, NN))

        return A

    def assembly_convection(self) -> SparseTensor:
        """
        """
        mesh = self.mesh
        ftype = mesh.ftype  # Floating point data type for matrix entries
        itype = mesh.itype  # Integer data type for indexing (not used directly)
        device = mesh.device  # Device context (e.g., CPU, GPU)
        GD = mesh.geo_dimension()  # Geometric dimension of the mesh

        node = self.mesh.entity('node')
        b = self.convection_coef(node) # shape == (GD, )

        # spacing of the mesh in each dimension
        h = mesh.h
        c = b / h / 2.0 

        NN = mesh.number_of_nodes()  # Total number of grid nodes
        K = mesh.linear_index_map('node')  # Multi-dimensional to linear index map
        shape = K.shape  # Shape of the index map array

        # Slices tuple for indexing all dimensions
        full_slice = (slice(None),) * GD

        val = bm.zeros(NN, dtype=ftype)
        A = spdiags(val, 0, NN, NN, format='csr') 
        
        # Off-diagonal contributions for each dimension
        for i in range(GD):
            # Number of nodes shifted along dimension i
            n_shift = math.prod(
                count for dim_idx, count in enumerate(shape) if dim_idx != i
            )
            # Off-diagonal value for neighbor entries
            off_value = bm.full(NN - n_shift, c[i], dtype=ftype)
            # Create slice objects to select neighbor index arrays
            s1 = full_slice[:i] + (slice(1, None),) + full_slice[i+1:]
            s2 = full_slice[:i] + (slice(None, -1),) + full_slice[i+1:]
            # Row indices for off-diagonal
            I = K[s1].flat
            J = K[s2].flat
            # Add entries for coupling in both directions
            A += csr_matrix((-off_value, (I, J)), shape=(NN, NN))
            A += csr_matrix(( off_value, (J, I)), shape=(NN, NN))
        return A

    def assembly_reaction(self) -> SparseTensor:
        """
        """
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        c = self.reaction_coef(mesh.entity('node'))
        val = bm.full(NN, c, dtype=mesh.ftype)
        D = spdiags(val, 0, NN, NN, format='csr')
        return D
