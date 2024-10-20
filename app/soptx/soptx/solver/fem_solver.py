from fealpy.backend import backend_manager as bm

from fealpy.typing import TensorLike

from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.fem.bilinear_form import BilinearForm
from fealpy.fem import DirichletBC
from fealpy.sparse import COOTensor

from fealpy.solver import cg, spsolve

from app.soptx.soptx.utils.timer import timer


class FEMSolver:
    def __init__(self, material_properties, tensor_space, pde):
        """
        Initialize the FEMSolver with the provided parameters.
        """
        self.material_properties = material_properties
        self.tensor_space = tensor_space
        self.pde = pde

    def assemble_stiffness_matrix(self):
        """
        Assemble the global stiffness matrix using the material properties and integrator.
        """
        integrator = LinearElasticIntegrator(material=self.material_properties, 
                                            q=self.tensor_space.p+3)
        # KE = integrator.assembly(space=self.tensor_space)
        bform = BilinearForm(self.tensor_space)
        bform.add_integrator(integrator)
        K = bform.assembly()

        return K
    
    def assemble_force_vector(self):
        """
        Assemble the global force vector using the force function.
        """
        force = self.pde.force
        F = self.tensor_space.interpolate(force)

        return F

    def apply_boundary_conditions(self, K: COOTensor, F: TensorLike) -> TensorLike:
        """
        Apply boundary conditions to the stiffness matrix and force vector.
        """
        # force = self.pde.force
        dirichlet = self.pde.dirichlet
        is_dirichlet_boundary_face = getattr(self.pde, 'is_dirichlet_boundary_face', None)
        is_dirichlet_boundary_edge = getattr(self.pde, 'is_dirichlet_boundary_edge', None)
        is_dirichlet_boundary_node = getattr(self.pde, 'is_dirichlet_boundary_node', None)
        is_dirichlet_boundary_dof = getattr(self.pde, 'is_dirichlet_boundary_dof', None)
        
        # K = self.assemble_stiffness_matrix()
        # F = self.tensor_space.interpolate(force)

        uh_bd = bm.zeros(self.tensor_space.number_of_global_dofs(), dtype=bm.float64)

        thresholds = []
        if is_dirichlet_boundary_face is not None:
            thresholds.append(is_dirichlet_boundary_face)
        if is_dirichlet_boundary_edge is not None:
            thresholds.append(is_dirichlet_boundary_edge)
        if is_dirichlet_boundary_node is not None:
            thresholds.append(is_dirichlet_boundary_node)
        if is_dirichlet_boundary_dof is not None:
            thresholds.append(is_dirichlet_boundary_dof)

        if thresholds:
            uh_bd, isDDof = self.tensor_space.boundary_interpolate(gD=dirichlet, 
                                                                uh=uh_bd, 
                                                                threshold=tuple(thresholds))

        F = F - K.matmul(uh_bd)
        F[isDDof] = uh_bd[isDDof]

        dbc = DirichletBC(space=self.tensor_space)
        K = dbc.apply_matrix(matrix=K, check=True)

        return K, F

    def solve(self, solver_method) -> TensorLike:
        """
        Solve the displacement field.
        """

        # tmr = timer("FEM Solver")
        # next(tmr)
        tmr = None

        K0 = self.assemble_stiffness_matrix()

        if tmr:
            tmr.send('Assemble Stiffness Matrix')

        F0 = self.assemble_force_vector()
        # if tmr:
        #     tmr.send('Assemble Force Vector')
        
        K, F = self.apply_boundary_conditions(K=K0, F=F0)
        if tmr:
            tmr.send('Apply Boundary Conditions')

        K = K.tocsr()
        
        uh = self.tensor_space.function()
        
        if solver_method == 'cg':
            uh[:] = cg(K, F, maxiter=5000, atol=1e-14, rtol=1e-14)
            if tmr:
                tmr.send('Solve System with CG')

        elif solver_method == 'mumps':
            raise NotImplementedError("Direct solver using MUMPS is not implemented.")
        
        elif solver_method == 'spsolve':
            uh[:] = spsolve(K, F)
            if tmr:
                tmr.send('Solve System with spsolve')

        else:
            raise ValueError(f"Unsupported solver method: {solver_method}")
        
        if tmr:
            tmr.send(None)

        return uh
