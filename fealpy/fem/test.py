from typing import Optional, Literal, Union
# from ..mesh.mesh_base import Mesh, SimplexMesh
# from ..backend import backend_manager as bm
# from ..typing import TensorLike, Index
# from ..functionspace.space import FunctionSpace as _FS
# from .integrator import (
#     LinearInt, OpInt, FaceInt,
#     enable_cache,
#     assemblymethod,
#     CoefLike
# )


# class InteriorPenaltyMatrixIntegrator(LinearInt, OpInt, FaceInt):
#     r"""
#     内罚有限元罚项矩阵积分子,支持Helmholtz方程和2D/3D问题
    
#     参数:
#         coef: 罚项系数，可以是实数或复数
#         q: 积分阶数
#         region: 积分区域
#         batched: 是否使用批处理模式
#         method: 积分方法 ('fast', 'assembly')
#         complex_mode: 是否处理复数情况
#     """
#     def __init__(self, 
#                  coef: Optional[Union[CoefLike, complex]] = None, 
#                  q: Optional[int] = None, 
#                  *,
#                  region: Optional[TensorLike] = None,
#                  batched: bool = False,
#                  method: Literal['fast', 'nonlinear', 'isopara', None] = None,
#                  complex_mode: bool = False) -> None:

#         super().__init__(method=method if method else 'assembly')
#         self.coef = coef
#         self.q = q
#         self.set_region(region)  # 设置积分区域
#         self.batched = batched

#     @enable_cache
#     def to_global_dof(self, space: _FS, /, indices=None) -> TensorLike:
#         """获取面(边)上的全局自由度索引"""
#         return space.face_to_dof(index=self.entity_selection(indices))
    
# A = InteriorPenaltyMatrixIntegrator(1, q=1)                                                                                              
# a = A.to_global_dof()
# print(a)

'''
@assemblymethod('fast')
    def fast_assembly(self, space: _FS, /, indices=None) -> TensorLike:
        """
        快速组装方法(针对单纯形网格优化)，支持Helmholtz方程
        
        参数:
            space: 有限元空间
            indices: 选择的面的索引
            
        返回:
            罚项矩阵(CSR格式)
        """
        mesh = space.mesh
        TD = mesh.top_dimension()
        GD = mesh.geo_dimension()
        
        if TD != GD:
            raise ValueError("只支持拓扑维度和几何维度相同的网格")
            
        # 如果不是单纯形网格，回退到标准组装方法
        if not isinstance(mesh, SimplexMesh):
            return self.assembly(space, indices=indices)
            
        # 获取积分点和权重
        bcs, ws = self.fetch_qf(space)
        # 获取面的度量
        fm = self.fetch_measure(space, indices)
        # 获取单位法向量
        n = self.fetch_face_unit_normal(space, indices)
        
        # 获取参考空间梯度
        gphi_ref = space.grad_basis(bcs, index=self.entity_selection(indices), variable='u')
        
        # 计算Jacobian矩阵和第一基本形式
        J = mesh.jacobi_matrix(bcs, index=self.entity_selection(indices), etype='face')
        G = mesh.first_fundamental_form(J)
        G_inv = bm.linalg.inv(G)
        scale = bm.sqrt(bm.linalg.det(G))
        
        # 将梯度从参考空间转换到物理空间
        gphi_physical = bm.einsum('qfdi, fdij, f -> qfdj', gphi_ref, G_inv, scale)
        # 计算法向导数
        gphi_n = bm.einsum('qfdj, fj -> qfd', gphi_physical, n)
        
        # 处理内部面和罚项系数
        index = self.entity_selection(indices)
        is_inner_face = ~mesh.boundary_face_flag()
        if indices is not None:
            is_inner_face = is_inner_face[indices]
        inner_index = index[is_inner_face[index]]
        
        coef = self._process_coef(self.coef, bcs, mesh, index)
        
        # 计算罚项矩阵
        if self.complex_mode:
            # 复数情况: 分别计算实部和虚部
            P_real = bm.einsum('q, qfi, qfj, f, f -> fij', 
                             ws, gphi_n.real, gphi_n.real, fm, coef.real)
            P_imag = bm.einsum('q, qfi, qfj, f, f -> fij', 
                             ws, gphi_n.imag, gphi_n.imag, fm, coef.imag)
            P = P_real + P_imag
        else:
            # 实数情况
            P = bm.einsum('q, qfi, qfj, f, f -> fij', 
                         ws, gphi_n, gphi_n, fm, coef)
        
        # 组装稀疏矩阵
        face2dof = space.face_to_dof(index=inner_index)
        I = bm.broadcast_to(face2dof[:, :, None], shape=P.shape)
        J = bm.broadcast_to(face2dof[:, None, :], shape=P.shape)
        
        gdof = space.number_of_global_dofs()
        from ..backend import csr_matrix, is_complex
        # 确定矩阵数据类型
        dtype = np.complex128 if (self.complex_mode or is_complex(coef)) else np.float64
        # 创建CSR格式稀疏矩阵
        P = csr_matrix((P.ravel(), (I.ravel(), J.ravel())), shape=(gdof, gdof), dtype=dtype)
        
        return P
'''
from fealpy.mesh import TriangleMesh, TetrahedronMesh
from fealpy.functionspace import LagrangeFESpace
mesh = TriangleMesh.from_box()
space = LagrangeFESpace(mesh=mesh)

# space 
print(space.face_to_dof().shape, space.face_to_dof()[0])

# mesh
print(mesh.quadrature_formula(q=3, etype='face'))
print(mesh.number_of_faces())
print(mesh.face_unit_normal().shape)
'''
    @enable_cache
    def to_global_dof(self, space: _FS, /, indices=None) -> TensorLike:
        """获取面(边)上的全局自由度索引"""
        return space.face_to_dof(index=self.entity_selection(indices))

    @enable_cache
    def fetch_qf(self, space: _FS):
        """获取积分公式(积分点和权重)"""
        mesh = space.mesh
        p = space.p
        # 默认积分阶数为p+3
        q = p + 3 if self.q is None else self.q  
        # 获取面的积分公式
        qf = mesh.quadrature_formula(q, 'face')
        # 返回积分点坐标和权重
        bcs, ws = qf.get_quadrature_points_and_weights()
        return bcs, ws
'''

