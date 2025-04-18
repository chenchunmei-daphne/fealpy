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

