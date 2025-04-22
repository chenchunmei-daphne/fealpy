from typing import Optional, Literal, Union
from fealpy.mesh.mesh_base import Mesh, SimplexMesh
from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Index
from fealpy.utils import process_coef_func
from fealpy.functionspace.space import FunctionSpace as _FS

from .integrator import (
    LinearInt, OpInt, FaceInt,
    enable_cache,
    assemblymethod,
    CoefLike
)


class InteriorPenaltyMatrixIntegrator(LinearInt, OpInt, FaceInt):
    r"""
    内罚有限元罚项矩阵积分子,支持Helmholtz方程和2D/3D问题
    
    参数:
        coef: 罚项系数，可以是实数或复数
        q: 积分阶数
        region: 积分区域
        batched: 是否使用批处理模式
        method: 积分方法 ('fast', 'assembly')
        complex_mode: 是否处理复数情况
    """
    def __init__(self, 
                 coef: Optional[Union[CoefLike, complex]] = None, 
                 q: Optional[int] = None, 
                 *,     
                 batched: bool = False,
                 method: Literal['fast', 'assembly', None] = None) -> None:
        super().__init__(method=method if method else 'assembly')
        self.coef = coef  
        self.q = q
        self.batched = batched

    @enable_cache
    def to_global_dof(self, space: _FS, /, indices=None) -> TensorLike:
        """获取面(边)上的全局自由度索引"""
        return space.face_to_dof(index=self.entity_selection(indices))

    @enable_cache
    def fetch_qf(self, space: _FS):
        """获取积分公式(积分点和权重)"""
        mesh = space.mesh
        p = space.p
        q = p + 3 if self.q is None else self.q  
        qf = mesh.quadrature_formula(q, 'face') # 获取面的积分公式
        bcs, ws = qf.get_quadrature_points_and_weights()  # 返回积分点坐标和权重
        return bcs, ws

    @enable_cache
    def fetch_measure(self, space: _FS, /, indices=None):
        """获取面的度量(长度/面积)"""
        mesh = space.mesh
        return mesh.entity_measure('face', index=self.entity_selection(indices))

    @enable_cache
    def fetch_face_unit_normal(self, space: _FS, /, indices=None):
        """获取面的单位法向量"""
        mesh = space.mesh
        return mesh.face_unit_normal(index=self.entity_selection(indices))

    @enable_cache
    def fetch_face_grad_basis(self, space: _FS, /, indices=None):
        """获取基函数在面上的梯度"""
        bcs = self.fetch_qf(space)[0]  # 获取积分点坐标
        return space.grad_basis(bcs, index=self.entity_selection(indices), variable='x')

    def _process_coef(self, coef, bcs, mesh, index):
        """
        处理罚项系数，支持复数
        
        参数:
            coef: 输入的罚项系数
            bcs: 重心坐标
            mesh: 网格对象
            index: 面索引
            
        返回:
            处理后的系数值
        """
        # 默认系数为1.0(实数或复数)
        if coef is None:
            return 1.0 if not self.complex_mode else 1.0 + 0j
        
        # 如果是简单数值类型，直接返回
        if isinstance(coef, (float, int, complex)):
            return coef
        
        # 处理空间变化的系数
        coef = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='face', index=index)
        
        # 如果需要复数模式但系数是实数，转换为复数
        if self.complex_mode and bm.isreal(coef).all():
            coef = coef + 0j  
            
        return coef

    @assemblymethod('assembly')
    def assembly(self, space: _FS, /, indices=None) -> TensorLike:
        pass
    

from fealpy.mesh import TriangleMesh, TetrahedronMesh
from fealpy.functionspace import LagrangeFESpace
mesh = TriangleMesh.from_one_triangle()
space = LagrangeFESpace(mesh=mesh)

print(mesh.entity('node'))

# space 
print(space.face_to_dof().shape, space.face_to_dof())

# mesh
print(mesh.quadrature_formula(q=3, etype='face'))
print(mesh.number_of_faces())
print(mesh.face_unit_normal())