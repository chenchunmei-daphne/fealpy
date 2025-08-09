from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian

# 方程的常数项
k = bm.tensor(1.0)
beta = bm.tensor(1.001)
gamma = bm.sqrt(beta**2 - 1.0)

# 定义域
def domain():
    return [0.0, 1.0, 0.0, 1.0]

# 真解
@cartesian
def solution(p):
    x = p[..., 0]
    y = p[..., 1]
    val = bm.exp(1j * beta * k * y) * bm.exp(-k * gamma * (x + 1.0))
    return val

# 源项
@cartesian
def source(p):
    val =bm.zeros_like(p[..., 0], dtype=bm.complex128)
    return val

# 梯度
@cartesian
def gradient(p):
    
    u = solution(p)
    du_dx = -k * gamma * u
    du_dy = 1j * beta * k * u
    val = bm.stack((du_dx, du_dy), axis=-1)
    return val

# dirichlet 边界条件
@cartesian
def robin(p, n):
    kappa = 1j * k
    grad = gradient(p)
    val = bm.sum(grad * n[:, None, :], axis=-1)
    val += kappa * solution(p)
    return val

from fealpy.backend import backend_manager as bm

backend = 'numpy'
device = 'cpu'
bm.set_backend(backend)

from fealpy.utils import timer
from fealpy import logger

logger.setLevel('WARNING')
tmr = timer()
next(tmr)

from fealpy.mesh import TriangleMesh

mesh = TriangleMesh.from_box(domain(), nx=4, ny=4)
maxit = 4

errorMatrix = bm.zeros((1, maxit), dtype=bm.float64)

from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import BilinearForm, ScalarSourceIntegrator, ScalarRobinSourceIntegrator
from fealpy.fem import LinearForm, ScalarDiffusionIntegrator, ScalarRobinBCIntegrator, ScalarMassIntegrator
from fealpy.fem import DirichletBC
from fealpy.solver import cg

for i in range(maxit):
    space = LagrangeFESpace(mesh, p=1)
    tmr.send(f'第{i}次空间时间')
    uh = space.function(dtype=bm.complex128)

    D = ScalarDiffusionIntegrator(q=3)
    M = ScalarMassIntegrator(coef=-k**2,q=3)
    R = ScalarRobinBCIntegrator(coef=1j * k, q=3)
    bform = BilinearForm(space)
    bform.add_integrator(D)
    bform.add_integrator(M)
    bform.add_integrator(R)
    A = bform.assembly()

    lform = LinearForm(space)
    lform.add_integrator(ScalarSourceIntegrator(source, q=3))
    lform.add_integrator(ScalarRobinSourceIntegrator(robin, q=3))
    F = lform.assembly()

    tmr.send(f'第{i}次矩阵组装时间')
    gdof = space.number_of_global_dofs()
    # A, F = DirichletBC(space, gd = dirichlet).apply(A, F)
    # tmr.send(f'第{i}次边界处理时间')
    uh[:] = cg(A, F)
    tmr.send(f'第{i}次求解器时间')
    errorMatrix[0, i] = mesh.error(solution, uh.value)
    if i < maxit-1:
        mesh.uniform_refine(n=1)
    tmr.send(f'第{i}次误差计算及网格加密时间')
    next(tmr)
print("最终误差",errorMatrix)
print("order : ", bm.log2(errorMatrix[0,:-1]/errorMatrix[0,1:]))

from matplotlib import pyplot as plt

bc = bm.array([[1/3, 1/3, 1/3]], dtype=bm.float64)
ps = mesh.bc_to_point(bc)
u = solution(ps)
uh0 = uh(bc)

fig, axes = plt.subplots(2, 2)
mesh.add_plot(axes[0, 0], cellcolor=u.real, linewidths=0)
axes[0, 0].set_title('真解的实部', fontname='Microsoft YaHei')
mesh.add_plot(axes[0, 1], cellcolor=uh0.real, linewidths=0)
axes[0, 1].set_title('数值解的实部', fontname='Microsoft YaHei')
mesh.add_plot(axes[1, 0], cellcolor=u.imag, linewidths=0)
axes[1, 0].set_title('真解的虚部', fontname='Microsoft YaHei')
mesh.add_plot(axes[1, 1], cellcolor=uh0.imag, linewidths=0)
axes[1, 1].set_title('数值解的虚部', fontname='Microsoft YaHei')
plt.show()