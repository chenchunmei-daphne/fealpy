
import argparse
import matplotlib.pyplot as plt
import time

from fealpy import logger
logger.setLevel('WARNING')
from fealpy.backend import backend_manager as bm
from fealpy.pde.helmholtz_2d import HelmholtzData2d
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import ScalarDiffusionIntegrator, ScalarMassIntegrator, ScalarRobinBCIntegrator 
from fealpy.fem import ScalarFaceInteriorPenaltyIntegrator    
from fealpy.fem import ScalarSourceIntegrator, ScalarRobinSourceIntegrator         
from fealpy.fem import BilinearForm, LinearForm
from fealpy.solver import cg
from fealpy.utils import timer

# 记录程序开始时间
start_time0 = time.time()

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        在三角形网格上使用任意次有限元方法求解二维 Helmholtz 方程 
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--wavenum', 
        default=1, type=int,
        help='模型的波数, 默认为 1.')

parser.add_argument('--ns',
        default=20, type=int,
        help='初始网格 x 和 y 方向剖分段数, 默认 20 段.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

parser.add_argument('--backend',
        default='numpy', type=str,
        help='默认后端为numpy')

parser.add_argument('--device',
        default='cpu', type=str,
        help='默认cpu上运行')

parser.add_argument('--gamma',
        default=0.1 + 1j * 0.1, type=float,
        help='默认内罚参数为100.0')


args = parser.parse_args()
# args.backend = 'pytorch'
bm.set_backend(args.backend)
p =args.degree
ns = args.ns
maxit = args.maxit
k = args.wavenum
kappa = k * 1j
gamma = args.gamma

tmr = timer()
next(tmr)

pde = HelmholtzData2d(k=k) 
domain = pde.domain()

errorMatrix = bm.zeros((4, maxit), dtype=bm.float64)

D = ScalarDiffusionIntegrator(coef=1, q=p+2)
M = ScalarMassIntegrator(coef=-k**2, q=p+2)
R = ScalarRobinBCIntegrator(coef=kappa, q=p+2)
G = ScalarFaceInteriorPenaltyIntegrator(coef=gamma, q=p+2)

f = ScalarSourceIntegrator(pde.source, q=p+2)
Vr = ScalarRobinSourceIntegrator(pde.robin, q=p+2)

NDof = bm.zeros(maxit, dtype=bm.int64, device=args.device)

# 记录循环开始时间
start_time = time.time()

for i in range(maxit):

    n = ns*(2**i)
    mesh = TriangleMesh.from_box(domain, nx=n, ny=n)
    mesh.ftype = complex
    space = LagrangeFESpace(mesh, p=p)
    NDof[i] = space.number_of_global_dofs()
    tmr.send(f'第{i}次空间时间')

    b = BilinearForm(space)
    b.add_integrator([D, M])
    b.add_integrator(R)
    b.add_integrator(G)

    l = LinearForm(space)
    l.add_integrator(f)
    l.add_integrator([Vr])

    A = b.assembly() 
    F = l.assembly()
    tmr.send(f'第{i}次矩组装时间')  
    
    uh = space.function(dtype=bm.complex128)
    uh[:] = cg(A, F)
    tmr.send(f'第{i}次求解器时间')
    uI = space.interpolate(pde.solution)
    
    D.clear()
    M.clear()
    R.clear()
    f.clear()
    Vr.clear()
    G.clear()

    errorMatrix[0, i] = mesh.error(pde.solution, uI)
    errorMatrix[1, i] = mesh.error(pde.gradient, uI.grad_value)
    errorMatrix[2, i] = mesh.error(pde.solution, uh, q=p+2)
    errorMatrix[3, i] = mesh.error(pde.gradient, uh.grad_value, q=p+2)
    tmr.send(f'第{i}次误差计算及网格加密时间')

next(tmr)

# 记录循环结束时间
end_time = time.time()

# 计算循环总运行时间
total_time = end_time - start_time
print(f"IPFEM loop total running time: {total_time} s")

bc = bm.array([[1/3, 1/3, 1/3]], dtype=bm.float64)
ps = mesh.bc_to_point(bc)
u = pde.solution(ps)
uI = uI(bc)
uh = uh(bc)


print('误差矩阵:', errorMatrix, sep='\n')
print('误差阶:', errorMatrix[:, 0:-1]/errorMatrix[:, 1:], sep='\n')

fig, axes = plt.subplots(2, 2)
axes[0, 0].set_title('Real Part of True Solution')
axes[0, 1].set_title('Imag Part of True Solution')
axes[1, 0].set_title("Real Part of Numerical Solution")
axes[1, 1].set_title("Imag Part of Numerical  Solution")
mesh.add_plot(axes[0, 0], cellcolor=bm.real(u), linewidths=0)
mesh.add_plot(axes[0, 1], cellcolor=bm.imag(u), linewidths=0) 
mesh.add_plot(axes[1, 0], cellcolor=bm.real(uh), linewidths=0)
mesh.add_plot(axes[1, 1], cellcolor=bm.imag(uh), linewidths=0) 

# 添加整个图的标题
plt.suptitle("Interior Penalty Finite Element Method ")

# 调整子图间距防止标题重叠
plt.tight_layout()

# 记录循环结束时间
end_time0 = time.time()
# 计算总运行时间
total_time0 = end_time0 - start_time0
print(f"IPFEM total running time: {total_time0} s")

plt.show()