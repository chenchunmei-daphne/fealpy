from fealpy.backend import bm
bm.set_backend("numpy")

import matplotlib.pyplot as plt
from fealpy.ml.sampler import ISampler, BoxBoundarySampler

domain = (-1, 1, -1, 1)
domain = (0, 1, 0, 1, 0, 1)
domain = (0, 1)

si = ISampler(domain, mode="linspace")
p = si.run(4)

# sout = BoxBoundarySampler(domain)
# sp = sout.run(2)
print(p.shape)

# exit()
# 可视化
if p.shape[1] == 1:
    print(p)

# 可视化
if p.shape[1] == 2: 
    plt.figure(figsize=(6, 6))
    plt.scatter(p[:, 0], p[:, 1], c='red', s=50)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Sampled Points in Domain')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().set_aspect('equal')  # 保证坐标轴比例一致
if p.shape[1] == 3: 

    # 创建3D图形
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')  # 使用3D投影
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c='red', s=50)

    # 设置坐标轴标签
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')  # 新增z轴标签

    # 设置标题和网格
    ax.set_title('3D Sampled Points in Domain')
    ax.grid(True, linestyle='--', alpha=0.5)

plt.show()