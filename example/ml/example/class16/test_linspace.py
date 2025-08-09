import numpy as np
import matplotlib.pyplot as plt
from fealpy.backend import bm
bm.set_backend("pytorch")
from fealpy.ml.sampler.sampler import ISampler, BoxBoundarySampler
from fealpy.ml.sampler import functional as F

domain = (-1, 1, -1, 1)
domain = (0, 1, 0, 1, 0, 1)
domain = (0, 1)
# iso = ISampler(domain, mode='random')

iso = BoxBoundarySampler(domain, mode='linspace', boundary=(0,))
p = iso.run(10)


# unique_points, counts = np.unique(p, axis=0, return_counts=True)

# num_repeats = np.sum(counts > 1)  # 重复点的数量
# total_repeats = np.sum(counts) - len(unique_points)  # 所有重复出现的总次数

# print(f"重复点的数量: {num_repeats}")
# print(f"所有重复出现的总次数: {total_repeats}")
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
# print(p)
print(p.shape)