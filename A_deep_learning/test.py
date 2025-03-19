import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh

mesh = TriangleMesh.from_polygon_gmsh([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], 0.1)  # 不均匀的网格


fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
# mesh.find_node(axes,showindex=True,fontsize=20)  # 显示节点编号
# mesh.find_edge(axes, showindex=True,fontsize=20)  # 显示边编号
# mesh.find_cell(axes, showindex=True,fontsize=20)  # 显示单元编号
# plt.show()

cell = mesh.entity('cell')  # (NC, 3)
node = mesh.entity('node')  # (N, 2)
print(mesh.entity_measure('edge')[0:3])

def compute_aspect_ratio(cell, node):
    """
    计算网格中每个三角形单元的长宽比（Aspect Ratio）
    """
    aspect_ratios = []

    for tri in cell:
        # 获取三角形顶点的坐标
        p0, p1, p2 = node[tri]

        # 计算三边长度
        edges = np.array([
            np.linalg.norm(p1 - p0),
            np.linalg.norm(p2 - p1),
            np.linalg.norm(p0 - p2)
        ])

        # 最长边
        L_max = np.max(edges)

        # 使用海伦公式计算面积
        s = np.sum(edges) / 2  # 半周长
        area = np.sqrt(s * (s - edges[0]) * (s - edges[1]) * (s - edges[2]))

        # 计算高度 h = (2 * 面积) / 最长边
        h = 2 * area / L_max

        # 计算长宽比
        aspect_ratio = L_max / h
        aspect_ratios.append(aspect_ratio)

    return np.array(aspect_ratios)


# 计算所有单元的长宽比
aspect_ratios = compute_aspect_ratio(cell, node)

# 绘制长宽比的直方图
plt.figure(figsize=(8, 5))
plt.hist(aspect_ratios, bins=10, edgecolor='black', alpha=0.7)
plt.xlabel("Aspect Ratio")
plt.ylabel("Frequency")
plt.title("Histogram of Triangle Aspect Ratios")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()
print(aspect_ratios)