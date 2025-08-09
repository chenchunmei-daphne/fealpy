import matplotlib.pyplot as plt
import torch

def show_heatmaps(matrices,titles=None, xlabel='X', ylabel='Y', figsize=(5,5), cmap="Reds"):
    ''' matrices是2维 '''

    matrices = matrices.numpy()
    fig, ax = plt.subplots()

    # 使用imshow函数绘制热力图
    cax = ax.imshow(matrices, cmap=cmap)

    # 添加颜色条
    fig.colorbar(cax)

    if titles is not None:
        ax.set_title(titles)
    # 设置坐标轴标签
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.show()

m = torch.eye(10)
m = torch.softmax(torch.rand(10, 10), dim=0)
show_heatmaps(m, titles='eye(10)', xlabel="Keys", ylabel="Queries")