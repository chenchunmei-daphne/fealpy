import matplotlib.pyplot as plt

from fealpy.backend import backend_manager as bm
bm.set_backend('pytorch')

def plot_error(**errors):
    """
    Plot a line chart comparing different error metrics.

    Parameters:
        **errors: Keyword arguments where each key is the name of an error metric
                 and the corresponding value is the list of error values.
                 Example: plot_error(train_error=train_err, test_error=test_err)

    Returns:
        matplotlib.figure.Figure: The figure object containing the plotted error curves.
    """

    var_names = list(errors.keys())  # 获取调用该函数时的变量名
    n = len(errors)

    if n == 1:
        fig = plt.figure()
        axes = fig.gca()
        # 假设每个损失列表的长度相同，这里使用第一个列表的长度
        y1 = range(1, 10 * len(next(iter(errors.values()))) + 1, 10)
        label_name = var_names[0]
        axes.plot(y1, list(errors.values())[0], label=label_name)
        axes.legend()  # 添加图例
    else:
        fig, axes = plt.subplots(1, n, figsize=(8, 7))
        for i, (name, error_list) in enumerate(errors.items()):
            y1 = range(1, 10 * len(error_list) + 1, 10)
            axes[i].plot(y1, error_list, label=name)
            axes[i].legend()  # 添加图例

    plt.draw()
    return fig  # 返回图形对象


def plot_mesh(mesh, solution, s1, s2):
    """
    Plot a heatmap comparison between the true solution and PINN solution of the PDE.

    Parameters:
        mesh: The mesh object representing the equation's domain.
        solution: The true solution function of the partial differential equation.
        s1: Neural network for the real part of the solution.
        s2: Neural network for the imaginary part of the solution.

    Returns:
        matplotlib.figure.Figure: A figure containing 4 subplots showing:
            - Top left: Real part of true solution
            - Top right: Imaginary part of true solution
            - Bottom left: Real part of PINN solution
            - Bottom right: Imaginary part of PINN solution
    """

    bc_ = bm.tensor([1 / 3, 1 / 3, 1 / 3], dtype=bm.float64)
    ps = mesh.bc_to_point(bc_)

    u_real = bm.real(solution(ps)).detach().numpy()
    u_imag = bm.imag(solution(ps)).detach().numpy()
    up_real = s1(ps).detach().numpy()
    up_imag = s2(ps).detach().numpy()

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    # 设置子图标题
    axes[0, 0].set_title('Real Part of True Solution')
    axes[0, 1].set_title('Imag Part of True Solution')
    axes[1, 0].set_title("Real Part of Pinn Module's Solution")
    axes[1, 1].set_title("Imag Part of Pinn Module's Solution")

    mesh.add_plot(axes[0, 0], cellcolor=u_real, linewidths=0, aspect=1)
    mesh.add_plot(axes[0, 1], cellcolor=u_imag, linewidths=0, aspect=1)
    mesh.add_plot(axes[1, 0], cellcolor=up_real, linewidths=0, aspect=1)
    mesh.add_plot(axes[1, 1], cellcolor=up_imag, linewidths=0, aspect=1)

    plt.draw()
    return fig  # 返回图形对象
