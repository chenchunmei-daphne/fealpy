import matplotlib.pyplot as plt
from fealpy.backend import backend_manager as bm
bm.set_backend('pytorch')
def plot_error(**errors):
    '''
    @brief: Plot a line chart.
    @param **errors: Accepts multiple error parameters with names to plot multiple error curves.
    @return: Returns the line chart.
    '''
    # 获取调用该函数时的变量名
    var_names = list(errors.keys())

    n = len(errors)

    if n == 1:
        fig = plt.figure()
        axes = fig.gca()
        # 假设每个错误列表长度相同，这里使用第一个列表的长度
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
    '''
      @brief: Plot a heatmap of the numerical solution of the equation, where different color intensities represent different data values.
      @param mesh: The mesh of the equation's domain.
      @param solution: The true solution of the partial differential equation.
      @param s1, s2: Respectively correspond to the neural networks for the real and imaginary parts of the solution.
      @return: Returns the heatmap.
      '''

    bc_ = bm.tensor([1 / 3, 1 / 3, 1 / 3], dtype=bm.float64)
    # ps = bm.tensor(mesh.bc_to_point(bc_), dtype=bm.float64)
    ps = mesh.bc_to_point(bc_)

    u_real = bm.real(solution(ps)).detach().numpy()
    u_imag = bm.imag(solution(ps)).detach().numpy()
    up_real = s1(ps).detach().numpy()
    up_imag = s2(ps).detach().numpy()

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    # 设置子图标题
    axes[0, 0].set_title('True Solution Real Part')
    axes[0, 1].set_title('True Solution Imag Part')
    axes[1, 0].set_title('Pinn Module Solution Real Part')
    axes[1, 1].set_title('Pinn Module Solution Imaginary Part')

    mesh.add_plot(axes[0, 0], cellcolor=u_real, linewidths=0, aspect=1)
    mesh.add_plot(axes[0, 1], cellcolor=u_imag, linewidths=0, aspect=1)
    mesh.add_plot(axes[1, 0], cellcolor=up_real, linewidths=0, aspect=1)
    mesh.add_plot(axes[1, 1], cellcolor=up_imag, linewidths=0, aspect=1)

    plt.draw()
    return fig  # 返回图形对象
