import matplotlib.pyplot as plt

def plot_error(*mse):
    '''
       @brief: Plot a line chart.
       @param *mse: Accepts multiple error parameters to plot multiple error curves.
       @return: Returns the line chart.
       '''
    n = len(mse)

    if n == 1:
        fig = plt.figure()
        axes = fig.gca()
        y1 = range(1, 10 * len(mse[0]) + 1, 10)
        axes.plot(y1, mse[0], label=f"the error graph")
        axes.legend()  # 添加图例

    else:
        fig, axes = plt.subplots(1, n, figsize=(8, 7))
        for i in range(n):
            y1 = range(1, 10 * len(mse[i]) + 1, 10)
            axes[i].plot(y1, mse[i], label=f"the {i}-th error graph")
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
    import numpy as np
    import torch
    bc_ = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)
    ps = torch.tensor(mesh.bc_to_point(bc_), dtype=torch.float64)

    u_real = torch.real(solution(ps)).detach().numpy()
    u_imag = torch.imag(solution(ps)).detach().numpy()
    up_real = s1(ps).detach().numpy()
    up_imag = s2(ps).detach().numpy()

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    mesh.add_plot(axes[0, 0], cellcolor=u_real, linewidths=0, aspect=1)
    mesh.add_plot(axes[0, 1], cellcolor=u_imag, linewidths=0, aspect=1)
    mesh.add_plot(axes[1, 0], cellcolor=up_real, linewidths=0, aspect=1)
    mesh.add_plot(axes[1, 1], cellcolor=up_imag, linewidths=0, aspect=1)

    plt.draw()
    return fig  # 返回图形对象
