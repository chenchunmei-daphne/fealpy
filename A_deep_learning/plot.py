import matplotlib.pyplot as plt
def plot(*mse):

    n = len(mse)
    fig, axes = plt.subplots(1, n, figsize=(8,6))

    for i in  range(n):
        y1 = range(len(mse[i]))
        axes[i].plot(y1, mse[i], label=f"the {i}-th error graph")
        axes[i].legend()  # 添加图例
    plt.show()

print('----------q')
