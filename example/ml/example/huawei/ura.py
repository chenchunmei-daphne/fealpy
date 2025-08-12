import numpy as np

def ura_weights_2d(M=8, N=8, d=0.5, L_theta=90, L_phi=90):
    """
    返回二维 URA 的波数加权矩阵 W (Q×L)，Q=M*N
    参数
    ----
    M,N : 阵元数
    d   : 阵元间距 / 波长
    L_theta : θ 方向的采样点数
    L_phi   : φ 方向的采样点数
    返回
    ----
    W   : ndarray, shape=(M*N, L_theta*L_phi)
    theta_grid : 弧度制 θ 网格 (L_theta,)
    phi_grid   : 弧度制 φ 网格 (L_phi,)
    """
    # 1. 生成方向网格
    theta = np.linspace(-np.pi/2, np.pi/2, L_theta)   # 仰角
    phi   = np.linspace(-np.pi/2, np.pi/2, L_phi)     # 方位角
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    theta_vec = theta_grid.ravel()
    phi_vec   = phi_grid.ravel()
    L = len(theta_vec)

    # 2. 阵元坐标 (按列优先拉直)
    m_idx = np.arange(M)   # 0..M-1
    n_idx = np.arange(N)   # 0..N-1
    m, n = np.meshgrid(m_idx, n_idx, indexing='ij')
    m = m.ravel()  # (Q,)
    n = n.ravel()  # (Q,)
    Q = M * N

    # 3. 计算波数加权矩阵
    alpha_x = d * np.sin(theta_vec) * np.cos(phi_vec)  # (L,)
    alpha_y = d * np.sin(theta_vec) * np.sin(phi_vec)  # (L,)

    # 利用广播一次性得到所有相位
    phase = 2j * np.pi * (m[:,None] * alpha_x + n[:,None] * alpha_y)
    W = np.exp(phase)  # shape (Q,L)

    return W, theta, phi

# ---------------- demo -----------------
if __name__ == "__main__":
    W, theta, phi = ura_weights_2d(M=8, N=8, d=0.5, L_theta=90, L_phi=90)
    print("W shape =", W.shape)   # 应该是 (64, 8100)