import  numpy as np
W = np.array([42, 51, 49, 55, 47, 58, 43, 59, 48, 41,
              60, 52, 61, 49, 57, 63, 45, 46, 62, 58])
yi_mean =np.array([6.2, 5.8, 6.7, 4.9, 5.2, 6.9, 4.3, 5.2, 5.7, 6.1,
                   6.3, 6.7, 5.9, 6.1, 6.0, 4.9, 5.3, 6.7, 6.1, 7.0])
M_mean = np.mean(W)
print(M_mean)
n = 20
y_meam = np.dot(yi_mean, W) / (n*M_mean)
print(y_meam)