import numpy as np
import numpy as np


## 2.4
# N, n = 10000, 200
# f = n / N
# p = 0.35
#
# # v_p = (1-f) / (n-1) * p * (1-p)
# v_p = 0.00112
# # a = np.sqrt(v_p) * 1.96
# a = 0.06556
# print(f, v_p, a, [p-a, p+a], sep='\n')


## 2.5
# d = [200,150,170,150,160,130,140,100,110,140,150,160,180,130,150,100,180,100,170,120]
#
# m= np.mean(d)
# print(m)
# s=0
# for i in d :
#     s+=i**2
# print(s/19 - m**2*20 / 19)  #
#
# # 计算总体方差
# variance_population = np.var(d)
# print("总体方差:", variance_population)
#
# # 计算样本方差
# variance_sample = np.var(d, ddof=1)
# print("样本方差:", variance_sample)
# #
# n = 20
# f = 20 / 200
# print(variance_sample * (1-f) / n, np.sqrt(37.172),  np.sqrt(variance_sample * (1-f) / n) * 1.96)
# print(m-11.95,m+11.95)
# a = np.sqrt(37.172) * 1.96
# print(m-a, 37m+a)


## 2.6
# N, n = 350, 50
# f = n / N
# m = 1120
# s = 25600
# Y = N * m
# print(("总体总值：", Y))
# vs = N**2 * (1-f) / n * s
# # vs = 53760000.00
# print('V(Y)=', vs,  np.sqrt(53760000))
# a = np.sqrt(vs) * 1.96  # 14370.957
# a = 14371.0
# print(7332.12*1.96 , np.sqrt(vs) * 1.96-m )
# print([Y-a, Y+a])
#
#
# ## 2.9
# import pandas as pd
#
# # # 读取Excel文件
# data = pd.read_excel("D:\\Da_课本\\抽样调查数据\\作业2-9.xlsx")
#
# # 选择最后两列数据
# y = data.iloc[:, 1]
# x= data.iloc[:, 2]
#
# # 计算这两列的协方差
# cov_matrix = np.cov(y, x, bias=False)  # 使用bias=True得到样本协方差
#
# print("协方差矩阵:\n", cov_matrix)
# print('cov(x):', np.var(x, ddof=1))
#
# sy, syx, sx = cov_matrix[0,0], cov_matrix[0,1], cov_matrix[1,1]
# y_m, x_m = 144.5, 1580
# N, n = 200, 20
# f = n / N
# r = y_m / x_m
# v = (1-f) / n * (sy-2*r * syx + r**2*sx)
# x = 1600
# y_r = x * r
# print(r, y_r)
# print([y_r-1.96*np.sqrt(v), y_r+1.96*np.sqrt(v)])
# 0.09145569620253165 146.32911392405063
# [143.58396482807257, 149.0742630200287]

sy, syx, sx = 826.05, 8831.58, 99578.95
y_m, x_m = 144.5, 1580
N, n = 200, 20
f = n / N
r = y_m / x_m
v = (1-f) / n * (sy-2*r * syx + r**2*sx)
x = 1600
y_r = x * r
print(r, y_r)
print([146.33-1.96*np.sqrt(v), 146.33+1.96*np.sqrt(v)])

# 0.09145569620253165 146.32911392405063
# [143.58396482807257, 149.0742630200287]

print(syx/np.sqrt(sy*sx))
p = 0.974
print((1-f)/n * (sy - 2*r*p*np.sqrt(sy*sx) + r**2*sx))