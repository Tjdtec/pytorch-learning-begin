import numpy as np
#
#
# xMat = np.mat([[1, 2, 3]])
#
#
# print(np.shape(xMat)[1])  # 返回列数
# print(np.shape(xMat)[0])  # 返回行数
# yMean = np.mean(xMat, 0)  # 返回每一列的平均值，以向量的形式
# print(yMean)
# print(xMat[0, :])
# returnMat = np.zeros((5, 3))
# for i in range(5):
#     returnMat[i, :] = xMat
#     print(returnMat)
#
# # 切分数据集为两个子集

import torch
import numpy as np

x = torch.arange(4.0, requires_grad=True)
y = x * x
print(y)

y.sum().backward()
x.grad

print(y.sum())
print(x.grad)

a = np.ones((3, 4))
list = []
list.append(a)
print(list[0])
