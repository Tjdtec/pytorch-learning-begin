import torch
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 获取最大值的位置方法
a = torch.randn(1, 10)
print(a)
b = torch.max(a)
print(b)
c = torch.max(a, 1)
print(c)
_, d = torch.max(a, 1)
print(d)

# 观察sigmoid分类图像
x = np.linspace(-100, 100, 1000)
y = torch.sigmoid(torch.tensor(x)).numpy()
plt.plot(x, y)
plt.title("Sigmoid(x)")
plt.xlabel("x")
plt.ylabel("sigmoid(x)")
plt.grid("True")
plt.show()

# # 观察tanh分类图像
# x = np.linspace(-100, 100, 1000)
# y = torch.tanh(torch.tensor(x, dtype=torch.float32)).data.numpy()
# plt.plot(x, y)
# plt.title("tanh(x)")
# plt.xlabel("x")
# plt.ylabel("tanh(x)")
# plt.grid("True")
# plt.show()
