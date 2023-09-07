import numpy as np
import torch

print(np.sqrt(1 * 1 + 2 * 2 + 3 * 3 + 4 * 4))
b10 = torch.tensor([1, 2, 3, 4], dtype=torch.float64)
b11 = torch.norm(b10, 2)
b12 = torch.tensor([[1, 2],
                    [3, 4]], dtype=torch.float64)
b13 = torch.norm(b12, 2)
print(b11)
print(b13)

b30 = torch.tensor([1, 2, 3, 4], dtype=torch.float64)
b31 = torch.norm(b30, np.inf)
print(b31)

b40 = torch.tensor([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]], dtype=torch.float64)
b41 = torch.trace(b40)
print(b41)
b42 = torch.t(b40)
b43 = torch.trace(b42)
print(b43)

# Frobenius范数
a1 = torch.tensor([[1.0, 2, 3, 4],
                   [5, 6, 7, 8]])
a2 = torch.t(a1)
a3 = torch.mm(a1, a2)
a4 = torch.sqrt(torch.trace(a3))
n1 = torch.norm(a1, 2)
print(a4 == n1)

# 特征值分解
a10 = torch.tensor([[3, 2, 1],
                    [0, -1, -2],
                    [0, 0, 3]], dtype=torch.float64)
a11_complex, a12_complex = torch.linalg.eig(a10)
a11_complex = torch.diag(a11_complex)
print("特征值:", a11_complex)
print("特征向量", a12_complex)

# 奇异值分解
b1 = torch.tensor([[1, 2, 3],
                   [4, 5, 6]], dtype=torch.float64)
U, D, V = torch.svd(b1, some=False)
D1 = torch.diag(D)
D2 = torch.zeros(2, 3)
D2[:, 0] = D1[:, 0]
D2[:, 1] = D1[:, 1]
D3 = torch.tensor(D2, dtype=torch.float64)
print(D3)
A = U @ D3 @ V.t()
print(A)

# 广义逆矩阵
a1 = torch.DoubleTensor([[1, 2, 3],
                         [4, 5, 6]])
a2 = torch.pinverse(a1)
print(a2)
