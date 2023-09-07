import torch

# Generate Vector
c01 = torch.linspace(1.0, 10.0, 4)
print(c01)

b5 = torch.arange(1, 5)
print(b5)

# Operate Vector
a1 = torch.tensor([1, 3])
a2 = torch.tensor([2, 4])
print(a1 + a2)

v1 = torch.tensor([1, 2])
v2 = torch.tensor([3, 4])
v3 = torch.cat([v1, v2])
v4 = torch.cat([v1, v1, v1])
print(v3)
print(v4)

b2 = torch.tensor([2, 4, 8])
b3 = torch.div(b2, 2)
print(b3)
b3 = b3 + 1
print(b3)

b10 = torch.tensor([1, 3])
b11 = torch.tensor([2, 7])
b12 = b10 * b11
print(b12)

# torch.log space默认以10的次方作为起始和结束值，最后一个参数是步长值
a6 = torch.logspace(0, 2, 4)
print(a6)

# 生成矩阵
a1 = torch.zeros(2, 2)
print(a1)
a2 = torch.zeros(2, 2, 2, 2, 2)
'''
前两个指行数和列数，之后的参数表示矩阵的个数 
按照乘法定则，如本题出去前面两个参数，后面三个参数均为2
则生成的矩阵个数为2*2*2=8个
'''
print(a2)

a3 = torch.ones(4, 4, dtype=torch.float64)
print(a3)
a4 = torch.ones_like(a3)
print(a4)

a5 = torch.eye(5)
print(a5)

b20 = torch.zeros(3, 10)
print(b20)
b20 = torch.t(b20)
print(b20)

# 矩阵的运算
b30 = torch.tensor([[1, 2],
                    [3, 4]])
b31 = torch.tensor([[10, 20],
                    [1, 1]])

b32 = torch.add(b30, b31)
print(b32)

b33 = b30 * b31
print(b33)  # Hadamard积

b34 = torch.mm(b30, b31)
# b34 = b30 @ b31
# b34 = torch.matmul(b30,b31)
print(b34)  # 矩阵的点积

b40 = torch.tensor([[1, 2],
                    [3, 4]], dtype=torch.float32)
b41 = torch.det(b40)
print(b41)
b42 = torch.inverse(b40)
print(b42)
b43 = torch.mm(b40, b42)
print(b43)  # 求逆矩阵

# empty和empty_like,full和full_like
a8 = torch.empty(3, 3)
print(a8)  # 随机生成一个3*3的矩阵，值随机分布
a9 = torch.empty_like(a8, dtype=torch.int32)
print(a9)
a10 = torch.full((3, 3), 1.01)
print(a10)
a11 = torch.full([4, 4], 2.01)
print(a11)

# 改变张量的形状
b1 = torch.arange(0,100)
print(b1)
b2 = b1.view(10, -1)  # -1表示这个位置的值由系统计算
print(b2)

#  矩阵拼接
v5 = torch.zeros(4,4)
v6 = torch.ones(4,4)
v7 = torch.cat([v5,v6],0)
#  v7 = torch.cat([v5, v6], 1)
print(v7)  # 0表示上下拼接，如果参数为1则为左右拼接

# 立方体拼接
v31 = torch.zeros([4, 4, 4])  # 构造一个三维的全0矩阵
v32 = torch.ones([4, 4, 4])
v33 = torch.cat([v31, v32], 0)  # 从0轴开始拼接
print(v33)
v34 = torch.cat([v31, v32], 1)  # 从1轴开始拼接
v35 = torch.cat([v31, v32], 2)  # 从2轴开始拼接
print(v34)
print(v35)
# 进入第四维以后的拼接同上，可以根据中括号的位置进行判断

# 读取其中某一维的值
a12 = torch.tensor([[3., 0.],
                   [-1., 0.],
                   [3., 0.]],dtype=torch.float64)
a13 = a12[:, 0]
print(a13)
v2 = torch.arange(0,4)
v2 = v2.view([2, 2])
print(v2)
print(v2[0])
print(v2[1])
print(v2[:, 0])
print(v2[:, 1])
v1 = torch.arange(0, 32)
v2 = torch.numel(v1)
print(v2)  # 元素个数
v3 = v1.view(2, 2, 2, 2, 2)
v4 = torch.ones_like(v3)
v5 = torch.cat([v3, v4], 2)
print(v5)
print(v5[1])  # 在第一维度切开取后半区
print(v5[:, 1, :])  # 在第二维度切开，取后半区
print(v5[:, :, 1, :])  # 在第三维度切开，取后半区

a = torch.tensor([[[1, 2, 3, 4, 5],
                   [6, 7, 8, 9, 0]],
                  [[10, 11, 12, 13, 14],
                   [15, 16, 17, 18, 19]]])
list_a = [a]
print(a.shape)
print(list_a[0][:, -1, :].shape)

