import torch
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

lr = 0.00001
epoch = 1000
X_train = torch.linspace(-10, 10, 100).view(-1, 1)
y_train = X_train * 2.0 + torch.normal(torch.zeros(100), std=0.5).view(-1, 1)
X = torch.autograd.Variable(X_train)  # 变量: pytorch
y = torch.autograd.Variable(y_train)

W = torch.autograd.Variable(torch.rand(1, 1), requires_grad=True)
b = torch.autograd.Variable(torch.rand(1, 1), requires_grad=True)


def MSELoss2(y, out):
    loss = y - out
    loss = loss * loss
    return loss.mean()


def model2(X, W, y):
    return X.mm(W) + b.expand_as(y)


# 进行一次梯度更新
def step():
    W.data.sub_(lr * W.grad.data)
    b.data.sub_(lr * b.grad.data)


def zero_grad():
    W.grad.data.zero_()
    b.grad.data.zero_()


for i in range(epoch):
    out = model2(X, W, y)
    loss = MSELoss2(out, y)
    loss.backward()
    step()
    zero_grad()
print(W.data)  # 打印结果
print(b.data)
