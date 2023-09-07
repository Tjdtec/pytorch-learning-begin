import torch

x = torch.autograd.Variable(torch.tensor(1.0), requires_grad=True)
print(x)
y = x * x + 1
y.backward()
print(x.grad)
x2 = torch.autograd.Variable(torch.tensor(0.0), requires_grad=True)
y2 = torch.sin(x2)
y2.backward()
print(x2.grad)
x3 = torch.autograd.Variable(torch.tensor(1.0), requires_grad=True)
y3 = torch.autograd.Variable(torch.tensor(2.0), requires_grad=True)
z3 = x3 * x3 + y3 * y3 + 1
z3.backward()
print(x3.grad)
print(y3.grad)
