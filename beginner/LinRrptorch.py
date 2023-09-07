import torch
import torch.utils.data as data

X_train = torch.linspace(-10, 10, 100).view(-1, 1)
y_train = X_train * 2 + torch.normal(torch.zeros(100), std=0.5).view(-1, 1)
X = torch.autograd.Variable(X_train)
y = torch.autograd.Variable(y_train)

torch_dataset = data.TensorDataset(X, y)
num_epochs = 1000
learning_rate = 0.001


#  model = torch.nn.Linear(1, 1)
class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


model = LinearRegression()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
train_loader = data.DataLoader(dataset=torch_dataset, batch_size=3, shuffle=True)
for epoch in range(num_epochs):
    step = 0
    for step, (batch_x, batch_y) in enumerate(train_loader):
        step = step + 1
        out = model(batch_x)
        loss = torch.nn.MSELoss()(out, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

for param in model.named_parameters():
    print(param)  # 打印权值
