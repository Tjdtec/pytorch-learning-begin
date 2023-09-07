import torch
import torch.utils.data as Data

BATCH_SIZE = 3

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)
torch_dataset = Data.TensorDataset(x, y)  # 将x,y组成一个完美的数据集


def x():
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # 打乱数据
        num_workers=2,  # 多线程读取数据, return x,y
    )
    for epoch in range(3):
        i = 0
        for step, (batch_x, batch_y) in enumerate(loader):
            i = i + 1
            print('Epoch:{}|num:{}|batch_x:{}|batch_y:{}'
                  .format(epoch, i, batch_x, batch_y))


if __name__ == '__main__':
    x()
