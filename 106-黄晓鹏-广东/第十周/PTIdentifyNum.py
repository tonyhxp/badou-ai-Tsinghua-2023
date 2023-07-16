import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision

class MnisNetWork(torch.nn.Module):
    def __init__(self):
        super(MnisNetWork, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 180)
        self.fc3 = torch.nn.Linear(180, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


class XModel:
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimist = self.create_optimizer(optimist)

    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }
        return support_cost[cost]

    def create_optimizer(self, optimist, **rests):
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }

        return support_optim[optimist]

    def train(self, train_loader, epoches=3):
        for e in range(epoches):
            running_loss =0.0
            for i ,data in enumerate(train_loader,0):
                inputs,labels = data

                self.optimist.zero_grad()
                outputs = self.net(inputs)

                loss = self.cost(outputs,labels)
                loss.backward()
                self.optimist.step()

                running_loss +=loss.item()
                if i % 100==0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (e + 1, (i + 1) * 1. / len(train_loader), running_loss / 100))
                    running_loss = 0.0
        print('Finished Training*******************************')

    def evaluate(self, test_loader):
        print('Evaluating*****************************')
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = self.net(images)
                predicted = torch.argmax(outputs,dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


def mnist_load_data():
    #transform的数据转换对象
    # transforms.Compose函数将多个数据转换操作组合在一起
    transform = transforms.Compose(
        [transforms.ToTensor(),# transforms.ToTensor()将图像转换为张量
         transforms.Normalize([0,], [1,])]) # transforms.Normalize()对图像进行标准化处理。

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    # 使用torch.utils.data.DataLoader创建了一个训练数据加载器trainloader，它将训练集对象trainset作为输入，batch_size = 32
    # 表示每个批次包含32个样本，shuffle = True表示在每个epoch中对数据进行随机洗牌，num_workers = 2
    # 表示使用2个进程加载数据。
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,shuffle=True, num_workers=2)
    return trainloader, testloader


if __name__ == '__main__':
    net = MnisNetWork()
    model = XModel(net, 'CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)