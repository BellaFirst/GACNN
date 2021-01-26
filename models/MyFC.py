import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import pdb
from _collections import OrderedDict

class MyFullConnect(nn.Module):
    def __init__(self, in_dim, hidden_layers,hidden_layer_neurons,out_dim): #num_hidden_layers_neurons = hidden_layer + 1
        super(MyFullConnect, self).__init__()
        self.hidden_layers = hidden_layers

        self.layer = []
        self.layer.append(nn.Linear(in_dim, hidden_layer_neurons[0]))
        self.layer.append(nn.ReLU(True))
        for i in range(self.hidden_layers):
            # pdb.set_trace()
            self.layer.append(nn.Linear(hidden_layer_neurons[i], hidden_layer_neurons[i+1]))
            # self.layer.append(nn.ReLU(True))

        self.layer.append(nn.Linear(hidden_layer_neurons[-1], out_dim))
        self.layer.append(nn.Softmax(dim=1))
        self.model = nn.Sequential(*self.layer) #加*表示：任意形参，不限制数量

    def forward(self, x):
        out = self.model(x)
        pdb.set_trace()
        return out

class MyFullConnect2(nn.Module):
    def __init__(self): #num_hidden_layers_neurons = hidden_layer + 1
        super(MyFullConnect2, self).__init__()

        self.layer = []
        self.layer.append(nn.Linear(28*28, 512))
        self.layer.append(nn.ReLU(True))
        self.layer.append(nn.Linear(512, 256))
        self.layer.append(nn.Linear(256, 128))
        self.layer.append(nn.Linear(128, 10))
        self.layer.append(nn.Softmax(dim=1))
        self.model = nn.Sequential(*self.layer) #加*表示：任意形参，不限制数量

    def forward(self, x):
        out = self.model(x)
        # pdb.set_trace()
        return out


class MyFC2(nn.Module):
    def __init__(self, in_dim, out_dim, neurons):
        super(MyFC2, self).__init__()
        self.l1 = torch.nn.Linear(in_dim, neurons[0])  # Linear中的w，b一定要初始化，若自己不定义初始化，则用默认的初始化方式初始化
        self.l2 = torch.nn.Linear(neurons[0], neurons[1])
        self.l3 = torch.nn.Linear(neurons[1], neurons[2])
        self.l4 = torch.nn.Linear(neurons[2], neurons[3])
        self.l5 = torch.nn.Linear(neurons[3], neurons[4])
        self.l6 = torch.nn.Linear(neurons[4], neurons[5])
        self.l7 = torch.nn.Linear(neurons[5], out_dim)
    # 激活函数既可以使用nn，又可以调用nn.functional
    def forward(self, x):
        # pdb.set_trace()
        out = F.relu(self.l1(x))  # 激活函数，直接调用torch.nn.functional中集成好的Relu
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.l5(out)
        out = self.l6(out)
        out = self.l7(out)
        out = F.softmax(out,dim=1)
        return out

if __name__=='__main__':
    data = torch.randn([5, 28 * 28])

    # neurons = [300,100,100,64]
    # net = MyFullConnect(28*28, 3, neurons, 10)
    # print(net)

    # data = torch.rand([10, 1, 28, 28])
    # out = net(data)
    # print(out)

    # model = MyFC2(28 * 28, 300, 100, 10)
    # out = model(data)
    # print(out)
    # pdb.set_trace()

    net = MyFullConnect2()
    print(net)
    out = net(data)
    print(out)
    # pdb.set_trace()

    net2 = MyFC2()
    print(net2)
    out2 = net2(data)
    print(out2)
