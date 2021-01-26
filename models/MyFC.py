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
            self.layer.append(nn.Linear(hidden_layer_neurons[i], hidden_layer_neurons[i+1]))
            self.layer.append(nn.ReLU(True))
        self.layer.append(nn.Linear(hidden_layer_neurons[-1], out_dim))
        self.layer.append(nn.Softmax(dim=0))
        self.model = nn.Sequential(*self.layer) #加*表示：任意行参，不限制数量

    def forward(self, x):
        x = self.model(x)
        return x


class MyFC2(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(MyFC2, self).__init__()

        self.l1 = torch.nn.Linear(in_dim, n_hidden_1)  # Linear中的w，b一定要初始化，若自己不定义初始化，则用默认的初始化方式初始化
        self.l2 = torch.nn.Linear(n_hidden_1, n_hidden_2)
        self.l3 = torch.nn.Linear(n_hidden_2, out_dim)

    # 激活函数既可以使用nn，又可以调用nn.functional
    def forward(self, x):
        out = F.relu(self.l1(x))  # 激活函数，直接调用torch.nn.functional中集成好的Relu
        out = self.l2(out)
        out = self.l3(out)
        return out

if __name__=='__main__':
    neurons = [16,32,32,64]
    net = MyFullConnect([100,1,28,28], 3, neurons, 10)
    print(net)

    # data = np.ones(shape=(784))
    data = torch.rand([10, 1, 28, 28])
    out = net(data)
    print(out)

    #model = MyFC2(28 * 28, 300, 100, 10)
    #print(model)
    # pdb.set_trace()

