# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
import multiprocessing as mp
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool

from Train import GACNN

"""
该案例展示了如何利用进化算法+多进程/多线程来优化SVM中的两个参数：C和Gamma。
在执行本案例前，需要确保正确安装sklearn，以保证SVM部分的代码能够正常执行。
本函数需要用到一个外部数据集，存放在同目录下的iris.data中，
并且把iris.data按3:2划分为训练集数据iris_train.data和测试集数据iris_test.data。
有关该数据集的详细描述详见http://archive.ics.uci.edu/ml/datasets/Iris
在执行脚本main.py中设置PoolType字符串来控制采用的是多进程还是多线程。
注意：使用多进程时，程序必须以“if __name__ == '__main__':”作为入口，
      这个是multiprocessing的多进程模块的硬性要求。
"""
"""
主要的参数为：
    # 隐藏层层数: num_layers [5-10]
    神经元个数，num_neurons [32-1024] 6层：前3层：128-1024，后三层：32-512
    batch_size [1,2]*100
    学习率: lr [1,10000] / 10000
"""

class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, PoolType, cfg):  # PoolType是取值为'Process'或'Thread'的字符串
        self.cfg = cfg
        name = 'MyProblem'
        M = 1 #目标维度
        maxormins = [1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标） 输出为网络训练的loss，最小化
        Dim = 4 #变量的个数
        varTypes = [1, 1, 1, 1, 1, 1, 1, 1] #初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [128,128,128,32,32,32,1,1]
        ub = [1024,1024,1024,512,512,2,1000]
        lbin = [1,1,1,1,1,1,1,1] # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1,1,1,1,1,1,1,1] # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)


    def aimFunc(self, pop):  # 目标函数
        num_neurons = []
        for i in range(self.cfg.PARA.GACNN_params.num_layers):
            num_neurons.append(pop.Phen[:,i])
        batch_size = pop.Phen[:,[6]] * 100
        lr = pop.Phen[:,[7]] / 10000
        pop.ObjV = np.array(GACNN(num_neurons,lr,batch_size))

        Vars = pop.Phen  # 得到决策变量矩阵
        args = list(
            zip(list(range(pop.sizes)), [Vars] * pop.sizes, [self.data] * pop.sizes, [self.dataTarget] * pop.sizes))
        if self.PoolType == 'Thread':
            pop.ObjV = np.array(list(self.pool.map(subAimFunc, args)))
        elif self.PoolType == 'Process':
            result = self.pool.map_async(subAimFunc, args)
            result.wait()
            pop.ObjV = np.array(result.get())


def subAimFunc(args):
    i = args[0]
    Vars = args[1]
    data = args[2]
    dataTarget = args[3]
    C = Vars[i, 0]
    G = Vars[i, 1]
    svc = svm.SVC(C=C, kernel='rbf', gamma=G).fit(data, dataTarget)  # 创建分类器对象并用训练集的数据拟合分类器模型
    scores = cross_val_score(svc, data, dataTarget, cv=30)  # 计算交叉验证的得分
    ObjV_i = [scores.mean()]  # 把交叉验证的平均得分作为目标函数值
    return ObjV_i



