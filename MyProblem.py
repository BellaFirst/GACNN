# -*- coding: utf-8 -*-
import pdb
import time
import argparse
import numpy as np
import geatpy as ea
from mmcv import Config
from log import Logger
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
import multiprocessing as mp
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool

from Train import GACNN

def parser():
    parse = argparse.ArgumentParser(description='Pytorch Cifar10 Training')
    parse.add_argument('--config','-c',default='./config/config.py',help='config file path')
    args = parse.parse_args()
    return args


"""
主要的参数为：
    # 隐藏层层数: num_layers [5-10]
    神经元个数，num_neurons [32-1024] 6层：前3层：128-1024，后三层：32-512
    batch_size [1,2]*100
    学习率: lr [1,10000] / 10000
"""

class MyGAProblem():
    def __init__(self, cfg, log):
        self.cfg = cfg
        self.log = log
        self.MAXGEN = cfg.PARA.GACNN_params.MAXGEN
        self.Nind = cfg.PARA.GACNN_params.Nind
        self.maxormins = cfg.PARA.GACNN_params.maxormins #-1：最大化 1：最小化
        self.xov_rate = cfg.PARA.GACNN_params.xov_rate #交叉概率
        self.num_hidden_neurons = cfg.PARA.GACNN_params.num_hidden_neurons

        ub_hidden_n = [cfg.PARA.GACNN_params.hidden_neurons[1]] * self.num_hidden_neurons
        lb_hidden_n = [cfg.PARA.GACNN_params.hidden_neurons[0]] * self.num_hidden_neurons
        ub_lr = [cfg.PARA.GACNN_params.lr[1]]
        lb_lr = [cfg.PARA.GACNN_params.lr[0]]
        ub_batch_size = [cfg.PARA.GACNN_params.batch_size[1]]
        lb_batch_size = [cfg.PARA.GACNN_params.batch_size[0]]
        ub = np.hstack((ub_hidden_n, ub_lr, ub_batch_size))
        lb = np.hstack((lb_hidden_n, lb_lr, lb_batch_size))
        self.varTypes = [1] * (self.num_hidden_neurons + 2)
        self.FieldDR = np.vstack((lb, ub, self.varTypes))

        hidden_neurons = np.random.randint(low=cfg.PARA.GACNN_params.hidden_neurons[0],
                                           high=cfg.PARA.GACNN_params.hidden_neurons[1] + 1,
                                           size=[self.Nind, self.num_hidden_neurons], dtype=int)
        lr = np.random.randint(low=cfg.PARA.GACNN_params.lr[0],
                               high=cfg.PARA.GACNN_params.lr[1] + 1,
                               size=[self.Nind, 1], dtype=int)
        batch_size = np.random.randint(low=cfg.PARA.GACNN_params.batch_size[0],
                                       high=cfg.PARA.GACNN_params.batch_size[1],
                                       size=[self.Nind, 1], dtype=int)
        self.chrom = np.hstack((hidden_neurons, lr, batch_size))

        # 记录每一代的数据
        self.obj_trace = np.zeros((self.MAXGEN, 2)) #[MAXGEN, 2] 其中[0]记录当代种群的目标函数均值，[1]记录当代种群最优个体的目标函数值
        self.var_trace = np.zeros((self.MAXGEN, self.num_hidden_neurons+2)) #记录当代种群最优个体的变量值
        self.time = None

        # 记录所有种群中的最优值
        self.best_gen = None
        self.best_Objv = None
        self.best_chrom_i = None

    def get_Objv_i(self, chrom):
        # chrom = np.int32(chrom)
        chrom = chrom.astype(int)
        hidden_neurons = chrom[:, :self.num_hidden_neurons]
        lr = chrom[:, self.num_hidden_neurons] / 10000
        batch_size = chrom[:, -1] * 100

        acc = []
        for i in range(self.Nind):
            temp_acc = GACNN(hidden_neurons[i], lr[i], batch_size[i])
            acc.append(temp_acc)
        Objv = np.array(acc).reshape(-1, 1)
        # Objv = np.random.rand(self.Nind, 1)
        return Objv

    def Evolution(self):
        start_time = time.time()
        Objv = self.get_Objv_i(self.chrom)
        best_ind = np.argmax(Objv * self.maxormins)

        for gen in range(self.MAXGEN):
            self.log.logger.info('==> This is No.%d GEN <==' % (gen))
            FitnV = ea.ranking(Objv * self.maxormins)
            Selch = self.chrom[ea.selecting('rws', FitnV, self.Nind-1), :]
            Selch = ea.recombin('xovsp', Selch, self.xov_rate)
            Selch = ea.mutate('mutswap', 'RI', Selch, self.FieldDR)

            NewChrom = np.vstack((self.chrom[best_ind, :], Selch))
            Objv = self.get_Objv_i(NewChrom)
            best_ind = np.argmax(Objv * self.maxormins)

            self.obj_trace[gen, 0] = np.sum(Objv) / self.Nind #记录当代种群的目标函数均值
            self.obj_trace[gen, 1] = Objv[best_ind]           #记录当代种群最有给他目标函数值
            self.var_trace[gen, :] = NewChrom[best_ind, :]    #记录当代种群最有个体的变量值
            self.log.logger.info('GEN=%d,best_Objv=%.5f,best_chrom_i=%s\n'
                                 %(gen, Objv[best_ind], str(NewChrom[best_ind, :]))) #记录每一代的最大适应度值和个体

        end_time = time.time()
        self.time = end_time - start_time

    def Plot_Save(self):
        self.best_gen = np.argmax(self.obj_trace[:, [1]])
        self.best_Objv = self.obj_trace[self.best_gen, 1]
        self.best_chrom_i = self.var_trace[self.best_gen]

        # pdb.set_trace()
        ea.trcplot(self.obj_trace, [['POP Mean Objv', 'Best Chrom i Objv']])

        with open(self.cfg.PARA.GACNN_params.save_bestdata_txt, 'a') as f:
            f.write('best_Objv=%.5f,best_chrom_i=%s,total_time=%.5f\n'%(self.best_Objv, str(self.best_chrom_i), self.time))

def main():
    args = parser()
    cfg = Config.fromfile(args.config)
    log = Logger(cfg.PARA.utils_paths.log_path + 'GACNN' + '_log.txt', level='info')

    log.logger.info('==> Evolution Begining <==')
    ga = MyGAProblem(cfg, log)
    # print(ga.chrom)

    ga.Evolution()
    ga.Plot_Save()



if __name__=='__main__':
    main()





