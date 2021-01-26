#-*- coding:utf-8 _*-
import os
import pdb
import argparse
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from mmcv import Config
import numpy as np
from dataset import Mnist
from models import MyFullConnect
from log import Logger

def parser():
    parse = argparse.ArgumentParser(description='Pytorch Cifar10 Training')
    # parse.add_argument('--local_rank',default=0,type=int,help='node rank for distributedDataParallel')
    parse.add_argument('--config','-c',default='./config/config.py',help='config file path')
    # parse.add_argument('--net','-n',type=str,required=True,help='input which model to use')
    # parse.add_argument('--net','-n',default='MyLenet5')
    # parse.add_argument('--pretrain','-p',action='store_true',help='Location pretrain data')
    # parse.add_argument('--resume','-r',action='store_true',help='resume from checkpoint')
    # parse.add_argument('--epoch','-e',default=None,help='resume from epoch')
    # parse.add_argument('--gpuid','-g',type=int,default=0,help='GPU ID')
    # parse.add_argument('--NumClasses','-nc',type=int,default=)
    args = parse.parse_args()
    return args

def train_valid(net, criterion, optimizer, train_loader, valid_loader, args, log, cfg, epoches=100):
    best_loss = 3
    for epoch in range(epoches):
        net.train()
        train_loss = 0.0
        train_total = 0.0
        for i, data in enumerate(train_loader, 0):
            train_length = len(train_loader)  # length = 54000 / batch_size
            inputs, labels = data #inputs[100,1,28,28] labels[100]没有进行onehot
            inputs = inputs.view(inputs.size(0),-1) #把图片拉平
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)

        net.eval()
        valid_loss = 0.0
        valid_total = 0.0
        with torch.no_grad():  # 强制之后的内容不进行计算图的构建，不使用梯度反传
            for i, data in enumerate(valid_loader, 0):
                valid_length = len(valid_loader)
                inputs, labels = data
                inputs = inputs.view(inputs.size(0), -1)  # 把图片拉平
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                valid_total += labels.size(0)
                # correct += (predicted == labels).sum()
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

        if  (valid_loss / valid_length) < best_loss:
            best_loss = valid_loss / valid_length
            log.logger.info('Save model to checkpoint ')
            checkpoint = {'net': net.state_dict(), 'epoch': epoch}
            if not os.path.exists(cfg.PARA.utils_paths.checkpoint_path + 'GACNN/'): os.mkdir(cfg.PARA.utils_paths.checkpoint_path + 'GACNN/')
            torch.save(checkpoint, cfg.PARA.utils_paths.checkpoint_path + 'GACNN/' + 'best_ckpt.pth')

        log.logger.info('Epoch:%d,Train_Loss:%.5f,Valid_Loss:%.5f '
                        % (epoch + 1, train_loss/train_length, valid_loss/valid_length))

        with open(cfg.PARA.utils_paths.visual_path + 'GACNN' + '_Loss.txt', 'a') as f:
            f.write('Epoch:%d,Train_Loss:%.5f,Valid_Loss:%.5f '
                        % (epoch + 1, train_loss/train_length, valid_loss/valid_length))
            f.write('\n')

def test(net, test_loader):
    with torch.no_grad():
        correct = 0
        total = 0
        net.eval()
        for i, data in enumerate(test_loader, 0):
            images, labels = data #labels是具体的数值
            images, labels = images.cuda(), labels.cuda()
            images = images.view(images.size(0), -1)  # 把图片拉平
            # pdb.set_trace()
            outputs = net(images) #outputs:[100,10]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()#.item()
            acc = correct // total
        return acc


def GACNN(hidden_layers,hidden_neurons,lr,batch_size):
    args = parser()
    cfg = Config.fromfile(args.config)
    log = Logger(cfg.PARA.utils_paths.log_path+ 'GACNN' + '_log.txt',level='info')

    log.logger.info('==> Preparing dataset <==')
    mnist = Mnist(batch_size=batch_size)
    train_loader, valid_loader = mnist.Download_Train_Valid()
    test_loader = mnist.Download_Test()

    log.logger.info('==> Loading model <==')
    net = MyFullConnect(in_dim=cfg.PARA.mnist_params.in_dim,
               hidden_layers=hidden_layers,
               hidden_layer_neurons=hidden_neurons,#[16,32,32,64],
               out_dim=cfg.PARA.mnist_params.out_dim)
    net = net.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr=lr)#, momentum=cfg.PARA.train.momentum

    log.logger.info('==> Waiting Train <==')
    train_valid(net=net, criterion=criterion, optimizer=optimizer,train_loader=train_loader,
                valid_loader=valid_loader, args=args, log=log, cfg=cfg, epoches=cfg.PARA.GACNN_params.epoch)

    log.logger.info("==> Waiting Test <==")
    checkpoint = torch.load(cfg.PARA.utils_paths.checkpoint_path + 'GACNN/' + 'best_ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    test_acc = test(net=net, test_loader=test_loader)

    with open(cfg.PARA.GACNN_params.save_data_txt,'a') as f:
        f.write('hidden_layers=%d,hidden_neurons=%s,Learning_rate=%.03f,batch_size=%d,Acc=%.5f\n'
                % (hidden_layers, str(hidden_neurons), lr, batch_size, test_acc))


if __name__=='__main__':
    hidden_layers = 3
    hidden_neurons = [300,100,100,50]
    lr = 0.01
    batch_size = 100
    GACNN(hidden_layers=hidden_layers,hidden_neurons=hidden_neurons,lr=lr,batch_size=batch_size)














