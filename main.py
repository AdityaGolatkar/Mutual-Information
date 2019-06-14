#!/usr/bin/env python3

import argparse
import os
import shutil
import time
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

import copy
from utils import *
from logger import Logger
from models import *
from IPython import embed
from datasets import *

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.fftpack as fftpack

parser = argparse.ArgumentParser(description='Mutual Information experiments')
parser.add_argument('--batch-size', default=128, type=int,metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('-b', '--beta', default=1.0, type=float,help='value of beta (default: 1.0)')
parser.add_argument('--dataset', default='mnist', type=str,help='dataset name')
parser.add_argument('--data-dim', default=20, type=int,help='data dimension')
parser.add_argument('--decay', default=0.97, type=float,help='Learning rate exponential decay')
parser.add_argument('--dropout', action='store_true',help='use dropout')
parser.add_argument('--estimator', default='mine', type=str,help='mi estimator')
parser.add_argument('--log-name', default=None, type=str,help='index for the log file')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
parser.add_argument('--mi-estimate', action='store_true',help='mi estimate')
parser.add_argument('--name', default='model', type=str,help='name of the run')
parser.add_argument('--m-hidden-sz', default=512, type=int,help='number of hidden units')
parser.add_argument('-o', '--optimizer', default='sgd', type=str,help='Optimizer to use')
parser.add_argument('--resume', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--rho', default=0.8, type=float,help='rho')
parser.add_argument('--save', dest='save', action='store_true',help='save the model every x epochs')
parser.add_argument('--sample-exp', action='store_true',help='sample experiment')
parser.add_argument('--save-final', action='store_true', help='save last version of the model')
parser.add_argument('--save-every', default=5, type=int,help='interval for saving')
parser.add_argument('--schedule', nargs='+', default=[200], type=int,help='number of total epochs to run')
parser.add_argument('--sched_type', default='exp', type=str,help='type of scheduling')
parser.add_argument('--sfe', action='store_true',help='Save first epoch')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',help='manual epoch number (useful on restarts)')
parser.add_argument('--s-input-sz', default=40, type=int,help='number of input units in statistics network')
parser.add_argument('--s-hidden-sz', default=512, type=int,help='number of hidden units in statistics network')
parser.add_argument('--weight-decay', '--wd', default=0.0, type=float,metavar='W', help='weight decay (default: 0.)')
args = parser.parse_args()

# def sample_batch(data, batch_size=100, sample_mode='joint'):
#     if sample_mode == 'joint':
#         index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
#         batch = data[index]
#     else:
#         joint_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
#         marginal_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
#         batch = np.concatenate([data[joint_index][:,0].reshape(-1,1),data[marginal_index][:,1].reshape(-1,1)],axis=1)
#     return batch

# def get_corr_data():
#     y = np.random.multivariate_normal(mean=[0,0],cov=[[1,args.rho],[args.rho,1]],size = 300)
#     return y

def sample_batch(data, batch_size=100, sample_mode='joint'):
    if sample_mode == 'joint':
        index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        batch = data[index]
    else:
        joint_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        marginal_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        batch = np.concatenate([data[joint_index][:,:args.data_dim].reshape(-1,args.data_dim),data[marginal_index][:,args.data_dim:].reshape(-1,args.data_dim)],axis=1)
    return batch

def get_corr_data():
    y = np.random.multivariate_normal(mean=[0,0],cov=[[1,args.rho],[args.rho,1]],size = (300,args.data_dim))
    y = y.transpose(0,2,1)
    y = y.reshape(-1,np.prod(y.shape[1:]))
    return y

def shuffle_features(X):
    b = torch.rand(X.size()).cuda()
    idx = b.sort(0)[1]
    adx = torch.arange(0, X.size(1)).long()
    return X[idx, adx[None, :]]

def mi_estimation(stat_net,features,shuff_feat,estimator):
    
    if estimator == 'mine':
        t = stat_net(features)
        et = torch.exp(stat_net(shuff_feat))
        mi_lb = torch.mean(t) - torch.log(torch.mean(et))
        return mi_lb, t, et
    
    elif estimator == 'mine-f':
        t = stat_net(features)
        et = torch.exp(stat_net(shuff_feat)-1)
        mi_lb = torch.mean(t) - torch.mean(et)
        return mi_lb, t, et
    
    elif estimator == 'jsd':
        t1 = stat_net(features)
        t2 = stat_net(shuff_feat)
        mi_lb = torch.mean(-1*F.softplus(-t1)) - torch.mean(F.softplus(t2))
        return mi_lb, t1, t2   
    
def get_MI(train_loader, model, stat_net, optimizer, epoch, estimator, ma_et, train=True, label='Epoch'):
    losses = AverageMeter()
    mi = AverageMeter()
    
    model.eval()
                
    ma_rate = 0.01
    for i, (input, target) in enumerate(train_loader):

        if not args.sample_exp:
            input = input.reshape(input.shape[0],-1).cuda()
            target = target.cuda()

            output, features = model(input)
            shuff_feat = shuffle_features(features)

            features = features.cuda()
            shuff_feat = shuff_feat.cuda()
        
        else:
            data = get_corr_data()
            features,shuff_feat = sample_batch(data,batch_size=args.batch_size),\
        sample_batch(data,batch_size=args.batch_size,sample_mode='marginal')
            features = torch.from_numpy(features).cuda().float()
            shuff_feat = torch.from_numpy(shuff_feat).cuda().float()
                
        mi_lb, t, et = mi_estimation(stat_net,features,shuff_feat,estimator)
        
        loss = -mi_lb
        if estimator == 'mine':
            if ma_et == -1:ma_et=torch.mean(et)
            else:ma_et = (1-ma_rate)*ma_et + ma_rate*torch.mean(et)
            loss = -(torch.mean(t) - (1/ma_et.mean()).detach()*torch.mean(et)) 
        
        losses.update(loss.item(), input.size(0))
        mi.update(mi_lb.item(), input.size(0))
        
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    print(f"{label}: [{epoch}] MI {mi.avg:.6f} Loss {losses.avg:.4f}")
    logger.append('mi', epoch=epoch, mi=mi.avg, losses=losses.avg)
    
    return ma_et

def train(train_loader, model, criterion, optimizer, epoch, train=True, label='Epoch'):
    losses = AverageMeter()
    errors = AverageMeter()

    if train:
        model.train()
    else:
        model.eval()
                
    for i, (input, target) in enumerate(train_loader):
        input = input.reshape(input.shape[0],-1).cuda()
        target = target.cuda()

        output, Lz = model(input)
        Lx = criterion(output, target)
        loss = Lx
        err, = get_error(output, target)
        
        losses.update(Lx.item(), input.size(0))
        errors.update(err.item(), input.size(0))
        
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    print(f"{label}: [{epoch}] lr: {optimizer.param_groups[0]['lr']:.4f} wd: {optimizer.param_groups[0]['weight_decay']:.4f} Loss {losses.avg:.3f}  Error: {errors.avg:.2f}")
    
    logger.append('train' if train else 'valid', epoch=epoch, loss=losses.avg, error=errors.avg, lr=optimizer.param_groups[0]['lr'], wd=optimizer.param_groups[0]['weight_decay'])
    

def validate(val_loader, model, criterion, optimizer, epoch, label=''):
    train(val_loader, model, criterion, optimizer, epoch, train=False, label=label)
                    
def adjust_learning_rate(optimizer, epoch, schedule):
          
    lr = args.lr      
    if args.sched_type == 'exp':
        lr = args.lr * args.decay**epoch              
    elif args.sched_type == 'const':
        lr = args.lr
    
    else:print('Please select a valid Scheduler')
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
          
def save_checkpoint(state, step=True):
    if step:
        epoch = state['epoch']
        target_file = logger['checkpoint_step'].format(epoch)
    else:
        target_file = logger['checkpoint']
    print("Saving {}".format(target_file))
    torch.save(state, target_file)
          

if __name__ == '__main__':

    mkdir('logs')
    mkdir('models')

    if args.mi_estimate:args.log_name += '_mi'
    logger = Logger(index=args.log_name)
    logger['args'] = args
    logger['checkpoint'] = os.path.join('models/', logger.index+'.pth')
    logger['checkpoint_step'] = os.path.join('models/', logger.index+'_{}.pth')

    print("[Logging in {}]".format(logger.index))
    
    #Dataset
    train_loader,val_loader,inp_size = get_data(args.dataset,args.batch_size)
    
    #DNN
    model = Network(args.dropout,args.m_hidden_sz,inp_size).cuda()
    stat_net = Statistic_Network(args.s_hidden_sz,args.s_input_sz).cuda()

    #Loss
    criterion = nn.CrossEntropyLoss().cuda()
    
    #Optimizer          
    parameters = model.parameters()
    stat_params = stat_net.parameters()
    if args.optimizer=='sgd':
        optimizer = torch.optim.SGD(parameters, args.lr,momentum=args.momentum, weight_decay=args.weight_decay)
        stat_optimizer = torch.optim.SGD(stat_params, args.lr,momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer=='adam':
        optimizer = torch.optim.Adam(parameters, args.lr, betas=(args.momentum, 0.999),weight_decay=args.weight_decay)
        stat_optimizer = torch.optim.Adam(stat_params, args.lr, betas=(args.momentum, 0.999),weight_decay=args.weight_decay)
    else:
        raise ValueError("Optimizer {} not valid.".format(args.optimizer))

    if args.resume:
        checkpoint_file = args.resume
        if not os.path.isfile(checkpoint_file):
            print("=== waiting for checkpoint to exist ===")
            try:
                while not os.path.isfile(checkpoint_file):
                    time.sleep(1)
            except KeyboardInterrupt:
                print("=== waiting stopped by user ===")
                import sys; sys.exit()
        print("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        if not args.mi_estimate:args.start_epoch = checkpoint['epoch']+1
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpoint_file, checkpoint['epoch']))
          
    if args.sfe:
        state = {'epoch':-1, 'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}
        save_checkpoint(state, step=False)
          
    try:
        if not args.mi_estimate:
            for epoch in range(args.start_epoch, args.schedule[-1]):
                
                if args.optimizer != 'adam':adjust_learning_rate(optimizer, epoch, args.schedule[:-1])
                loss = train(train_loader, model, criterion, optimizer, epoch)
                validate(val_loader, model, criterion, optimizer, epoch)
                
                #Save
                state = {'epoch':epoch, 'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}
                if args.save_final and epoch == (args.schedule[-1]-1):
                    save_checkpoint(state, step=False)
                if args.save and epoch % args.save_every == 0:
                    save_checkpoint(state, step=True)
        else:
            ma_et = -1
            for epoch in range(args.start_epoch, args.schedule[-1]):
                ma_et = get_MI(train_loader, model, stat_net, stat_optimizer, epoch, args.estimator, ma_et, train=True, label='Epoch')
        
        logger['finished'] = True
        
    except KeyboardInterrupt:
        print("Run interrupted")
        logger.append('interrupt', epoch=epoch)
    print("[Logs in {}]".format(logger.index))