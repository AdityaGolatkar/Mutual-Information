import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset, DataLoader
import torchvision

from IPython import embed

def get_data(dataset,batch_size,):
    if dataset == 'mnist':        
        train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data/mnist', train=True, download=True,
                               transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])),batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data/mnist', train=False, download=True,
                               transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])),batch_size=batch_size, shuffle=False)
        inp_size = 784
        return train_loader,test_loader,inp_size
    
    elif dataset == 'fmnist':        
        train_loader = torch.utils.data.DataLoader(datasets.FashionMNIST('../data/fmnist', train=True, download=True,
                               transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])),batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(datasets.FashionMNIST('../data/fmnist', train=False, download=True,
                               transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])),batch_size=batch_size, shuffle=False)
        inp_size = 784
        return train_loader,test_loader,inp_size
    else:print('Please enter correct dataset')