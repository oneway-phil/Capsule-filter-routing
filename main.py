

import torch
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
#from torchsummary import summary
from capsule_network import CapsuleNetwork
import numpy as np
import matplotlib.pyplot as plt
import math
import time 
import os
import argparse
#
# Settings.
#
learning_rate = 0.001
batch_size = 10
#
# Load CIFAR-10 dataset.
#
import torchvision
from torch.utils.data import Dataset
# Normalization for CIFAR-10 dataset.

"""
train_transform = transforms.Compose(
    [ 
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

test_transform = transforms.Compose([
    
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
])
train_set = torchvision.datasets.SVHN(root='/content/data', split="train",download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True)

test_set = torchvision.datasets.SVHN(root='/content/data', split="test",download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False)
"""
#CIFAR-10

train_transform = transforms.Compose(
    [
    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(), # Convert a PIL Image or numpy.ndarray to tensor.
    # Normalize a tensor image with mean 0.1307 and standard deviation 0.3081
    transforms.Normalize((0.4924044, 0.47831464, 0.44143882), (0.25063434, 0.2492162,  0.26660094)),
    ])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.49053448, 0.47128814, 0.43724576), (0.24659537, 0.24846372, 0.26557055))
])
train_set = torchvision.datasets.CIFAR10(root='/data', train=True,
                                        download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True)

test_set = torchvision.datasets.CIFAR10(root='/data', train=False,
                                       download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#FashionMNIST
"""
train_transform = transforms.Compose([      
                          transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                      ])
test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

train_set = torchvision.datasets.FashionMNIST(root='/content/data', train=True,
                                        download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True)

test_set = torchvision.datasets.FashionMNIST(root='/content/data', train=False,
                                       download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False)
"""
"""
#CIFAR100
train_transform = transforms.Compose(
    [
    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(), # Convert a PIL Image or numpy.ndarray to tensor.
    # Normalize a tensor image with mean 0.1307 and standard deviation 0.3081
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
train_set = torchvision.datasets.CIFAR100(root='/content/data', train=True,
                                        download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True)

test_set = torchvision.datasets.CIFAR100(root='/content/data', train=False,
                                       download=False, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False)
"""
#
# Create capsule network.
#
conv_inputs = 3
#output_unit_size = 16
#num_output_units=10
network = CapsuleNetwork(            image_width=32,
                         image_height=32,
                         image_channels=3,
                         conv_inputs =conv_inputs,
                        )
#print(network)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
network = network.to(device)

#count number
def count_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.numel())
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(network)
print(total_params)

"""
import torch
from ptflops import get_model_complexity_info
model = network
model_name = 'capsule'
flops, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=False)
print("%s |flop =%s |params = %s" % (model_name, flops, params))
"""
from torch.optim.lr_scheduler import ReduceLROnPlateau
#优化器以及学习率衰减
optimizer = optim.Adam(network.parameters(), lr=learning_rate)
#optimizer = optim.SGD(network.parameters(), lr=learning_rate,momentum=0.9)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma = 0.1)
#scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.1, patience=5, verbose=True)


#加载模型 并将起始epoch进行更改
"""
checkpoint = torch.load("D:\code/test1/model31_0.031504,88.290,92.920,93.150,33m39s.pth")
network.load_state_dict(checkpoint['state'])
start_epoch = checkpoint['epoch']
learning_rate =checkpoint['lr']
optimizer = optim.Adam(network.parameters(), lr=learning_rate)
print('===> Load last checkpoint data')
print(learning_rate)
network.to(device)
"""

from torch.optim.lr_scheduler import _LRScheduler

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
# Converts batches of class indices to classes of one-hot vectors.


def to_one_hot(x, length):
    batch_size = x.size(0)
    x_one_hot = torch.zeros(batch_size, length)
    for i in range(batch_size):
        x_one_hot[i, x[i]] = 1.0
    return x_one_hot

def test():
    network.eval()
    test_loss = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    for data, target in test_loader:
        target_indices = target
        target_one_hot = to_one_hot(target_indices, length=10)

        data, target = Variable(data).to(device), Variable(target_one_hot).to(device)

        L1_mid,L2_mid,digit_capsule = network(data)

        test_loss += network.loss(data,L1_mid,L2_mid,digit_capsule,target, size_average=False).data[0]        
        
        v_mag1 = torch.sqrt((L1_mid**2).sum(dim=2, keepdim=True))
        pred1 = v_mag1.data.max(1, keepdim=True)[1].cpu()
        correct1 += pred1.eq(target_indices.view_as(pred1)).sum()

        v_mag2 = torch.sqrt((L2_mid**2).sum(dim=2, keepdim=True))
        pred2 = v_mag2.data.max(1, keepdim=True)[1].cpu()
        correct2 += pred2.eq(target_indices.view_as(pred2)).sum()

        v_mag3 = torch.sqrt((digit_capsule**2).sum(dim=2, keepdim=True))
        pred3 = v_mag3.data.max(1, keepdim=True)[1].cpu()
        correct3 += pred3.eq(target_indices.view_as(pred3)).sum()
        

        print('Test Stage:   [{}/{} ] \tLoss: {:.6f}'.format(
            len(data),
            len(test_loader.dataset),
            test_loss))

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: (L1:{:.2f}%,L2:{:.2f}%,L3:{:.2f}%)\n'.format(
        test_loss,
        100. * correct1 / len(test_loader.dataset),
        100. * correct2 / len(test_loader.dataset),
        100. * correct3 / len(test_loader.dataset),))
    accuracy1 = 100. * correct1 / len(test_loader.dataset)
    accuracy2 = 100. * correct2 / len(test_loader.dataset)
    accuracy3 = 100. * correct3 / len(test_loader.dataset)
    return test_loss,accuracy1,accuracy2,accuracy3

def train(epoch):

    last_loss = None
    log_interval = 1
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target_one_hot = to_one_hot(target, length=10)

        data, target = Variable(data).to(device), Variable(target_one_hot).to(device)

        optimizer.zero_grad()

        L1_mid,L2_mid,digit_capsule = network(data)

        loss  = network.loss(data,L1_mid,L2_mid,digit_capsule, target)
        loss.backward()
        last_loss = loss.data

        optimizer.step()

        """
        for name, parms in network.named_parameters():
            print('-->name:', name, '-->grad_requirs:', parms.requires_grad,
                  ' -->grad_value:', parms.grad)
        """#打印梯度情况
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data))

    return last_loss

if __name__=='__main__':
    num_epochs = 50
    scheduler_steplr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler_steplr)
    
    for epoch in range(1, num_epochs + 1):
        start = time.time()
        scheduler_warmup.step(epoch)
        print(epoch, optimizer.param_groups[0]['lr'])
        last_loss = train(epoch)
        test_loss,accuracy1,accuracy2,accuracy3 = test()
        end = time.time()
        training_time = end-start
        #scheduler.step(test_loss)
        # 保存模型
        print('===> Saving models...')
        state = {
            'state': network.state_dict(),
            'epoch': epoch,   # 将epoch一并保存
            'accuracy': accuracy3,
            'test_loss': test_loss,   
            'lr':optimizer.param_groups[0]['lr'], 
            #'optimizer':optimizer.state_dict(),          
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, 'D:\code/test1/main{}_{:.6f},{:.3f},{:.3f},{:.3f},{:.0f}m{:.0f}s.pth'.format(epoch,test_loss,accuracy1,accuracy2,accuracy3,training_time // 60,training_time % 60))




