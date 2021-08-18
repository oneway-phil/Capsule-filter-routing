

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
from collections import OrderedDict


#用于定义卷积层
class _DenseLayer(nn.Sequential):
    def __init__(self,  num_input_features, growth_rate, bn_size,drop_rate):
        super(_DenseLayer, self).__init__()
        
        self.add_module("conv1", nn.Conv2d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False))
        self.add_module("norm1", nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module("relu1", nn.ReLU())

        self.add_module("conv2", nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False))
        self.add_module("norm2", nn.BatchNorm2d(growth_rate))
        self.add_module("relu2", nn.ReLU())
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)

        return torch.cat([x, new_features], 1)
class _DenseBlock(nn.Sequential):
    def __init__(self,num_layers, num_input_features, bn_size, growth_rate,drop_rate):
        super(_DenseBlock,self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate
                                )
            self.add_module("denselayer%d" % (i + 1,), layer)

class _Transition(nn.Sequential):
    def __init__(self,  num_input_feature, num_output_features,kernel_size,stride):
        super(_Transition, self).__init__()
        
        self.add_module('conv', nn.Conv2d(num_input_feature, num_output_features,kernel_size, stride, bias=False))
        self.add_module('norm', nn.BatchNorm2d(num_output_features))
        self.add_module('relu', nn.ReLU())
        
def squash1(s):

    m =torch.softmax(s,dim=2)
    c = torch.mul(0.5*(1+ m),s)
    return c

def squash2(s):
    # This is equation 1 from the paper.
    mag_sq = torch.sum(s**2, dim=2, keepdim=True)
    mag = torch.sqrt(mag_sq)
    s = (mag_sq / (1 + mag_sq)) * (s / mag)
    return s






