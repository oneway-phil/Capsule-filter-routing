

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
from capsule_conv_layer import _DenseBlock, _Transition
from collections import OrderedDict
import numpy
import torch.nn.functional as func
import math

#用于定义胶囊层
class DenseNet1(nn.Module):
    def __init__(self,num_features,num_layers=8,bn_size = 4,drop_rate=0):
        """
        :param block_config: (list of 3 ints) number of layers in each DenseBlock
        """
        super(DenseNet1, self).__init__()

        growth_rate = 32
        
        self.L1 = nn.Sequential()
        block = _DenseBlock(num_layers, num_features, bn_size, growth_rate,drop_rate)
        self.L1.add_module("denseblock%d" % (1), block)
        num_features += num_layers * growth_rate

        transition = _Transition(num_features, 96, kernel_size=5, stride=2)
        self.L1.add_module("transition%d" % (1), transition)

        self.L1.add_module("relu5", nn.ReLU(inplace=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        L1 =self.L1(x)

        return L1
class DenseNet2(nn.Module):
    def __init__(self,num_layers=8,bn_size = 4,drop_rate=0):
        """
        :param block_config: (list of 3 ints) number of layers in each DenseBlock
        """
        super(DenseNet2, self).__init__()
        growth_rate = 32

        num_features = 96
        self.L2 = nn.Sequential()
        block = _DenseBlock(num_layers, num_features, bn_size, growth_rate,drop_rate)
        self.L2.add_module("denseblock%d" % (1), block)
        num_features += num_layers * growth_rate

        transition = _Transition(num_features, 120, kernel_size=5, stride=2)
        self.L2.add_module("transition%d" % (1), transition)
        self.L2.add_module("relu5", nn.ReLU(inplace=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self,L1):
        L2 =self.L2(L1)
        return L2
class DenseNet3(nn.Module):
    def __init__(self,num_layers=8,bn_size = 4,drop_rate=0):
        """
        :param block_config: (list of 3 ints) number of layers in each DenseBlock
        """
        super(DenseNet3, self).__init__()
        growth_rate = 32
        num_features = 120

        self.L3 = nn.Sequential()
        block = _DenseBlock(num_layers, num_features, bn_size, growth_rate,drop_rate)
        self.L3.add_module("denseblock%d" % (1), block)
        num_features += num_layers * growth_rate

        transition = _Transition(num_features, 144, kernel_size=3, stride=1)
        self.L3.add_module("transition%d" % (1), transition)

        self.L3.add_module("relu5", nn.ReLU(inplace=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    def forward(self,L2):
        L3 =self.L3(L2)
        return L3

class capsule(nn.Module):
    def __init__(self,conv_inputs):
      super(capsule, self).__init__()
      
      self.L1 = DenseNet1(conv_inputs,num_layers=8,bn_size = 4,drop_rate=0)
      self.L2 = DenseNet2(num_layers=8,bn_size = 4,drop_rate=0)
      self.L3 = DenseNet3(num_layers=8,bn_size = 4,drop_rate=0)

    def forward(self, x):
      
      L1 = self.L1(x)
      L2 = self.L2(L1)
      b,num1,_,_ = L1.size()
      b,num2,_,_ = L2.size()
      L1 = torch.reshape(L1, (b, num1, -1))
      L1 = torch.reshape(L1,(b,12,8,-1))
      L1 = L1.transpose(1,2)

      L3 = self.L3(L2)

      b,num3,_,_ = L3.size()
      L2 = torch.reshape(L2, (b, num2, -1))
      L2 = torch.reshape(L2,(b,12,10,-1))
      L2 = L2.transpose(1,2)

      L3 = torch.reshape(L3, (b, num3, -1))
      L3 = torch.reshape(L3,(b,12,12,-1))
      L3 = L3.transpose(1,2)

      return L1,L2,L3


class cap_reshape(nn.Module):
  def __init__(self):
    super(cap_reshape,self).__init__()


  def forward(self,L1,L2,L3):

      b = L1.size(0)

      L1 = torch.reshape(L1,(b,8,-1))
      # L2
      L2 = torch.reshape(L2, (b, 10, -1))
      # L3
      L3 = torch.reshape(L3, (b, 12, -1))
      return L1,L2,L3


class Self_Attn(nn.Module):

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()

        self.gamma = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #
        
        self.W1 = nn.Parameter(torch.FloatTensor(1, 1,in_dim))
        self.W2 = nn.Parameter(torch.FloatTensor(1, in_dim,in_dim))
        self.W4 = nn.Parameter(torch.FloatTensor(1, in_dim,in_dim))
        self.W3 = nn.Parameter(torch.FloatTensor(1, in_dim, in_dim))


        nn.init.constant_(self.W1.data, 0)
        nn.init.constant_(self.W2.data, 0)
        nn.init.constant_(self.W4.data, 0)
        nn.init.constant_(self.W3.data, 0)


        #nn.init.kaiming_uniform_(self.W4.data)
        
    def forward(self, x):
        
        batch_size, C, w_h = x.size()
        W1 = torch.cat([self.W1] * batch_size, dim=0)
        W2 = torch.cat([self.W2] * batch_size, dim=0)
        W3 = torch.cat([self.W3] * batch_size, dim=0)
        W4 = torch.cat([self.W4] * batch_size, dim=0)

        proj_query = torch.bmm(W1, x).permute(0, 2, 1)  # B X N*C
        proj_key = torch.bmm(W2, x)
        proj_key = torch.norm(proj_key, p=2, dim=1, keepdim=True)
        energy = torch.bmm(proj_query, proj_key) / numpy.sqrt(C)
        attention = torch.softmax(energy, dim=2)

        proj_value = torch.matmul(W3, x)  # B X C X N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        proj_key2 = torch.bmm(W4, x)
        key2 = torch.softmax(proj_key2, dim=2)
        out = self.gamma * out + x + self.gamma2 * key2

        return out

class Filter_high(nn.Module):   #筛选k个活跃胶囊，每个level选择的k也不一定
    def __init__(self):
        super(Filter_high, self).__init__()
        self.k = nn.Parameter(torch.tensor(0.7, dtype=torch.float))
    def forward(self, x):
        num1 = x.size(2)
        a = torch.norm(x, p=2, dim=1, keepdim=True)
        a2, indices = torch.sort(a, dim=2, descending=True)

        amax = a2[:,:,0]
        amed = torch.median(a2,dim=2)[0]
        amean = torch.mean(a2,dim=2)

        prop = amed / amax
        prop = torch.mean(prop,dim=0)
        if (prop*num1 >num1*0.8):
          prop = 0.8
        if (prop*num1 <0.1*num1):
          prop = 0.1

        
        knum = prop * num1
        capsnum = int(knum)
        index1 = indices[:, :, :capsnum]

        a1, b, c = x.size()
        _, _, c2 = index1.size()
        a = []
        for i in range(0, a1):
            a2 = x[i, :, index1[i]]
            a.append(a2)
        cap = torch.cat(a, dim=0)
        cap = cap.view(a1, b, c2)
        return cap

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CapsuleLayer(nn.Module):
    def __init__(self, in_units, in_channels, num_units, unit_size):
        super(CapsuleLayer, self).__init__()

        self.in_units = in_units
        self.in_channels = in_channels
        self.num_units = num_units

        self.W = nn.Parameter(torch.randn(1, in_channels, num_units, unit_size, in_units))
    @staticmethod
    def squash(s):
        # This is equation 1 from the paper.
        mag_sq = torch.sum(s**2, dim=2, keepdim=True)
        mag = torch.sqrt(mag_sq)
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return s

    def forward(self, x):

        return self.routing(x)

    def routing(self, x):
        
        batch_size = x.size(0)
        # (batch, in_units, features) -> (batch, features, in_units)
        x = x.transpose(1, 2)
        # (batch, features, in_units) -> (batch, features,  in_units, 1)
        x = torch.stack([x] * self.num_units, dim=2).unsqueeze(4)
        W = torch.cat([self.W] * batch_size, dim=0)

        u_hat = torch.matmul(W, x)
        # Initialize routing logits to zero.
        b_ij = Variable(torch.zeros(1, self.in_channels, self.num_units, 1)).cuda()
        # Iterative routing.
        num_iterations = 3
        for iteration in range(num_iterations):
            # Convert routing logits to softmax.
            # (batch, features, num_units, 1, 1)
            c_ij = F.softmax(b_ij,dim=1).to(device)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4).cuda()
            # Apply routing (c_ij) to weighted inputs (u_hat).
            # (batch_size, 1, num_units, unit_size, 1)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            # (batch_size, 1, num_units, unit_size, 1)
            v_j = CapsuleLayer.squash(s_j)
            # (batch_size, features, num_units, unit_size, 1)
            v_j1 = torch.cat([v_j] * self.in_channels, dim=1).cuda()
            # (1, features, num_units, 1)
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True).cuda()
            # Update b_ij (routing)
            b_ij = b_ij + u_vj1
        v_j = v_j.squeeze(1)
        v_j = v_j.squeeze(3)
        return v_j

class SelfRouting2d(nn.Module):
    def __init__(self, in_units, num_units,  unit_size):
        super(SelfRouting2d, self).__init__()

        self.num_units = num_units
        self.W1 = nn.Parameter(torch.FloatTensor(1, 1, num_units, unit_size, in_units))
        self.W2 = nn.Parameter(torch.FloatTensor(1, 1 ,1, unit_size, in_units))
        self.b2 = nn.Parameter(torch.FloatTensor(1, 1 ,1,unit_size))
        nn.init.kaiming_uniform_(self.W1.data)
        
        nn.init.constant_(self.W2.data, 0)
        nn.init.constant_(self.b2.data, 0)
        
        #self.orientation = Agreement_filter()
    def forward(self, x):
        #x = squash2(x)
        batch_size = x.size(0)
        # (batch, in_units, features) -> (batch, features, in_units)
        x = x.transpose(1, 2)
        x = torch.stack([x] * self.num_units, dim=2).unsqueeze(4)
        W1 = torch.cat([self.W1] * batch_size, dim=0)
        W2 = torch.cat([self.W2] * batch_size, dim=0)
        W2 = torch.stack([W2] * self.num_units, dim=2).squeeze(3)
        u_hat = torch.matmul(W1, x)
        #层归一化

        logit = torch.matmul(W2, x).squeeze(-1) + self.b2
        cij = torch.softmax(logit, dim=2)  # cij
        cij = cij.unsqueeze(4)

        a = torch.norm(x, p=2, dim=3, keepdim=True)
        ar = a * cij
        ar_sum = ar.sum(dim=1, keepdim=True)
        coeff = (ar / (ar_sum))
        pose_out = (coeff * u_hat).sum(dim=1)
        pose_out = pose_out.squeeze(3)
        #pose_out = squash2(pose_out)
        
        return  pose_out


#new trying
class fSelfRouting2d(nn.Module):
    def __init__(self,num_units, in_size,  out_size):
        super(fSelfRouting2d, self).__init__()

        self.num_units = num_units
        self.W1 = nn.Parameter(torch.FloatTensor(1,  1,  out_size, in_size,))
        self.W2 = nn.Parameter(torch.FloatTensor(1,  1,  out_size, in_size))
        self.b2 = nn.Parameter(torch.FloatTensor(1,  1,  out_size,1))
        nn.init.kaiming_uniform_(self.W1.data)
        nn.init.kaiming_uniform_(self.W2.data)
        #nn.init.constant_(self.W2.data, 0)
        nn.init.constant_(self.b2.data, 0)
        # self.orientation = Agreement_filter()
        #self.filter_weight = filter_weight()
    def forward(self, x):

        batch_size = x.size(0)
        # (batch, in_units, features) -> (batch, features, in_units)
        x = x.transpose(1, 2)
        #x = torch.stack([x] * self.num_units, dim=2).unsqueeze(4)
        x = x.unsqueeze(3)
        W1 = torch.cat([self.W1] * batch_size, dim=0)
        W2 = torch.cat([self.W2] * batch_size, dim=0)
        #W2 = torch.stack([W2] * self.num_units, dim=2).squeeze(3)
        u_hat = torch.matmul(W1,x)
        u_hat = u_hat.transpose(1, 2)
        num1 = u_hat.size(2)
        a = torch.norm(u_hat, p=2, dim=1, keepdim=True)

        a2, indices = torch.sort(a, dim=2, descending=True)
        amax = a2[:, :, 0,]
        amean = torch.mean(a2, dim=2)
        amed = torch.median(a2, dim=2)[0]
        prop = amed / amax
        prop = torch.mean(prop, dim=0)
        knum = prop * num1
        print(prop)
        index1 = indices[:, :, :int(knum),:]
        a1, b, c,_ = u_hat.size()
        _, _, c2,_ = index1.size()
        a = []
        for i in range(0, a1):
            a2 = u_hat[i, :, index1[i]]
            a.append(a2)
        cap = torch.cat(a, dim=0)
        u_hat = cap.view(a1, b, c2,-1)

        logit = torch.matmul(W2,x) + self.b2
        logit = logit.transpose(1,2)

        log = []
        for i in range(0,a1):
            a2 = logit[i,:,index1[i]]
            log.append(a2)
        logit = torch.cat(log,dim=0)
        logit = logit.view(a1, b, c2,1)
        logit = logit.transpose(1,2)
        logit  = torch.stack([logit] * self.num_units, dim=2)
        cij = torch.softmax(logit, dim=2)  # cij
        a = torch.norm(x, p=2, dim=2, keepdim=True)

        aa = []
        for i in range(0,a1):
            a_a = a[i,index1[i],:]
            aa.append(a_a)
        aa = torch.cat(aa,dim=0)
        a = aa.view(a1,c2,1,1)
        a = torch.stack([a] * self.num_units, dim=2)
        ar =  a * cij
        ar_sum = ar.sum(dim=1, keepdim=True)
        coeff = (ar / (ar_sum)).squeeze(-1)
        u_hat = u_hat.transpose(1,2).squeeze(-1)
        u_hat = torch.stack([u_hat] * self.num_units, dim=2)
        pose_out = (coeff * u_hat).sum(dim=1)
        #pose_out = pose_out.squeeze(3)
        # pose_out = squash2(pose_out)

        return pose_out



#decoder
class Decoder_cifar10(nn.Module):
    def __init__(self, caps_size=16, num_caps=1, img_size=32, img_channels=3):
        super(Decoder_cifar10,self).__init__()

        self.num_caps = num_caps
        self.img_channels = img_channels
        self.img_size = img_size

        self.dense = torch.nn.Linear(caps_size * num_caps, 1024)
        self.relu = nn.ReLU(inplace=True)

        self.reconst_layers1 = nn.Sequential(nn.BatchNorm2d(num_features=16, momentum=0.8),
                                             nn.ConvTranspose2d(in_channels=16, out_channels=64,
                                                                kernel_size=3, stride=1, padding=1
                                                                )
                                             )
        self.reconst_layers2 = nn.ConvTranspose2d(in_channels=64, out_channels=32,
                                                  kernel_size=3, stride=2, padding=1
                                                  )

        self.reconst_layers3 = nn.ConvTranspose2d(in_channels=32, out_channels=16,
                                                  kernel_size=3, stride=2, padding=1
                                                  )

        self.reconst_layers4 = nn.ConvTranspose2d(in_channels=16, out_channels=3,
                                                  kernel_size=3, stride=1, padding=1
                                                  )
        self.reconst_layers5 = nn.ReLU()

    def forward(self, x):
        # x.shape = (batch, 1, capsule_dim(=16))
        batch = x.shape[0]

        x = torch.reshape(x,(batch,1,-1))

        x = self.dense(x)
        x = self.relu(x)
        x = x.reshape(-1, 16, 8, 8)
        x = self.reconst_layers1(x)

        x = self.reconst_layers2(x)

        # padding
        p2d = (1, 0, 1, 0)
        x = func.pad(x, p2d, "constant", 0)
        x = self.reconst_layers3(x)

        # padding
        p2d = (1, 0, 1, 0)
        x = func.pad(x, p2d, "constant", 0)
        x = self.reconst_layers4(x)

        x = self.reconst_layers5(x)

        x = x.reshape(-1, 3, self.img_size, self.img_size)

        return x  # dim: (batch, 1, imsize, imsize)



class Combine(nn.Module):
    def __init__(self):
        super(Combine,self).__init__()

    def forward(self, x):
        # x.shape = (batch, 1, capsule_dim(=16))
        batch = x.shape[0]
        a = torch.norm(x, p=2, dim=2, keepdim=True)
        ax = torch.softmax(a,dim=1)
        x = x*ax
        x = torch.sum(x,dim=1,keepdim=True)

        return x  # dim: (batch, 1, imsize, imsize)

#find the capsule which has biggest modules
class findmax(nn.Module):
    def __init__(self):
        super(findmax,self).__init__()

    def forward(self, x):
        # x.shape = (batch, 1, capsule_dim(=16))
        v_mag = torch.sqrt((x ** 2).sum(dim=2, keepdim=True))
        _, v_max_index = v_mag.max(dim=1)
        v_max_index = v_max_index.data
        ba, num, _ = v_mag.size()
        aa = []
        for i in range(0, ba):
            ind = v_max_index[i]
            a2 = x[i, ind]
            aa.append(a2)
        output = torch.cat(aa, dim=0)
        output =output.unsqueeze(1)
        return output # dim: (batch, 1, imsize, imsize)
