
import torch
import torch.nn as nn
import torchvision.utils as vutils
import torch.nn.functional as F

from capsule_layer import   capsule,Filter_high,SelfRouting2d,Decoder_cifar10,CapsuleLayer,\
                  Combine,findmax,Self_Attn,cap_reshape
from capsule_conv_layer import squash1,squash2

#整个网络框架
class CapsuleNetwork(nn.Module):
    def __init__(self,
                 image_width,
                 image_height,
                 image_channels,
                 conv_inputs,
                 ):
        super(CapsuleNetwork, self).__init__()
        self.reconstructed_image_count = 0
        self.image_channels = image_channels
        self.image_width = image_width
        self.image_height = image_height
        self.conv_inputs = conv_inputs
        self.abc = capsule(conv_inputs)
        self.cap = cap_reshape()

        #CBAM
        self.self_att1 = Self_Attn(8)
        self.self_att2 = Self_Attn(10)
        self.self_att3 = Self_Attn(12)
        #路由
        self.filter = Filter_high()

        #CFR
        self.L1p = SelfRouting2d(
                                in_units=8,
                               num_units=10,
                               unit_size=10)
        self.L2p = SelfRouting2d(
                                 in_units=10,
                                num_units=10,
                                unit_size=12) 
        self.L3p = SelfRouting2d(
                                in_units=12,
                                num_units=10,
                                unit_size=16)
        
        self.gamma2 = nn.Parameter(torch.tensor(0.7, dtype=torch.float))
        self.W2 = nn.Parameter(torch.FloatTensor(1, 10, 1))
        nn.init.constant_(self.W2.data, 0)
        #self.gamma2 = nn.Parameter(torch.zeros(1))
        #self.gamma3 = nn.Parameter(torch.zeros(1))
        self.gamma3 = nn.Parameter(torch.tensor(0.7, dtype=torch.float))
        #DR
        """
        self.L1p = CapsuleLayer(
                                in_units=8,
                                in_channels=2352,
                               num_units=10,
                               unit_size=10)
        self.L2p = CapsuleLayer(
                                 in_units=10,
                                 in_channels=300,
                                num_units=10,
                                unit_size=12)
        self.L3p = CapsuleLayer(
                                in_units=12,
                                in_channels=108,
                                num_units=10,
                                unit_size=16)
        """

        self.find =findmax()
        #self.Decoder = Decoder_cifar10(caps_size=40, num_caps=1, img_size=32, img_channels=3)
        #FCM
        self.fc1 = torch.nn.Linear(100,10)
        self.fc2 = torch.nn.Linear(120,10)
        self.fc3 = torch.nn.Linear(160,10)
    def forward(self, x):
        #卷积部分
        L1, L2, L3 = self.abc(x)
        L1,L2,L3 = self.cap(L1,L2,L3)
        #筛选1
        L1 = self.filter(L1)
        #CFR
        L1_mid = self.L1p(L1)
        L1_max = self.find(L1_mid)
        """
        L1_max = L1_max.transpose(1,2)
        out = L1_mid.reshape(L1_mid.size(0), -1)
        L1_mid = self.fc1(out)
        """
        #CBAM
        L2 = self.self_att2(L2)
        #筛选2
        L2 = self.filter(L2)
        b = L1_max.size(0)
        W2 = torch.cat([self.W2] * b, dim=0)
        L1_max = W2 * L1_max
        #CSM
        L2 = torch.cat((L2,L1_max),dim=2)
        #CFR
        L2_mid = self.L2p(L2)
        L2_max = self.find(L2_mid)
        L2_max = L2_max.transpose(1,2)
        """
        out = L2_mid.reshape(L2_mid.size(0), -1)
        L2_mid = self.fc2(out)
        """
        #CBAM
        L3 = self.self_att3(L3)
        #筛选3
        L3 = self.filter(L3)
        L2_max = self.gamma3 * L2_max
        #CSM
        L3 = torch.cat((L3,L2_max),dim=2)
        #CFR
        L3_mid = self.L3p(L3)
        """
        out = L3_mid.reshape(L3_mid.size(0), -1)
        L3_mid = self.fc3(out)
        """
        return L1_mid,L2_mid,L3_mid

    def loss(self, images,digit_capsule1,digit_capsule2,digit_capsule3, target, size_average=True):
        #交叉熵损失
        """
        loss = nn.CrossEntropyLoss()
        target = torch.max(target, 1)[1]
        loss1 = loss(digit_capsule1, target)
        loss2 = loss(digit_capsule2, target)
        loss3 = loss(digit_capsule3, target)
        """
        loss1 = self.margin_loss(digit_capsule1,target, size_average)
        loss2 = self.margin_loss(digit_capsule2,target, size_average)
        loss3 = self.margin_loss(digit_capsule3,target, size_average)
        
        #output = torch.cat((self.L1_max,self.L2_max,self.L3_max),dim=2)
        #re_loss = self.reconstruction_loss(images, output, size_average)
        
        if size_average==True:
          print('Loss1: {:.6f},Loss2: {:.6f},Loss3: {:.6f}'.format(loss1.data,loss2.data,loss3.data))
        loss = loss1+ loss2+ loss3 #+ re_loss
        return loss


    def margin_loss(self, input, target, size_average=True):
        batch_size = input.size(0)
        # ||vc|| from the paper.
        v_mag = torch.sqrt((input**2).sum(dim=2, keepdim=True))

        # Calculate left and right max() terms from equation 4 in the paper.
        #zero = Variable(torch.zeros(1))
        m_plus = 0.9
        m_minus = 0.1
        max_l =  F.relu(m_plus - v_mag).view(batch_size, -1) ** 2
        max_r = F.relu(v_mag - m_minus).view(batch_size, -1) ** 2

        # This is equation 4 from the paper.
        loss_lambda = 0.5
        T_c = target
        L_c = T_c * max_l + loss_lambda * (1.0 - T_c) * max_r
        L_c = L_c.sum(dim=1)


        if size_average:
            L_c = L_c.mean()

        return L_c

    #重构损失
    def reconstruction_loss(self, images, input, size_average=True):
        # Get the lengths of capsule outputs.

        output = self.Decoder(input)

        # Save reconstructed images occasionally.
        if self.reconstructed_image_count % 10 == 0:
            if output.size(1) == 2:
                # handle two-channel images
                zeros = torch.zeros(output.size(0), 1, output.size(2), output.size(3))
                output_image = torch.cat([zeros, output.data.cpu()], dim=1)
            else:
                # assume RGB or grayscale
                output_image = output.data.cpu()
            vutils.save_image(output_image, "reconstruction.png")
        self.reconstructed_image_count += 1
        mse_loss = nn.MSELoss(reduction='none')
        loss = mse_loss(output.view(output.shape[0], -1), images.view(output.shape[0], -1))
        error = torch.sum(loss, dim=1) *0.0005

        if size_average:
            error = error.mean()
        return error



