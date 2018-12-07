import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.init import kaiming_normal_
from torch.nn.utils import weight_norm
from torch.autograd.variable import Variable
import math

__all__ = ['convlarge']


# noise function taken from blog : https://ferretj.github.io/ml/2018/01/22/temporal-ensembling.html?fbclid=IwAR1MEqzhwrl1swzLUDA0kZFN2oVTdcNa497c1l3pC-Xh2kYPlPjRiO0Oucc
class GaussianNoise(nn.Module):

    def __init__(self, shape=(100, 1, 28, 28), std=0.05):
        super(GaussianNoise, self).__init__()
        self.noise1 = Variable(torch.zeros(shape).cuda())
        self.std1 = std
        self.register_buffer('noise2',self.noise1) # My own contribution , registering buffer for data parallel usage

    def forward(self, x):
        c = x.shape[0]
        self.noise2.data.normal_(0, std=self.std1)
        return x + self.noise2[:c]


class Net(nn.Module):
    def __init__(self,args,std = 0.15):
        super(Net, self).__init__()
        self.args = args

        self.std = std
        self.gn = GaussianNoise(shape=(args.batch_size,3,32,32),std=self.std)
        if self.args.BN:
            self.BN1a = nn.BatchNorm2d(128)
            self.BN1b = nn.BatchNorm2d(128)
            self.BN1c = nn.BatchNorm2d(128)
        self.conv1a = (nn.Conv2d(3, 128, 3,padding=1))
        self.conv1b = (nn.Conv2d(128, 128, 3,padding=1))
        self.conv1c = (nn.Conv2d(128, 128, 3,padding=1))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.5)
        if self.args.BN:
            self.BN2a = nn.BatchNorm2d(256)
            self.BN2b = nn.BatchNorm2d(256)
            self.BN2c = nn.BatchNorm2d(256)
        self.conv2a = (nn.Conv2d(128, 256, 3, padding=1))
        self.conv2b = (nn.Conv2d(256, 256, 3, padding=1))
        self.conv2c = (nn.Conv2d(256, 256, 3, padding=1))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.5)#nn.Dropout2d

        if self.args.BN:
            self.BN3a = nn.BatchNorm2d(512)
            self.BN3b = nn.BatchNorm2d(256)
            self.BN3c = nn.BatchNorm2d(128)
        self.conv3a = (nn.Conv2d(256, 512, 3))
        self.conv3b = (nn.Conv2d(512, 256, 1))
        self.conv3c = (nn.Conv2d(256, 128, 1))
        self.pool3 = nn.AvgPool2d(6,6)


        self.dense = (nn.Linear(128, 10))
        if self.args.BN:
            self.BNdense = nn.BatchNorm1d(10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # for m in self.modules(): # TODO THIS IS A BIG PROBLEM
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv1d) or isinstance(
        #             m, nn.Linear):
        #         kaiming_normal_(m.weight.data)  # initialize weigths with normal distribution
        #         if m.bias is not None:
        #             m.bias.data.zero_()  # initialize bias as zero
        #     elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()



    def forward(self, x):

        if self.training:
             x = self.gn(x)
        if self.args.BN:
            x = F.leaky_relu(self.BN1a(self.conv1a(x)),negative_slope = 0.1)#self.BN1a
            x = F.leaky_relu(self.BN1b(self.conv1b(x)),negative_slope = 0.1)#self.BN1b
            x = F.leaky_relu(self.BN1c(self.conv1c(x)),negative_slope = 0.1)#self.BN1c
            x = self.drop1(self.pool1(x))#

            x = F.leaky_relu(self.BN2a(self.conv2a(x)), negative_slope = 0.1)#self.BN2a
            x = F.leaky_relu(self.BN2b(self.conv2b(x)), negative_slope = 0.1)#self.BN2b
            x = F.leaky_relu(self.BN2c(self.conv2c(x)), negative_slope = 0.1)#self.BN2c
            x = self.drop2(self.pool2(x))#

            x = F.leaky_relu(self.BN3a(self.conv3a(x)),negative_slope = 0.1)#self.BN3a
            x = F.leaky_relu(self.BN3b(self.conv3b(x)),negative_slope = 0.1)#self.BN3b
            x = F.leaky_relu(self.BN3c(self.conv3c(x)),negative_slope = 0.1)#self.BN3c
            x = self.pool3(x)


            h = x

            x = x.view(-1, 128)
            x = self.BNdense(self.dense(x))#F.softmax(,dim=1)# self.BNdense
        else:
            x = F.leaky_relu((self.conv1a(x)),negative_slope = 0.1)#self.BN1a
            x = F.leaky_relu((self.conv1b(x)),negative_slope = 0.1)#self.BN1b
            x = F.leaky_relu((self.conv1c(x)),negative_slope = 0.1)#self.BN1c
            x = self.drop1(self.pool1(x))#

            x = F.leaky_relu((self.conv2a(x)), negative_slope = 0.1)#self.BN2a
            x = F.leaky_relu((self.conv2b(x)), negative_slope = 0.1)#self.BN2b
            x = F.leaky_relu((self.conv2c(x)), negative_slope = 0.1)#self.BN2c
            x = self.drop2(self.pool2(x))#

            x = F.leaky_relu((self.conv3a(x)),negative_slope = 0.1)#self.BN3a
            x = F.leaky_relu((self.conv3b(x)),negative_slope = 0.1)#self.BN3b
            x = F.leaky_relu((self.conv3c(x)),negative_slope = 0.1)#self.BN3c
            x = self.pool3(x)


            h = x

            x = x.view(-1, 128)
            x =(self.dense(x))#F.softmax(,dim=1)# self.BNdense


        if self.args.sntg == True:
            return x,h
        else:
            return x


def convlarge(args,data= None,nograd=False):
    model = Net(args)
    if data is not None:
        model.load_state_dict(data['state_dict'])



    model = model.cuda()
    model = nn.DataParallel(model).cuda()

    if nograd:
        for param in model.parameters():
            param.detach_()


    return model
