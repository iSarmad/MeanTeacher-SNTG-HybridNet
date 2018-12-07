import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.init import kaiming_normal_
from torch.nn.utils import weight_norm
from torch.autograd.variable import Variable
import math

__all__ = ['hybridnet']


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


# Copied from Hybrid Net Repository
#https://github.com/dakshitagrawal97/HybridNet
class Net(nn.Module):
    def __init__(self,args,std = 0.15):
        super(Net, self).__init__()
        self.args = args

        self.std = std
        self.gn = GaussianNoise(shape=(args.batch_size,3,32,32),std=self.std)
        #
        # encoder for classification branch
        self.conv1c = torch.nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.conv2c = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3c = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool1c = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout1c = torch.nn.Dropout()
        self.conv4c = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5c = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv6c = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool2c = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout2c = torch.nn.Dropout()
        self.conv7c = torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.conv8c = torch.nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv9c = torch.nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)

        # encoder for unsupervised branch
        self.conv1u = torch.nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.conv2u = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3u = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool1u = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        self.dropout1u = torch.nn.Dropout()
        self.conv4u = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5u = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv6u = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool2u = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        self.dropout2u = torch.nn.Dropout()
        self.conv7u = torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.conv8u = torch.nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv9u = torch.nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)

        # decoder for classification branch
        self.tconv9c = torch.nn.ConvTranspose2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.tconv8c = torch.nn.ConvTranspose2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.tconv7c = torch.nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.upsample1c = torch.nn.Upsample(scale_factor=2)
        self.tconv6c = torch.nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.tconv5c = torch.nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.tconv4c = torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.upsample2c = torch.nn.Upsample(scale_factor=2)
        self.tconv3c = torch.nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.tconv2c = torch.nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.tconv1c = torch.nn.ConvTranspose2d(128, 3, kernel_size=3, stride=1, padding=1)

        # decoder for unsupervised branch
        self.tconv9u = torch.nn.ConvTranspose2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.tconv8u = torch.nn.ConvTranspose2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.tconv7u = torch.nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.unpool1u = torch.nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.tconv6u = torch.nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.tconv5u = torch.nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.tconv4u = torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.unpool2u = torch.nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.tconv3u = torch.nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.tconv2u = torch.nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.tconv1u = torch.nn.ConvTranspose2d(128, 3, kernel_size=3, stride=1, padding=1)

        # batch normalization layers for classification branch
        self.bnconv1c = torch.nn.BatchNorm2d(128)
        self.bnconv2c = torch.nn.BatchNorm2d(128)
        self.bnconv3c = torch.nn.BatchNorm2d(128)
        self.bnconv4c = torch.nn.BatchNorm2d(256)
        self.bnconv5c = torch.nn.BatchNorm2d(256)
        self.bnconv6c = torch.nn.BatchNorm2d(256)
        self.bnconv7c = torch.nn.BatchNorm2d(512)
        self.bnconv8c = torch.nn.BatchNorm2d(256)
        self.bnconv9c = torch.nn.BatchNorm2d(128)
        self.bntconv9c = torch.nn.BatchNorm2d(256)
        self.bntconv8c = torch.nn.BatchNorm2d(512)
        self.bntconv7c = torch.nn.BatchNorm2d(256)
        self.bntconv6c = torch.nn.BatchNorm2d(256)
        self.bntconv5c = torch.nn.BatchNorm2d(256)
        self.bntconv4c = torch.nn.BatchNorm2d(128)
        self.bntconv3c = torch.nn.BatchNorm2d(128)
        self.bntconv2c = torch.nn.BatchNorm2d(128)
        self.bntconv1c = torch.nn.BatchNorm2d(3)

        # batch normalization layers for unsupervised branch
        self.bnconv1u = torch.nn.BatchNorm2d(128)
        self.bnconv2u = torch.nn.BatchNorm2d(128)
        self.bnconv3u = torch.nn.BatchNorm2d(128)
        self.bnconv4u = torch.nn.BatchNorm2d(256)
        self.bnconv5u = torch.nn.BatchNorm2d(256)
        self.bnconv6u = torch.nn.BatchNorm2d(256)
        self.bnconv7u = torch.nn.BatchNorm2d(512)
        self.bnconv8u = torch.nn.BatchNorm2d(256)
        self.bnconv9u = torch.nn.BatchNorm2d(128)
        self.bntconv9u = torch.nn.BatchNorm2d(256)
        self.bntconv8u = torch.nn.BatchNorm2d(512)
        self.bntconv7u = torch.nn.BatchNorm2d(256)
        self.bntconv6u = torch.nn.BatchNorm2d(256)
        self.bntconv5u = torch.nn.BatchNorm2d(256)
        self.bntconv4u = torch.nn.BatchNorm2d(128)
        self.bntconv3u = torch.nn.BatchNorm2d(128)
        self.bntconv2u = torch.nn.BatchNorm2d(128)
        self.bntconv1u = torch.nn.BatchNorm2d(3)




        # classifier
        self.avgpool1c = torch.nn.AvgPool2d(kernel_size=6, stride=1)
        self.fc1c = torch.nn.Linear(128, 10)


        self.dense = (nn.Linear(128, 10))
        self.BNdense = nn.BatchNorm1d(10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()




    def forward(self, x):

        if self.training:
             x = self.gn(x)

        # forward pass through the encoder of the classification branch
        conv1c = F.leaky_relu(self.bnconv1c(self.conv1c(x)), negative_slope=0.1)
        conv2c = F.leaky_relu(self.bnconv2c(self.conv2c(conv1c)), negative_slope=0.1)
        conv3c = F.leaky_relu(self.bnconv3c(self.conv3c(conv2c)), negative_slope=0.1)
        pool1c = self.pool1c(conv3c)
        dropout1c = self.dropout1c(pool1c)
        conv4c = F.leaky_relu(self.bnconv4c(self.conv4c(dropout1c)), negative_slope=0.1)
        conv5c = F.leaky_relu(self.bnconv5c(self.conv5c(conv4c)), negative_slope=0.1)
        conv6c = F.leaky_relu(self.bnconv6c(self.conv6c(conv5c)), negative_slope=0.1)
        pool2c = self.pool2c(conv6c)
        dropout2c = self.dropout2c(pool2c)
        conv7c = F.leaky_relu(self.bnconv7c(self.conv7c(dropout2c)), negative_slope=0.1)
        conv8c = F.leaky_relu(self.bnconv8c(self.conv8c(conv7c)), negative_slope=0.1)
        conv9c = F.leaky_relu(self.bnconv9c(self.conv9c(conv8c)), negative_slope=0.1)

        # forward pass through the classifier
        global_avg_c = self.avgpool1c(conv9c).view(conv9c.size(0), -1)
        y = self.fc1c(global_avg_c)

        # forward pass through the decoder of the classification branch
        tconv9c = F.leaky_relu(self.bntconv9c(self.tconv9c(conv9c)), negative_slope=0.1)
        tconv8c = F.leaky_relu(self.bntconv8c(self.tconv8c(tconv9c)), negative_slope=0.1)
        tconv7c = F.leaky_relu(self.bntconv7c(self.tconv7c(tconv8c)), negative_slope=0.1)
        upsample1c = self.upsample1c(tconv7c)
        tconv6c = F.leaky_relu(self.bntconv6c(self.tconv6c(upsample1c)), negative_slope=0.1)
        tconv5c = F.leaky_relu(self.bntconv5c(self.tconv5c(tconv6c)), negative_slope=0.1)
        tconv4c = F.leaky_relu(self.bntconv4c(self.tconv4c(tconv5c)), negative_slope=0.1)
        upsample2c = self.upsample2c(tconv4c)
        tconv3c = F.leaky_relu(self.bntconv3c(self.tconv3c(upsample2c)), negative_slope=0.1)
        tconv2c = F.leaky_relu(self.bntconv2c(self.tconv2c(tconv3c)), negative_slope=0.1)
        x_c = F.leaky_relu(self.bntconv1c(self.tconv1c(tconv2c)), negative_slope=0.1)

        # forward pass through the encoder of the unsupervised branch
        conv1u = F.leaky_relu(self.bnconv1u(self.conv1u(x)), negative_slope=0.1)
        conv2u = F.leaky_relu(self.bnconv2u(self.conv2u(conv1u)), negative_slope=0.1)
        conv3u = F.leaky_relu(self.bnconv3u(self.conv3u(conv2u)), negative_slope=0.1)
        pool1u, indices1u = self.pool1u(conv3u)
        dropout1u = self.dropout1u(pool1u)
        conv4u = F.leaky_relu(self.bnconv4u(self.conv4u(dropout1u)), negative_slope=0.1)
        conv5u = F.leaky_relu(self.bnconv5u(self.conv5u(conv4u)), negative_slope=0.1)
        conv6u = F.leaky_relu(self.bnconv6u(self.conv6u(conv5u)), negative_slope=0.1)
        pool2u, indices2u = self.pool2u(conv6u)
        dropout2u = self.dropout2u(pool2u)
        conv7u = F.leaky_relu(self.bnconv7u(self.conv7u(dropout2u)), negative_slope=0.1)
        conv8u = F.leaky_relu(self.bnconv8u(self.conv8u(conv7u)), negative_slope=0.1)
        conv9u = F.leaky_relu(self.bnconv9u(self.conv9u(conv8u)), negative_slope=0.1)

        # forward pass through the decoder of the unsupervised branch
        tconv9u = F.leaky_relu(self.bntconv9u(self.tconv9u(conv9u)), negative_slope=0.1)
        tconv8u = F.leaky_relu(self.bntconv8u(self.tconv8u(tconv9u)), negative_slope=0.1)
        tconv7u = F.leaky_relu(self.bntconv7u(self.tconv7u(tconv8u)), negative_slope=0.1)
        unpool1u = self.unpool1u(tconv7u, indices2u)
        tconv6u = F.leaky_relu(self.bntconv6u(self.tconv6u(unpool1u)), negative_slope=0.1)
        tconv5u = F.leaky_relu(self.bntconv5u(self.tconv5u(tconv6u)), negative_slope=0.1)
        tconv4u = F.leaky_relu(self.bntconv4u(self.tconv4u(tconv5u)), negative_slope=0.1)
        unpool2u = self.unpool2u(tconv4u, indices1u)
        tconv3u = F.leaky_relu(self.bntconv3u(self.tconv3u(unpool2u)), negative_slope=0.1)
        tconv2u = F.leaky_relu(self.bntconv2u(self.tconv2u(tconv3u)), negative_slope=0.1)
        x_u = F.leaky_relu(self.bntconv1u(self.tconv1u(tconv2u)), negative_slope=0.1)

        return y, x_c, x_u


def hybridnet(args,data= None,nograd=False):
    model = Net(args)
    if data is not None:
        model.load_state_dict(data['state_dict'])

    model = model.cuda()
    model = nn.DataParallel(model).cuda()

    if nograd:
        for param in model.parameters():
            param.detach_()


    return model
