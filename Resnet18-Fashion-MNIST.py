import torch 
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import math
import numpy as np
import cv2
import datetime
from time import time
from torch.autograd import Variable
from torchvision.transforms import Compose, ToTensor, Resize
import gc
gc.collect()

#对输入图像进行处理，转换为（224，224）,因为resnet18要求输入为（224，224），并转化为tensor
def input_transform():
    return Compose([
                Resize(224),   #改变尺寸
                ToTensor(),      #变成tensor
                ])

# Mnist 手写数字，数据导入
train_data = torchvision.datasets.FashionMNIST(
    root='./Fashion-mnist/',    # 保存或者提取位置
    train=True,  # this is training data
    transform=input_transform(),    # 转换 PIL.Image or numpy.ndarray 成
                                                    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=True
,          # 没下载就下载, 下载了就不用再下了
)

test_data = torchvision.datasets.FashionMNIST(
    root='./Fashion-mnist/',    # 保存或者提取位置
    train=False,  # this is training data
    transform=input_transform(),    # 转换 PIL.Image or numpy.ndarray 成
                                                    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=True,          # 没下载就下载, 下载了就不用再下了
)


BATCH_SIZE = 32

'''
进行批处理
'''
loader = Data.DataLoader(dataset=train_data,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=0)

'''
定义resnet18
'''
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    #inplanes其实就是channel,叫法不同
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        #把shortcut那的channel的维度统一
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,   #因为mnist为（1，28，28）灰度图，因此输入通道数为1
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        #downsample 主要用来处理H(x)=F(x)+x中F(x)和xchannel维度不匹配问题
        downsample = None
        #self.inplanes为上个box_block的输出channel,planes为当前box_block块的输入channel
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    #[2, 2, 2, 2]和结构图[]X2是对应的
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained: #加载模型权重
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = resnet18()
net = net.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_func = torch.nn.CrossEntropyLoss()
for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(loader):
        b_x = Variable(batch_x).to(device)
        b_y = Variable(batch_y).to(device)
        predict = net(b_x) 
        loss = loss_func(predict, b_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0: 
            print('epoch:{}, step:{}, loss:{}'.format(epoch, step, loss))
            print(datetime.datetime.now())

        # if step==500:
        #     break
net.eval()
# torch.save(net,'net_cpu_Adam_cross_B20_S400.pth')
torch.save(net,'Resnet_MNIST_GPU_Adam_cross_E4_B32_GTX950.pth')
print("All Done")
print(datetime.datetime.now())