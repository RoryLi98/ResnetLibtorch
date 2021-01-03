import torch 
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import math
import numpy as np
import cv2
from time import time
from torch.autograd import Variable
from torchvision.transforms import Compose, ToTensor, Resize
import gc
gc.collect()

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

#====================================================================================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model =torch.load("/home/link/NetworkModel1/net_cpu_Adam_cross_B20_S200_E5.pth")                  #Ubuntu
model =torch.load("D:\\Github\\Resnet-Libtorch\\Resnet_MNIST_GPU_Adam_cross_E3_B32_GTX950.pth")     #Windows GPU
# model =torch.load("D:\\Github\\Resnet-Libtorch\\Resnet_MNIST_CPU_Adam_cross_E5_B20_S400.pth")       #Windows CPU

model = model.to(device)                                                 # GPU MOD
model.eval()
script_model = torch.jit.script(model)
script_model.save("MNIST-GPU.pt")

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.

# img_cv = cv2.imread("/home/link/NetworkModel1/C++/5.png")               #Ubuntu
img_cv = cv2.imread("D:\\Github\\Resnet-Libtorch\\Example\\5.png")            #Windows
img_gray = cv2.cvtColor(img_cv,cv2.COLOR_RGB2GRAY)
img_np = cv2.resize(img_gray, (224, 224))
img_np = np.expand_dims(img_np ,axis=0)                                   # appear at the axis position in the expanded array shape.

tensor_cv = torch.from_numpy(img_np)                                      # Mat 2 Tensor

img_np1=tensor_cv.unsqueeze(0)
print("img_cv.shape：", img_cv.shape)
print("img_cv.ndim：", img_cv.ndim)
print("img_np.shape：", img_np.shape)
print("img_np.ndim：", img_np.ndim)
print("tensor_cv.size：", tensor_cv.size())
print("img_np1.size：", img_np1.size())
img_np1 = img_np1.float()
img_np1 = img_np1.div(255)

img_np1 = img_np1.to(device)                                               # GPU MOD
print("-------------------------")
print("SourceModel:")
start = time()
output1 = model(img_np1)
stop = time()
print("UsingTime:"+str(stop-start) + "s")
print(output1)
print(output1.max(1))

print("-------------------------")
print("ScriptModel:")
start = time()
output = script_model(img_np1)
stop = time()
print("UsingTime:"+str(stop-start) + "s")
print(output)
print(output.max(1))

# Label	Description
# 0	    0
# 1	    1
# 2	    2
# 3	    3
# 4	    4
# 5	    5
# 6	    6
# 7	    7
# 8	    8
# 9	    9