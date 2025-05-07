import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from mamba_ssm import Mamba
from torch import nn
from torchvision.models.resnet import Bottleneck

class ResNet269(nn.Module):
    def __init__(self, num_classes=20):
        super(ResNet269, self).__init__()

        # Stem部分与标准ResNet相同
        self.inplanes = 64
        # 加mamba
        self.SSM = nn.Sequential(
            Mamba(3, d_state=16, expand=2),
            nn.PReLU())
        ###################
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet的4个主层，block数量更多：在152基础上继续增加
        self.layer1 = self._make_layer(Bottleneck, 64, 6)   # 原来是3
        self.layer2 = self._make_layer(Bottleneck, 128, 12, stride=2)  # 原来是8
        self.layer3 = self._make_layer(Bottleneck, 256, 64, stride=2)  # 原来是36
        self.layer4 = self._make_layer(Bottleneck, 512, 6, stride=2)   # 原来是3

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)



    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []

        # 第一个block，可能需要降采样
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        # 后续block
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 加mamba
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.SSM(x)
        x = x.reshape(B, H, W, C).transpose(1, 3)
        ###################
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # /4
        x = self.layer2(x)  # /8
        x = self.layer3(x)  # /16
        x = self.layer4(x)  # /32

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class _PSPModule(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size):
        super(_PSPModule, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.GroupNorm(32, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.size()[2:]
        x = self.pool(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        return x


class PSPNet(nn.Module):
    def __init__(self, num_classes):
        super(PSPNet, self).__init__()
        self.num_classes = num_classes

        # Backbone: ResNet50 pretrained
        resnet = ResNet269()
        # resnet = models.resnet152(pretrained=True)
        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Pyramid Pooling Module
        self.ppm1 = _PSPModule(2048, 512, pool_size=1)
        self.ppm2 = _PSPModule(2048, 512, pool_size=2)
        self.ppm3 = _PSPModule(2048, 512, pool_size=3)
        self.ppm4 = _PSPModule(2048, 512, pool_size=6)

        # Extra Conv Block (增强特征表达)
        self.extra_conv = nn.Sequential(nn.Conv2d(2048 + 4 * 512, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))



        # Bottleneck after Extra Conv
        self.bottleneck = nn.Sequential(
            nn.Dropout2d(0.3),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.size()[2:]

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # PSP Module
        p1 = self.ppm1(x)
        p2 = self.ppm2(x)
        p3 = self.ppm3(x)
        p4 = self.ppm4(x)

        x = torch.cat([x, p1, p2, p3, p4], dim=1)
        x = self.extra_conv(x)  # Extra feature enhancement
        x = self.bottleneck(x)

        # Upsample to input size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        return x

