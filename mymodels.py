import math
from functools import partial

import torch
import torch.nn as nn

mymodels = ["myresnet18"]

class MyResnetBasicBlock(nn.Module):
    def __init__(self,in_planes,out_planes,stride=1):
        super(MyResnetBasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_planes,out_planes,kernel_size=3,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.downsample = None
        if stride > 1 or in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes,out_planes,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_planes)
            )
    def forward(self,x):
        cached = x
        o = self.conv1(x)
        o = self.bn1(o)
        o = self.relu1(o)

        o = self.conv2(o)
        o = self.bn2(o)

        if self.downsample:
            cached = self.downsample(cached)
        
        o = self.relu2(o + cached)
        return o
        
        
class MyResnet18(nn.Module):
    def __init__(self):
        super(MyResnet18,self).__init__()
        self.conv1 = nn.Conv2d(3,16,kernel_size=3,padding=1,bias=False)
        self.first_block = nn.Sequential(
            MyResnetBasicBlock(16,16),
            MyResnetBasicBlock(16,16),
            MyResnetBasicBlock(16,16)
        )
        self.second_block = nn.Sequential(
            MyResnetBasicBlock(16,32,stride=2),
            MyResnetBasicBlock(32,32),
            MyResnetBasicBlock(32,32)
        )
        self.third_block = nn.Sequential(
            MyResnetBasicBlock(32,64,stride=2),
            MyResnetBasicBlock(64,64),
            MyResnetBasicBlock(64,64)
        )
        self.avg_pool = nn.AvgPool2d(2)
        self.fc = nn.Linear(1024,10)

    def forward(self,x):
        o = self.conv1(x)
        o = self.first_block(o)
        o = self.second_block(o)
        o = self.third_block(o)
        o = self.avg_pool(o)
        o = o.view((-1,1024))
        o = self.fc(o)
        return o

def create_mymodel(model_name):
    if model_name == "myresnet18":
        return MyResnet18()
    else:
        return None
