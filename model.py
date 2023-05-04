import torch
import torchvision
import time
import engine as eng
from torch import nn
import easydict
import torchinfo
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from pathlib import Path
from sklearn import preprocessing
import pandas as pd
import glob
import eda

class resnet18(nn.Module):
    def __init__(self):
        super(resnet18,self).__init__()
        self.name = 'resnet18'
        self.layer1 = nn.Linear(1000,32)
        self.Relu1 = nn.ReLU()
        self.Dropout1 = nn.Dropout(p=0.7)
        self.layer2 = nn.Linear(32, 2)
        self.net = torchvision.models.resnet18(weights = torchvision.models.ResNet18_Weights.DEFAULT)
        for p in self.net.parameters():
            p.requires_grad=False
    def forward(self,x):
        return self.layer2(self.Dropout1(self.Relu1(self.layer1(self.net(x)))))
    
class googlenet(nn.Module):
    def __init__(self):
        super(googlenet,self).__init__()
        self.name = 'googlenet'
        self.layer1 = nn.Linear(1000,2)
        self.net = torchvision.models.googlenet(weights = torchvision.models.GoogLeNet_Weights.DEFAULT)
        for p in self.net.parameters():
            p.requires_grad=False
    def forward(self,x):
        return self.layer1(self.net(x))