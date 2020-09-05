import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class STPN(nn.Module):
    def __init__(self, num_classes=20,stream="rgb"):
        super(STPN, self).__init__()
        self.stream = stream
        D = 1024
        self.fc1 = nn.Linear(D,256)
        nn.init.normal_(self.fc1.weight, std=0.001)
        nn.init.constant_(self.fc1.bias, 0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256,1)
        nn.init.normal_(self.fc2.weight, std=0.001)
        nn.init.constant_(self.fc2.bias, 0)
        self.sigmoid = nn.Sigmoid()
        self.fc3 = nn.Linear(D, num_classes)

        nn.init.normal_(self.fc3.weight, std=0.001)
        nn.init.constant_(self.fc3.bias, 0)
    
    def forward(self, x):
        inp = self.sigmoid(self.fc2(self.relu(self.fc1(x)))) ## 128,90,1024, 128,90,1
        x = x*inp
        x = torch.sum(x, dim=1)
        x = self.fc3(x) ## 128,90,20  #128,90,1
        x = self.sigmoid(x)
        return x, inp