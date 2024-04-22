# # -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class cnn(nn.Module):
    def __init__(self):
      super().__init__() # could be written as: super(network, self).__init__() *** what is the differece?   
      self.CNN = nn.Sequential(
          nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), #224x224
          nn.BatchNorm2d(32),
          nn.ReLU(),
          nn.MaxPool2d(2, 2), # 112x112
          nn.ReLU(),

          nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), #112x112
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.MaxPool2d(2, 2), #56x56
          nn.ReLU(),

          nn.Flatten(), 
          nn.Linear(64*56*56, 512),
          nn.BatchNorm1d(512),
          nn.ReLU(),
          nn.Dropout(p=0.5),
          
          nn.Linear(512,64),
          nn.BatchNorm1d(64),
          nn.ReLU(),
          nn.Dropout(p=0.5),
          nn.Linear(64, 2))
        
    def forward(self, xb):
        return self.CNN(xb)
    

