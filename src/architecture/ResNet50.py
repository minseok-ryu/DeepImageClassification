#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 12:36:47 2021

@author: homai
"""

# # -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50
from torchvision.models import resnet18

def resNet50(fc_nodes, num_class, pretrained=True, changing=False):
    net = resnet50(pretrained=pretrained)
    
    if changing:
        net.fc = nn.Sequential(
                            nn.Linear(in_features=2048, out_features=fc_nodes), 
                            nn.BatchNorm1d(fc_nodes),
                            nn.ReLU(inplace=True), 
                            nn.Dropout(p=0.5),
                            nn.Linear(in_features=fc_nodes, out_features=num_class)
                            )
    else:
        net.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=num_class))
        
    return net
    
    