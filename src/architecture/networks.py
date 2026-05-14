#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 12:45:00 2021

@author: homai
"""

import os
import torch
import torchvision
import torch.nn as nn

# location where pretrained models will be downloaded
os.environ['TORCH_HOME'] = "../data/models/"

def _cuda_enabled(model):
    if torch.cuda.is_available():
        print('========= Model was sent to CUDA.. yay :) =========')
        model = model.cuda()
    if torch.cuda.device_count() > 1:
        print('========= nn.DataParallel is being used yay :) =========')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model = nn.DataParallel(model)
        print("#devices=",torch.cuda.device_count() )
    return model

def mobilenet_v2(in_channels, out_features):
    model = torchvision.models.mobilenet_v2(pretrained=True)
    model.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=out_features, bias=True)
    return _cuda_enabled(model)

def resnet18(in_channels, out_features):
    model = torchvision.models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, out_features)
    return _cuda_enabled(model)

def resnet50(in_channels, out_features):
    model = torchvision.models.resnet50(pretrained=True)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, out_features)
    return _cuda_enabled(model)

def resnet50_2(in_channels, out_features, fc_nodes= 256):
    model = torchvision.models.resnet50(pretrained=True)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=5, bias=False)
    model.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=fc_nodes), 
                             nn.BatchNorm1d(fc_nodes),
                             nn.ReLU(inplace=True), 
                             nn.Dropout(p=0.5),
                             nn.Linear(in_features=fc_nodes, out_features=out_features))
    return _cuda_enabled(model)
    
def resnet101(in_channels, out_features):
    model = torchvision.models.resnet101(pretrained=True)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, out_features)
    return _cuda_enabled(model)

def wide_resnet50_2(in_channels, out_features):
    model = torchvision.models.wide_resnet50_2(pretrained=True)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, out_features)
    return _cuda_enabled(model)

def wide_resnet101_2(in_channels, out_features):
    model = torchvision.models.wide_resnet101_2(pretrained=True)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, out_features)
    return _cuda_enabled(model)

def densenet121(in_channels, out_features, pretrained, dropout, cuda=True):
    model = torchvision.models.densenet121(pretrained=pretrained, drop_rate=dropout)
    model.features.conv0 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.classifier = nn.Linear(in_features=1024, out_features=out_features, bias=True)
    return _cuda_enabled(model)














