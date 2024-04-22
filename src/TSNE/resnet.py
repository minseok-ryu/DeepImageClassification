#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 16:19:04 2022

@author: homai
"""

import torch
from torch import nn
from torchvision import models
from torch.hub import load_state_dict_from_url


# Define the architecture by modifying resnet.
# Original code is here
# https://github.com/pytorch/vision/blob/b2e95657cd5f389e3973212ba7ddbdcc751a7878/torchvision/models/resnet.py
class ResNet101(models.ResNet):
    def __init__(self, model, model_dir_name, num_classes=2, do_rand_model=False, pretrained=True, **kwargs):
        # # Start with standard resnet101 defined here
        # # https://github.com/pytorch/vision/blob/b2e95657cd5f389e3973212ba7ddbdcc751a7878/torchvision/models/resnet.py
        super().__init__(block=models.resnet.Bottleneck, layers=[3, 4, 23, 3], num_classes=num_classes, **kwargs)
        # if pretrained:
        #     state_dict = load_state_dict_from_url(models.resnet.model_urls['resnet101'], progress=True)
        #     self.load_state_dict(state_dict)
        # self.load_state_dict(state_dict)
        if pretrained and do_rand_model==False:
            if torch.cuda.is_available(): #  ===> map_location=='gpu'
                # print('gpu')
                # if isinstance(model, (nn.DataParallel)):
                #     self.module.load_state_dict(torch.load(model_dir_name))  #  + ".pth"  # your model will be loaded to multi-gpu model.
                # else:
                #     self.load_state_dict(torch.load(model_dir_name)) #  + ".pth"
                
                self.load_state_dict(torch.load(model_dir_name))
            else:                        #  ===> map_location=='cpu'
                print('cpu')
                # model.load_state_dict(torch.load(model_dir, map_location=lambda storage, loc: storage)) # + ".pth"
                model.load_state_dict(torch.load(model_dir_name, map_location='cpu'))  #+ ".pth"
       
        elif pretrained and do_rand_model:
            state_dict = load_state_dict_from_url(models.resnet.model_urls['resnet101'], progress=True)
            self.load_state_dict(state_dict)
            
            
    # Reimplementing forward pass.
    # Replacing the following code
    # https://github.com/pytorch/vision/blob/b2e95657cd5f389e3973212ba7ddbdcc751a7878/torchvision/models/resnet.py#L197-L213
    def _forward_impl(self, x):
        # Standard forward for resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Notice there is no forward pass through the original classifier.
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # basically the  x = self.fc(x) has been commented to be able to extract features (not classification outputs)

        return x
    
    
    
 # Define the architecture by modifying resnet.
 # Original code is here
 # https://github.com/pytorch/vision/blob/b2e95657cd5f389e3973212ba7ddbdcc751a7878/torchvision/models/resnet.py
class ResNet50(models.ResNet):
    def __init__(self, model, model_dir_name, num_classes=2, do_rand_model=False, pretrained=True, **kwargs):
        # Start with standard resnet101 defined here
        # https://github.com/pytorch/vision/blob/b2e95657cd5f389e3973212ba7ddbdcc751a7878/torchvision/models/resnet.py
        super().__init__(block=models.resnet.Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes, **kwargs)
        # if pretrained:
        #     state_dict = load_state_dict_from_url(models.resnet.model_urls['resnet50'], progress=True)
        #     self.load_state_dict(state_dict)
        
        if pretrained and do_rand_model==False:
            if torch.cuda.is_available(): #  ===> map_location=='gpu'
                # if isinstance(model, (nn.DataParallel)):
                #     self.module.load_state_dict(torch.load(model_dir_name))  #  + ".pth"  # your model will be loaded to multi-gpu
                # else:
                #     self.load_state_dict(torch.load(model_dir_name)) #  + ".pth"
                self.load_state_dict(torch.load(model_dir_name)) #  + ".pth"
            else:                        #  ===> map_location=='cpu'
                # model.load_state_dict(torch.load(model_dir, map_location=lambda storage, loc: storage)) # + ".pth"
                model.load_state_dict(torch.load(model_dir_name, map_location='cpu'))  #+ ".pth"

        elif pretrained and do_rand_model:
            state_dict = load_state_dict_from_url(models.resnet.model_urls['resnet50'], progress=True)
            self.load_state_dict(state_dict)

    # Reimplementing forward pass.
    # Replacing the following code
    # https://github.com/pytorch/vision/blob/b2e95657cd5f389e3973212ba7ddbdcc751a7878/torchvision/models/resnet.py#L197-L213
    def _forward_impl(self, x):
        # Standard forward for resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Notice there is no forward pass through the original classifier.
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # basically the  x = self.fc(x) has been commented to be able to extract features (not classification outputs)

        return x
         
    
    