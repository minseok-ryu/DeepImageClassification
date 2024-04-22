#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 19:39:24 2022

@author: homai
"""

import os
import torch
import random
import shutil
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_model(weights, model_dir):
    # make_dir(model_dir)
    # torch.save(weights, model_dir + ".pth")
    # saving
    if isinstance(weights, (nn.DataParallel)):
        torch.save(weights.module.state_dict(), model_dir+ ".pth")
    else:
        torch.save(weights.state_dict(), model_dir+ ".pth")


def load_model_weights(model, model_dir, map_location):
    print('Was the pretrained DeeplabV3+ResNet backbone trained using nn.DataParallel?', isinstance(model, nn.DataParallel), ':>')
    
    if map_location=='gpu':
        print("Let's use", torch.cuda.device_count(), "GPU!")
        print('Was my model trained using nn.DataParallel?', isinstance(model, nn.DataParallel), ':>')           
        # model.load_state_dict(torch.load(model_dir + ".pth")) # raises RuntimeError: Error(s) in loading state_dict for DataParallel:
        
        if isinstance(model, (nn.DataParallel)):
            model.module.load_state_dict(torch.load(model_dir))  #  + ".pth"  # your model will be loaded to multi-gpu model.
        else:
            model.load_state_dict(torch.load(model_dir)) #  + ".pth"
        
        #In a case when you already saved multi-gpu model parameters as .module.xxx and loading to a single-gpu model, then you should do:
        #model = model.module  # make it single-gpu
        
    elif map_location=='cpu':
        print('Loading model on CPU!')
        model.load_state_dict(torch.load(model_dir, map_location=lambda storage, loc: storage)) # + ".pth"
        # model.load_state_dict(torch.load(model_dir, map_location='cpu'))  #+ ".pth"
    return model



