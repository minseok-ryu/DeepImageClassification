#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 18:40:12 2021

@author: homai
"""

import os
import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import warnings; warnings.filterwarnings("ignore")

from src.data_utils.get_custom_datasetloader import*
from src.prediction.classification import*
from src.plottings.plotting import*

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from torchvision.models import resnet50
from torchvision.models import googlenet
from torchvision.models import densenet121
from torchvision.models import vgg11
from torchvision.models import alexnet

from src.data_utils.CIFAR_to_png import get_annotation_file
from src.prediction.CNN_CIFAR import cnn


if __name__ == "__main__":
    # ------------------------ initializing paramters
    # root_dir = './data/Kaggle_fundus/'
    # classes_list = ['glaucoma', 'non-glaucoma']

    # ---------------------- get annotations file for all classes in a single .csv file
    # annotations = get_annotation_file(classes_list, base_dir+'Kaggle_fundus/')
    # annotations.to_csv(base_dir+'Kaggle_fundus/annotations.csv', sep=',', index=False)


    # ------------------------ initializing paramters
    data_dir = './data/cifar10/'
    saving_dir = './models/CIFAR10/'
    

    mean = [0.4914, 0.4822, 0.4465]
    std  = [0.2023, 0.1994, 0.2010]
    augnment = True
    
    batch_size_train = 128
    batch_size_valid = 400
    val_test_Ratio = 0.2
    nworkers = 4
    
    lr = 0.0001; nEpoch=100
    mntum=0.9; weight_decay= 5e-4
    nstopping_criteria = 8
    valid_acc_max = 0
    
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    # ------------------------ reading annotation file
    annotations = pd.read_csv(data_dir+'annotations.csv')
    annotations = annotations.sample(frac=1).reset_index(drop=True)
    
    # ------------------------ setting transfomations on train/val/test sets
    train_loader, valid_loader, test_loader= get_dataloaders(annotations,data_dir,augnment,val_test_Ratio,batch_size_train,batch_size_valid,nworkers)

    # ------------------------ print parameters
    n_batch = len(train_loader); n_batch_val = len(valid_loader)
    print('='*75); print('| '+'#Epochs='+str(nEpoch)+', lr='+str(lr)+', tr batch size='+str(batch_size_train)+', number bach='+str(n_batch)+', momentum='+str(mntum)+' |'); print('='*75)

    # ----------------------- initialize the model
    net = cnn()

    # net = vgg11(pretrained=True)
    # net.classifier[6] = nn.Sequential(nn.Linear(in_features=4096, out_features=10, bias=True)) #vgg11
    print(net)

    #net = resnet50(pretrained=True)
    #net.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=10))
     
    #------------------------ set up saving paths
    saving_path_modl = './models/CIFAR10_'+net.__class__.__name__+'_B'+str(batch_size_train)+'_lr'+str(lr*100)+'_w_L2'
    saving_path_res = saving_path_modl.replace('models','results')    

    # ----------------------- train the model
    results = train_validation_phase(net,batch_size_train,lr,mntum,weight_decay,nEpoch,n_batch,n_batch_val, \
                                     saving_path_modl,valid_acc_max,nstopping_criteria,train_loader,valid_loader)

    # ---------------------- testing the model
    net = cnn()
    net = nn.DataParallel(net.cuda())
    net.load_state_dict(torch.load(saving_path_modl, map_location={'cuda:0':'cuda:1'})) # map_location={'cuda:1':'cuda:0'}

    test_acc = test_phase(net,saving_path_modl,lr,test_loader)
    results.loc[results.index[0],'test_acc'] = test_acc.item()
        
    results.to_csv(saving_path_res+'_res.csv', sep=',', index=False)
    # ----------------------- plotting results (loss, accuracy)
    ntick = 5 
    res_plotter(results,ntick,net.__class__.__name__,nEpoch,lr,batch_size_train,saving_path_res)








