#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 15:54:25 2021

@author: homai
"""

import os
import copy
import glob
import argparse
import itertools
import numpy as np
import pandas as pd
from pytz import timezone
from datetime import datetime
import matplotlib.pyplot as plt 
from tzlocal import get_localzone
from os import listdir,mkdir,rmdir
from os.path import join,isdir,isfile
import warnings; warnings.filterwarnings("ignore")

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from architecture import networks
from prediction.train import*
from prediction.test import*
from architecture.CNN import*
from architecture.networks import*
from plottings.plotting import*
from data_utils.dataspliter import*
from func_utils.get_annotation import*
# from func_utils.performance_funcs import*
# from func_utils.hyperparameter_search import*
from data_utils.get_custom_datasetloader import*
from TSNE.visualize_features import*

from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score



def getOutFileName(net, argpars, net_number):
    now_CDT = datetime.now(timezone('America/Chicago'))
    current_time = now_CDT.strftime("%y-%m-%d_%H-%M")
    fileName = current_time+'_'+net.__class__.__name__+str(net_number)+'_lr'+str(argpars.lr)+'_B'+str(argpars.batch_size_train)
    return fileName

def str_to_class(str):
    return getattr(sys.modules[__name__], str)

def main_func(args):
    print('Current directory: ', os.getcwd())

    # Reading command line arguments into parser.
    parser = argparse.ArgumentParser(description = "Glaucoma prediction via classification.")

    # Filepaths
    ## os.system('mount_smbfs //hrashi4:H19_O94_M19_ay@10.157.80.76/aio /Users/homai/Desktop/mntpoint')
    # parser.add_argument("--pAnnot", dest="annot_dir", type=str, default='/Volumes/homa/Homa/Glaucoma_prediction/Real_World_Data_paper/GL_classification/IODA_IODA/dataset/') # '/Users/homai/Desktop/mntpoint/Datasets/FundusDiseases/Resized/'
    # parser.add_argument("--pSaving_model", dest="save_dir_modl", type=str, default='/Volumes/homa/Homa/Glaucoma_prediction/Real_World_Data_paper/GL_classification/IODA_IODA/prediction/models/') #'/Users/homai/Desktop/mntpoint/Projects/homa/Glaucoma_prediction/entire_images/models/7.grayScale/'
    # parser.add_argument("--pSaving_res", dest="save_dir_res", type=str, default='/Volumes/homa/Homa/Glaucoma_prediction/Real_World_Data_paper/GL_classification/IODA_IODA/prediction/results/') #'/Users/homai/Desktop/mntpoint/Projects/homa/Glaucoma_prediction/entire_images/results/7.grayScale/'
    # parser.add_argument("--ptsne_dir", dest="tsne_dir", type=str, default='/Volumes/homa/Homa/Glaucoma_prediction/Real_World_Data_paper/RW_data_inventory/tsne/original_1/RWD/') 
    
    parser.add_argument("--pAnnot", dest="annot_dir", type=str, default='../../saveDir/dataset/') #  '../../data/'   #  '../../train_test_val_dirs' 
    parser.add_argument("--pSaving_model", dest="save_dir_modl", type=str, default='../../saveDir/prediction/models/') 
    parser.add_argument("--pSaving_res", dest="save_dir_res", type=str, default='../../saveDir/prediction/results/') 
    # parser.add_argument("--ptsne_dir", dest="tsne_dir", type=str, default='../../tsne/') 
    
    parser.add_argument("--lr", dest="lr", type=np.float32, default=1e-4) #
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=2000) #
    parser.add_argument("--early_stopping", dest="early_stopping", type=int, default=500)
    parser.add_argument("--image_resize", dest="img_size", type=int, default=128)
    
    parser.add_argument("--bs_train", dest="batch_size_train", type=int, default=32)
    parser.add_argument("--bs_valid", dest="batch_size_valid", type=int, default=200)
    parser.add_argument("--bs_test", dest="batch_size_test", type=int, default=300)  
    
    parser.add_argument("--apply_CLAHE", dest="CLAHE", action='store_true')
    parser.add_argument("--apply_grayScale", dest="grayScale", action='store_true') 
    parser.add_argument("--apply_gaussianBlur", dest="gaussianBlur", action='store_true')
    
    parser.add_argument("--net_archit", dest="net_archit", type=str, default='resnet50')  
    parser.add_argument("--augmnt", dest="augnment", action='store_true') 
    parser.add_argument("--weight_decay", dest="weight_decay", type=np.float32, default=1e-4) 

    parser.add_argument("--validation_ratio", dest="val_Ratio", type=np.float32, default=0.1)   
    parser.add_argument("--test_ratio", dest="test_Ratio", type=np.float32, default=0.18)    
    
    # -------------- If you are using "repeat_train_data", ENSURE to USE augmentation
    parser.add_argument("--repeat_train_batch", dest="repeat_train_data", type=int, default=3)
    
    parser.add_argument("--phase", dest="phase", type=str, default='img_quality')   
    parser.add_argument("--nworkers", dest="nworkers", type=int, default=4)   
    parser.add_argument("--ntick", dest="ntick", type=int, default=4)   
    
    num_classes=2
    in_channels =3 # because I am using RGB images
    
    net_number = 50 # ResNet50
    valid_acc_max = 0
    
    apply_CLAHE = False
    apply_grayScale = False
    
    # Creating Parser Object
    opts = parser.parse_args(args[1:])
    print('Saving directory = ' , opts.save_dir_res, '!')
    
  # # ------------------------ split data into train/val/test sets
    # train_annotations, val_annotations, test_annotations = Data_splitter(opts)
    # save_resized_DataInFile(train_annotations, opts.annot_dir+'train', 512)
    # save_resized_DataInFile(val_annotations, opts.annot_dir+'val', 512)
    # save_resized_DataInFile(test_annotations, opts.annot_dir+'test', 512)

    # train_annotations = pd.read_csv(opts.annot_dir + 'train_annotations.csv')
    # val_annotations = pd.read_csv(opts.annot_dir + 'val_annotations.csv')
    # test_annotations = pd.read_csv(opts.annot_dir + 'test_annotations.csv')
    
    # save_org_DataInFile(train_annotations, opts.annot_dir+'train')
    # save_org_DataInFile(val_annotations, opts.annot_dir+'val')
    # save_org_DataInFile(test_annotations, opts.annot_dir+'test')

    # sys.exit()    
  # # ------------------------ Loading dataset: get dataloader object
  
    # dataloaders = get_dataloaders_test_needFolders(opts)  
    if opts.phase =='test':
        dataloaders = get_dataloaders_test_needFolders(opts)
        # dataloaders = get_dataloaders_test(opts)
    else:
        dataloaders= get_dataloaders_needFolders(opts)
        # dataloaders= get_dataloaders(opts)
    
    
    if opts.grayScale:
        in_channels = 1
    print('in_channels: ================== ', in_channels)    

    # ------------------------ init models
    net_archit = str_to_class(opts.net_archit)
    net = net_archit(in_channels=in_channels, out_features=num_classes); #print(net)
  
    # net = networks.densenet121(in_channels=in_channels, out_features= num_classes); print(net)

    # ----------------------- get output file name
    outFileName = getOutFileName(net,opts,net_number);   print(outFileName)
    
    # ----------------------- get metrics      
    metrics = {'accuracy_py': accuracy_score, 'sensitivity':recall_score, 'precision':precision_score, 'f1_score': f1_score, 'auroc': roc_auc_score} #  'auroc': roc_auc_score
    
    # ----------------------- loss function and optimizer
    # criterion = torch.nn.BCELoss()
    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(net.parameters(), lr=opts.lr, betas=(0.9, 0.999), eps=1e-07, weight_decay=opts.weight_decay, amsgrad=False)
    
    if opts.phase == 'train':
        opts = parser.parse_args(args[1:])
        train_validation_phase(dataloaders, net, outFileName, criterion, optimizer, opts, False, valid_acc_max, metrics=metrics, plot=True, write_csv=True)


    elif opts.phase == 'continue_training':
        opts = parser.parse_args(args[1:])
        
        fileName = input("Enter the file name of the model that you want to continue training on: ")
        results0 = pd.read_csv(opts.save_dir_res+fileName+'.csv')
        valid_acc_max = np.max(results0['val_accuracies'])

        train_validation_phase(dataloaders, net, fileName, criterion, optimizer, opts, True, valid_acc_max, metrics=metrics, plot=True, write_csv=True)
    
    
    elif opts.phase == 'hp_search': # hyperparameter_search
        opts = parser.parse_args(args[1:])
        
        learning_rate = 0.00001
        max_acc = testing_my_code_func(dataloaders, net, outFileName, opts, False, valid_acc_max, learning_rate, metrics=metrics,  plot=True, write_csv=True)
        
        # results.to_csv(opts.save_dir_res+outFileName+'_hp.csv', sep=',', index=False)
        # loss_val = np.min(results['val_losses'])
        optimal_lr = HP_tunning_v2(max_acc)
        print('optimal lr = ', optimal_lr)


    elif opts.phase == 'test':
       opts = parser.parse_args(args[1:])
       fileName = input("Enter the file name of the model that you want to test the model on: ")
    
       net_archit = str_to_class(opts.net_archit)
       net = net_archit(in_channels=in_channels, out_features=num_classes); print(net)
  
       # net = resNet50(250, num_classes, pretrained=True, changing=False)
       # net = networks.densenet121(in_channels=in_channels, out_features=num_classes)
       test_phase(opts, dataloaders, net, opts.save_dir_modl+fileName, fileName, criterion, metrics=metrics, write_csv=True)


    elif opts.phase == 'plot':
       opts = parser.parse_args(args[1:])
       fileName = input("Enter the file name of the model that you want to plot: ") 
       
       results = pd.read_csv(opts.save_dir_res+fileName+'.csv')
       loss_plotter(results, net.__class__.__name__, opts, fileName)
       metric_plot(results, net.__class__.__name__, opts, fileName, metrics)

       get_conMx_roc(net, num_classes, dataloaders,  'test', opts.save_dir_modl+fileName, opts.save_dir_res+fileName)


    elif opts.phase == 'TSNE':
        opts = parser.parse_args(args[1:])
        print(opts)
        fileName = input("Enter the file name of the model that you want to test the model on: ")
    
        net_archit = str_to_class(opts.net_archit)
        net = net_archit(in_channels=in_channels, out_features=num_classes); print(net)
       
        TSNE_visualization(opts, fileName, dataloaders['test'], net)
         
    
    elif opts.phase == 'img_quality':
        opts = parser.parse_args(args[1:])
        print(opts)
        
        net = models.resnet50(pretrained = True)
        
        dataloaders = get_dataloaders_inference(opts, opts.tsne_dir)  
        
        TSNE_visualization_data_discovery(opts, dataloaders['infer'], net)
    
    
    
    
if __name__ == "__main__":
    main_func(sys.argv)  
    
    
    
    
    
    
    