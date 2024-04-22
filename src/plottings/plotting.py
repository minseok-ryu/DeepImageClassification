#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 07:45:58 2021

@author: homai
"""


import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import warnings; warnings.filterwarnings("ignore")

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim




def loss_plotter(results, net, argpars, outFileName):
    is_test_done = False
    if 'test_acc' in results.columns:
        is_test_done = True
    
    # ----- train
    fig=plt.figure(); 
    plt.scatter(np.arange(len(results)), results['train_losses'], c='red') 
    plt.scatter(np.arange(len(results)), results['val_losses'], c='blue')
    plt.xticks(np.arange(len(results))); plt.legend(['train loss', 'validation loss']); 
    plt.plot(results['train_losses'], color='k', alpha=0.6); plt.plot(results['val_losses'], color='k', alpha=0.6)
    xx=[i for i in range(0,len(results), argpars.ntick)]; plt.xticks(np.round(xx,3), rotation = 90)
    plt.xlabel('Epoch'); plt.ylabel('Cross entropy loss');
    # plt.title(net+': img W/H='+str(argpars.img_size)+', lr='+str(argpars.lr)+', |batch|='+str(argpars.batch_size_train))    
    if is_test_done:
        plt.text(np.max(xx), results.val_losses.min()+0.1, 'Test loss = '+str(np.round(results.loc[0,'test_loss'],3)),
             verticalalignment='bottom', horizontalalignment='right',color='k', fontweight= 'bold', fontsize=12)
    # plt.grid(color='gray', alpha=0.3); 
    plt.savefig(argpars.save_dir_res+outFileName+'_loss.pdf')
    plt.show()

    # ----- validation    
    fig=plt.figure(); 
    plt.scatter(np.arange(len(results)),results['train_accuracies'], c='orange'); 
    plt.scatter(np.arange(len(results)),results['val_accuracies'], c='green'); 
    plt.legend(['train accuracy', 'validation accuracy']); plt.xticks(np.arange(len(results))) 
    plt.plot(results['train_accuracies'], color='k', alpha=0.6); plt.plot(results['val_accuracies'], color='k', alpha=0.6)
    xx=[i for i in range(0,len(results), argpars.ntick)]; plt.xticks(xx,rotation = 90)
    plt.xlabel('Epoch'); plt.ylabel('Classification accuracy'); 
    plt.title(net+': img W/H='+str(argpars.img_size)+', lr='+str(argpars.lr)+', |batch|='+str(argpars.batch_size_train))
    if is_test_done:
        plt.text(np.max(xx), results.val_accuracies.max(), 'Test accuracy = '+str(np.round(results.loc[0,'test_acc'],3)),
                 verticalalignment='bottom', horizontalalignment='right',color='k', fontweight= 'bold', fontsize=12)
    # plt.grid(color='gray', alpha=0.3); 
    plt.savefig(argpars.save_dir_res+outFileName+'_accu.pdf')

    plt.show()
    


def loss_plotter_here(results, net, argpars, outFileName, save_dir_res):
    is_test_done = False
    if 'test_acc' in results.columns:
        is_test_done = True
    
    # ----- train
    fig=plt.figure(); 
    plt.scatter(np.arange(len(results)), results['train_losses'], c='red') 
    plt.scatter(np.arange(len(results)), results['val_losses'], c='blue')
    plt.xticks(np.arange(len(results))); plt.legend(['train loss', 'validation loss']); 
    plt.plot(results['train_losses'], color='k', alpha=0.6); plt.plot(results['val_losses'], color='k', alpha=0.6)
    xx=[i for i in range(0,len(results), 100)]; plt.xticks(np.round(xx,3), rotation = 90)
    plt.xlabel('Epoch'); plt.ylabel('Cross entropy loss');
    if is_test_done:
        plt.text(np.max(xx), results.val_losses.min()+0.1, 'Test loss = '+str(np.round(results.loc[0,'test_loss'],3)),
             verticalalignment='bottom', horizontalalignment='right',color='k', fontweight= 'bold', fontsize=12)
    # plt.grid(color='gray', alpha=0.3); 
    plt.savefig(save_dir_res+outFileName+'_loss.pdf')
    plt.show()

    # ----- validation    
    fig=plt.figure(); 
    plt.scatter(np.arange(len(results)),results['train_accuracies'], c='orange'); 
    plt.scatter(np.arange(len(results)),results['val_accuracies'], c='green'); 
    plt.legend(['train accuracy', 'validation accuracy']); plt.xticks(np.arange(len(results))) 
    plt.plot(results['train_accuracies'], color='k', alpha=0.6); plt.plot(results['val_accuracies'], color='k', alpha=0.6)
    xx=[i for i in range(0,len(results), 100)]; plt.xticks(xx,rotation = 90)
    plt.xlabel('Epoch'); plt.ylabel('Classification accuracy'); 
    if is_test_done:
        plt.text(np.max(xx), results.val_accuracies.max(), 'Test accuracy = '+str(np.round(results.loc[0,'test_acc'],3)),
                 verticalalignment='bottom', horizontalalignment='right',color='k', fontweight= 'bold', fontsize=12)
    # plt.grid(color='gray', alpha=0.3); 
    plt.savefig(save_dir_res+outFileName+'_accu.pdf')

    plt.show()


def metric_plot(results, net_name, argpars, outFileName, metrics):
    is_test_done = False
    if 'test_loss' in results.columns:
        is_test_done = True
    
    field_names = ['epoch'] + [f'train_{m}' for m in metrics.keys()] + [f'val_{m}' for m in metrics.keys()]
    
    for name, metric in metrics.items():
        fig = plt.figure()     
        
        plt.scatter(np.arange(len(results)), results['train_'+name], c='red') 
        plt.scatter(np.arange(len(results)), results['val_'+name], c='blue')
        plt.xticks(np.arange(len(results))); plt.legend(['train '+name, 'validation '+name], fontsize=12); 
        plt.plot(results['train_'+name], color='k', alpha=0.6); plt.plot(results['val_'+name], color='k', alpha=0.6)
        
        xx=[i for i in range(0,len(results), argpars.ntick)]; plt.xticks(np.round(xx,3), rotation = 90)
        
        plt.xlabel('Epoch'); plt.ylabel(name, fontsize=14);
        plt.title(net_name+': img W/H='+str(argpars.img_size)+', lr='+str(argpars.lr)+', |batch|='+str(argpars.batch_size_train))    
        if is_test_done:
            plt.text(np.max(xx), np.min(results['val_'+name]), 'Test '+name+' = '+str(np.round(results.loc[0,'test_'+name],3)),
                 verticalalignment='bottom', horizontalalignment='right',color='k', fontweight= 'bold', fontsize=12)
        
        # plt.grid(color='gray', alpha=0.3); 
        plt.savefig(argpars.save_dir_res+outFileName+'_'+name+'.png')
        plt.show()    
       
    
    
  
# filename = '23-06-14_16-52_DataParallel50_lr1e-04_B16_v2.csv'
# res_dir = '/Volumes/Homa/Homa/Glaucoma_prediction/1.RWD_paper/GL_classification/PublicIODA/prediction/results/'
# net = 'Resnet50'

# results = pd.read_csv(res_dir+filename)
# loss_plotter_here(results, net, '', filename, res_dir)


  