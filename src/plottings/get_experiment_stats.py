#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 15:45:00 2021

@author: homai
"""

import os
import sys
import cv2
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 



filename = '23-06-14_21-48_DataParallel50_lr0.003_B32_v2.csv'
res_dir = '/Volumes/Homa/Homa/Glaucoma_prediction/1.RWD_paper/GL_classification/PublicIODA/prediction/results/'

print(os.listdir())

results = pd.read_csv(res_dir + filename) 

df = pd.DataFrame()

df.loc[0, 'Model_name'] = filename.replace('.csv', '')

df.loc[0, 'min train_loss'] = np.min(results['train_losses'])
df.loc[0, 'min val_loss'] = np.min(results['val_losses'])
if 'test_loss' in results.columns:
    df.loc[0, 'test_loss'] = results.loc[0,'test_losses']
else:
    df.loc[0, 'test_loss'] = np.nan

df.loc[0, 'max train_accuracy'] = np.max(results['train_accuracies'])
df.loc[0, 'max val_accuracy'] = results['max_val_acc'].iloc[-1]
if 'test_accuracies' in results.columns:
    df.loc[0, 'test_accuracy'] = results.loc[0,'test_accuracies']
else:
    df.loc[0, 'test_accuracy'] = np.nan


df.loc[0, 'max train_sensitivity'] = np.max(results['train_sensitivity'])
df.loc[0, 'max val_sensitivity'] = np.max(results['val_sensitivity'])
if 'test_sensitivity' in results.columns:
    df.loc[0, 'test_sensitivity'] = results.loc[0,'test_sensitivity']
else:
    df.loc[0, 'test_sensitivity'] = np.nan 


df.loc[0, 'max train_precision'] = np.max(results['train_precision'])
df.loc[0, 'max val_precision'] = np.max(results['val_precision'])
if 'test_precision' in results.columns:
    df.loc[0, 'test_precision'] = results.loc[0,'test_precision']
else:
    df.loc[0, 'test_precision'] = np.nan 


df.loc[0, 'max train_f1_score'] = np.max(results['train_f1_score'])
df.loc[0, 'max val_f1_score'] = np.max(results['val_f1_score'])
if 'test_f1_score' in results.columns:
    df.loc[0, 'test_f1_score'] = results.loc[0,'test_f1_score']
else:
    df.loc[0, 'test_f1_score'] = np.nan


df.loc[0, 'nEpoch'] = len(results)
df.loc[0, 'early_stopping']= np.unique(results['early_stopping'])
df.loc[0, 'batch_size_train'] = np.unique(results['batch_size_train'])
df.loc[0, 'lr']= np.unique(results['lr'])
df.loc[0, 'weight_decay'] = np.unique(results['weight_decay'])
df.loc[0, 'augment'] = np.unique(results['augment'])
df.loc[0, 'img_size'] = np.unique(results['img_size'])
df.loc[0, 'CLAHE'] = np.unique(results['CLAHE'])
df.loc[0, 'grayscale'] = np.unique(results['grayscale'])
df.loc[0, 'GaussianBlur'] = np.unique(results['GaussianBlur'])
df.loc[0, 'loss_func'] = np.unique(results['loss_func'])
df.loc[0, 'opt_solver'] = np.unique(results['opt_solver'])


if 'net_archit' in results.columns:
    df.loc[0, 'net_archit'] = np.unique(results['net_archit'])

if 'repeat_train_data' in results.columns:
    df.loc[0, 'repeat_train_data'] = np.unique(results['repeat_train_data'])



df.to_csv(res_dir.replace('results','exp') + filename, sep=',', index=False)











