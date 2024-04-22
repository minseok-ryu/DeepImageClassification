#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 09:20:31 2023

@author: homai
"""


import sys
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

import scipy.stats as st
from scipy.stats import sem, t, iqr
import warnings; warnings.filterwarnings("ignore")


def get_CI_networks(experiment_names):
    # the data in the csv file must be in the following format: 
    # Columns: Score (with the confidence score of your prediction) & Prediction (1 - correct, 0 - wrong)
    for iax, iexp in enumerate(experiment_names):
        if iexp == 'IODA_IODA' or iexp == 'IODA_Public':
            model_name = '21-11-30_20-04_ResNet50_lr1e-04_B32'
        elif iexp == 'Public_Public' or iexp == 'Public_IODA':
            model_name = '21-11-28_19-20_ResNet50_lr1e-04_B32'
            
        path = '/Volumes/homa/Homa/Glaucoma_prediction/Real_World_Data_paper/GL_classification/'+iexp+'/prediction/results/'
        res = pd.read_csv(path + model_name + '_for_CI_calculation.csv') #load csv file here

        df = pd.DataFrame()
        
        df['pred']   = res['prediction'] # prediction
        df['target']  = res['target']
        
        score_temp = df.iloc[:, 0]
        score = score_temp.to_numpy()
        
        y_temp = df.iloc[:, 1]
        y = y_temp.to_numpy()
        
        get_CI_acc(iexp, score, y)


def get_acc_CI_physicians(experiment_names):
    for iax, iexp in enumerate(experiment_names):
        path = '/Volumes/homa/Homa/Glaucoma_prediction/Real_World_Data_paper/GL_classification/Clinicians_label/'
        res = pd.read_csv(path + 'clinicians_labels_'+iexp+'.csv') #load csv file here
        
        df = pd.DataFrame()
        
        df['pred']   = res['doct_label'] # prediction
        df['target']  = res['target']
        
        score_temp = df.iloc[:, 0]
        score = score_temp.to_numpy()
        
        y_temp = df.iloc[:, 1]
        y = y_temp.to_numpy()
        
        get_CI_acc(iexp, score, y)
    
def get_CI_acc(iexp, score, y):
    acc_list = list()
    #bootstrap resamples for CI calc
    n_bootsraps = 1000
    for i in range(n_bootsraps):
        # idx = np.random.choice(np.arange(len(y)), int(len(y)/100), replace=False)
        idx = np.random.choice(np.arange(len(y)), int(len(y) - 10), replace=False)
        score_sample = score[idx]  # prediction
        y_sample = y[idx]          # target
        n_corrects = sum(1 for i in range(len(y_sample)) if score_sample[i]== y_sample[i])
        acc = (n_corrects/len(score_sample))
        
        acc_list.append(acc)
    
    ##========================= calculate stats
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(acc_list, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(acc_list, p))
    ci_ROC = (upper-lower)

    ##========================= print stats of interest
    print(f'Experiment: {iexp}')
    # print(f'Scipy {100-alpha*100}% CI for accuracy = [{low_b*100:.2f}, {up_b*100:.2f}]')
    print('%.1f confidence interval for accuracy %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
    print('Confidence interval for accuracy  ', ci_ROC)
    print('Mean accuracy  ', np.mean(acc_list))
    print('-'*50)
    
    
# =====================================================    
# expe_names = ['Public_Public', 'IODA_IODA', 'Public_IODA', 'IODA_Public']
# get_CI_networks(expe_names)

# =====================================================
exp_names = ['IODA_withGT', 'Public_withGT']
get_acc_CI_physicians(exp_names)
