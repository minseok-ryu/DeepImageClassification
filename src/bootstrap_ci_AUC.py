#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 10:57:05 2022

@author: homai
"""

import os
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



def mean_confidence_interval(data, confidence=0.9):
    #--------------------------------------------------------------------------
    # Computes confidence interval around mean
    #--------------------------------------------------------------------------
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sem(a)   # sem : Computes standard error of the mean.
    h = se * t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h, se


def annotateOnFigure_v2(string, colr, x, y, fSiz, ax):
    ax.annotate(string, 
                 color=colr, xy=(x, y), xycoords='axes fraction', xytext=(-10, 10), 
                 textcoords='offset pixels', horizontalalignment='center',
                 verticalalignment='center', fontsize=fSiz,
                 bbox=dict(facecolor='none', edgecolor='k', pad=3, linewidth =0.5, alpha=0.5) # boxstyle='round,pad=1',
                 )  




experiment_names = ['IODA_IODA', 'Public_IODA', 'PublicIODA_IODA', 'IODA_Public', 'Public_Public', 'PublicIODA_Public']
label            = ['RWD->RWD',  'Public->RWD', '(RWD+Public)->RWD', 'RWD->Public', 'Public->Public','(RWD+Public)->Public']

saving_dir = '/Volumes/homa/Homa/Glaucoma_prediction/Real_World_Data_paper/GL_classification/'
save_fig = True

log = []
fig, axs = plt.subplots(2, 3, figsize=(11,6), sharey=True, sharex=True); axs = axs.flatten()   # , constrained_layout=False 

for iax, iexp in enumerate(experiment_names):
    if iexp == 'IODA_IODA' or iexp == 'IODA_Public':
        path = '/Volumes/homa/Homa/Glaucoma_prediction/Real_World_Data_paper/GL_classification/'+iexp+'/prediction/results/'        
        model_name = '21-11-30_20-04_ResNet50_lr1e-04_B32_for_CI_calculation.csv'
    elif iexp == 'Public_Public' or iexp == 'Public_IODA':
        path = '/Volumes/homa/Homa/Glaucoma_prediction/Real_World_Data_paper/GL_classification/'+iexp+'/prediction/results/'        
        model_name = '21-11-28_19-20_ResNet50_lr1e-04_B32_for_CI_calculation.csv'
        
    elif iexp == 'PublicIODA_IODA':
        path = '/Volumes/homa/Homa/Glaucoma_prediction/Real_World_Data_paper/GL_classification/PublicIODA/prediction/results/'        
        path = os.path.join(path, 'test_on_IODA')
        model_name = '23-06-20_11-36_DataParallel50_lr1e-04_B128.pth_for_CI_calculation_cleanTest.csv'
        
    elif iexp == 'PublicIODA_Public':
        path = '/Volumes/homa/Homa/Glaucoma_prediction/Real_World_Data_paper/GL_classification/PublicIODA/prediction/results/'        
        path = os.path.join(path, 'test_on_Public')
        model_name = '23-06-20_11-36_DataParallel50_lr1e-04_B128.pth_for_CI_calculation_cleanTest.csv'
        
        
    res = pd.read_csv(os.path.join(path, model_name)) #load csv file here

    df = pd.DataFrame()
    
    df['Score']   = res['prob_col2'] # prediction
    df['target']  = res['target']
    
    score_temp = df.iloc[:, 0]
    score = score_temp.to_numpy()
    
    y_temp = df.iloc[:, 1]
    y = y_temp.to_numpy()
    
    auroc_list = list()
    #bootstrap resamples for CI calc
    n_bootsraps = 1000
    for i in range(n_bootsraps):
        # idx = np.random.choice(np.arange(len(y)), int(len(y)/100), replace=False)
        idx = np.random.choice(np.arange(len(y)), int(len(y) - 10), replace=False)
        score_sample = score[idx]  # prediction
        y_sample = y[idx]          # target
        
        #print('AUPR: ', AUPR)
        fpr, tpr, thresholds = metrics.roc_curve(y_sample, score_sample)
        AUROC = auc(fpr, tpr)
        #print('AUROC: ', AUROC)
        auroc_list.append(AUROC)
    
    ##========================= calculate stats
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(auroc_list, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(auroc_list, p))
    ci_ROC = (upper-lower)
    
    fpr, tpr, thresholds = metrics.roc_curve(y, score)
    AUPR = auc(fpr, tpr)
    
    #========================== print CI using scipy calculation    
    mu, low_b, up_b, _ = mean_confidence_interval(auroc_list)

    
    ##========================= print stats of interest
    print('-'*50)
    # print(f'Scipy {100-alpha*100}% CI for accuracy = [{low_b*100:.2f}, {up_b*100:.2f}]')
    print(iexp, '%.1f confidence interval for AUROC %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
    print(iexp, 'Confidence interval for AUC   ', ci_ROC)
    print(iexp, 'Mean AUC   ', np.mean(auroc_list))
    print(iexp, 'AUROC: ', AUPR)
    print(iexp, '-'*50)
    ##========================= plot roc curve  for all samples
    a = axs[iax].plot(fpr, tpr, color='blue', linewidth=1.5)#, marker='.')
    axs[iax].set_xlim([-0.01, 1])
    axs[iax].set_ylim([0, 1.15])
    # plt.title("Test data - ROC curve" + sheet_name)

    # annotateOnFigure_v2(label[iax], 'k', 0.35, 0.91 , 10, axs[iax])
    
    #add confidence
    axs[iax].fill_between(fpr, (tpr-ci_ROC), (tpr+ci_ROC), color='blue', alpha=.1, linewidth=1.5)
    b = axs[iax].plot([0,1],[0,1], color='red', linewidth=1.5)


annotateOnFigure_v2(f'A: {label[0]}', 'k', 0.3, 0.91, 11, axs[0])
annotateOnFigure_v2(f'B: {label[1]}', 'k', 0.3 , 0.91, 11, axs[1])
annotateOnFigure_v2(f'C: {label[2]}', 'k', 0.42, 0.91, 11, axs[2])
annotateOnFigure_v2(f'D: {label[3]}', 'k', 0.3, 0.91, 11, axs[3])
annotateOnFigure_v2(f'E: {label[4]}', 'k', 0.3, 0.91, 11, axs[4])
annotateOnFigure_v2(f'F: {label[5]}', 'k', 0.42, 0.91, 11, axs[5])


# Set common labels
fig.text(0.51, 0.02 , 'False Positive Rate', ha='center', va='center', fontsize=15) # # xlabel
fig.text(0.07, 0.5, 'True Positive Rate', ha='center', va='center', rotation='vertical', fontsize=15) # ylabel

log.append('ROC curve'); log.append('Random model')
plt.legend(log, loc='upper left')
axs[iax].legend(log, bbox_to_anchor= (0.2, -0.4), ncol=2, borderaxespad=0, frameon=True, fontsize=15) # 
plt.show()


# Set the borders to a given color...
for ax in axs:
    # ax.tick_params(color='k', labelcolor='k')
    for spine in ax.spines.values():
        spine.set_edgecolor('grey')


# plt.subplots_adjust(wspace=0.2, hspace=0.1)
if save_fig:
    plt.savefig(saving_dir+'ROC_curves_glClassification_revision_v2.png', bbox_inches='tight', pad_inches = 0.1, dpi=350)







