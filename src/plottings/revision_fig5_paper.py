#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 11:35:10 2023

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



experiment_names = ['IODA_IODA', 'Public_IODA', 'PublicIODA_IODA']

saving_dir = '/Volumes/homa/Homa/Glaucoma_prediction/Real_World_Data_paper/GL_classification/'
save_fig = True

log = []; idx = 0
fig, axs = plt.subplots(1,1, figsize=(7,6))
color = ['green', 'blue', 'darkorange', 'red', 'purple']

for index, iexp in enumerate(experiment_names):
    if iexp == 'IODA_IODA':
        path = '/Volumes/homa/Homa/Glaucoma_prediction/Real_World_Data_paper/GL_classification/'+iexp+'/prediction/results/'        
        model_name = '21-11-30_20-04_ResNet50_lr1e-04_B32_for_CI_calculation.csv'
    elif iexp == 'Public_IODA':
        path = '/Volumes/homa/Homa/Glaucoma_prediction/Real_World_Data_paper/GL_classification/'+iexp+'/prediction/results/'        
        model_name = '21-11-28_19-20_ResNet50_lr1e-04_B32_for_CI_calculation.csv'
        
    elif iexp == 'PublicIODA_IODA':
        path = '/Volumes/homa/Homa/Glaucoma_prediction/Real_World_Data_paper/GL_classification/PublicIODA/prediction/results/'        
        path = os.path.join(path, 'test_on_IODA')
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
    axs.plot(fpr, tpr, color=color[index], linewidth=1.5)#, marker='.')
    axs.set_xlim([-0.01, 1])
    axs.set_ylim([0, 1.1])
    
    #add confidence
    axs.fill_between(fpr, (tpr-ci_ROC), (tpr+ci_ROC), color=color[index], alpha=.1, linewidth=1.5)
    axs.plot([0,1],[0,1], color='red', linewidth=1.5)

    axs.scatter(0.05,0.776, color='purple', marker = 'x', s=120)



# Set common labels
fig.text(0.51, 0.03 , 'False Positive Rate', ha='center', va='center', fontsize=15) # # xlabel
fig.text(0.05, 0.5, 'True Positive Rate', ha='center', va='center', rotation='vertical', fontsize=15) # ylabel

log.append('RWD model'); log.append('Public model'); log.append('RWD+Public model'); log.append('Random model'); log.append('Physician performance')

# Define the line styles and marker
line_styles = ['-'] * 4
line_styles.append('')  # Empty string for the last item
marker_styles = [''] * 4
marker_styles.append('x')  # 'x' marker for the last item

# Create a custom legend
handles = [plt.Line2D([], [], marker=mstyle, color=icolor, linestyle=lstyle, linewidth=1, markersize=11 if mstyle == 'x' else 6) for lstyle, mstyle, icolor in zip(line_styles, marker_styles, color)]
legend = plt.legend(handles, log, fontsize=13, bbox_to_anchor=(0.45, 0.35), ncol=1)

plt.show()




plt.subplots_adjust(wspace=0.2, hspace=0.1)
if save_fig:
    plt.savefig(saving_dir+'ROC_curves_physicians_RWD_revision.png', bbox_inches='tight', pad_inches = 0.1, dpi=350)







