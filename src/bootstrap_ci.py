#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 13:05:46 2022

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
import warnings; warnings.filterwarnings("ignore")


path = '/Volumes/homa/Homa/Glaucoma_prediction/Real_World_Data_paper/GL_classification/Public_IODA/prediction/results/'
res = pd.read_csv(path+'21-11-28_19-20_ResNet50_lr1e-04_B32_for_CI_calculation2.csv') #load csv file here
# the data in the csv file must be in the following format: 
# Columns: Score (with the confidence score of your prediction) & Prediction (1 = correct, 0 = wrong)


df = pd.DataFrame()

df['Score']       = res['prob_col2'] # prediction
df['target']  = res['target']


score_temp = df.iloc[:, 0]
score = score_temp.to_numpy()

y_temp = df.iloc[:, 1]
y = y_temp.to_numpy()
# print(y)

aupr_list = list()
auroc_list = list()

#bootstrap resamples for CI calc
n_bootsraps = 1000
for i in range(n_bootsraps):
    # idx = np.random.choice(np.arange(len(y)), int(len(y)/100), replace=False)
    idx = np.random.choice(np.arange(len(y)), int(len(y) - 10), replace=False)
    score_sample = score[idx]  # prediction
    y_sample = y[idx]          # target
    precision, recall, thresholds = precision_recall_curve(
    y_sample, score_sample)
    AUPR = auc(recall, precision)
    #print('AUPR: ', AUPR)
    fpr, tpr, thresholds = metrics.roc_curve(y_sample, score_sample)
    AUROC = auc(fpr, tpr)
    #print('AUROC: ', AUROC)
    aupr_list.append(AUPR)
    auroc_list.append(AUROC)

alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(aupr_list, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(aupr_list, p))
print('%.1f confidence interval for AUPR %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
ci_PR = (upper-lower)
print('Confidence interval for PR ', ci_PR)


alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(auroc_list, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(auroc_list, p))
print('%.1f confidence interval for AUROC %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
ci_ROC = (upper-lower)
print('Confidence interval for AUC   ', ci_ROC)
print('Mean AUC   ', np.mean(auroc_list))

##========================= plot precision-recall curve for all samples
sheet_name = " "

precision, recall, thresholds = precision_recall_curve(
    y, score)
precision = np.insert(precision,0, 0, axis=0)
recall = np.insert(recall, 0, 1, axis=0)
AUPR = auc(recall, precision)
print('AUPR: ', AUPR)

fig = plt.figure()
plt.plot(recall, precision)#, marker='-')
plt.xlim([0, 1.01])
plt.ylim([0, 1.1])
plt.title("Test data - Precision-Recall curve " + sheet_name)
plt.ylabel('Precision')
plt.xlabel('Recall')
#add confidence
plt.fill_between(recall, (precision-ci_PR), (precision+ci_PR), color='b', alpha=.1)
plt.show()


##========================= plot roc curve  for all samples
fpr, tpr, thresholds = metrics.roc_curve(y, score)


AUPR = auc(fpr, tpr)
print('AUROC: ', AUPR)

log = []
fig = plt.figure(figsize=(4,4), dpi=300)
a = plt.plot(fpr, tpr, color='blue')#, marker='.')
plt.xlim([-0.01, 1])
plt.ylim([0, 1.1])
# plt.title("Test data - ROC curve" + sheet_name)
plt.ylabel('True Positive Rate', fontsize=13)
plt.xlabel('False Positive Rate', fontsize=13)
#add confidence
plt.fill_between(fpr, (tpr-ci_ROC), (tpr+ci_ROC), color='blue', alpha=.1)
b = plt.plot([0,1],[0,1], color='red')

log.append('ROC curve'); log.append('Random model')
plt.legend(log, loc='upper left')
plt.show()






