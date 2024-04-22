#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 12:20:47 2021

@author: homai

import torch
y_true = torch.randint(0, 1, (20,))
y_pred = torch.randint(0, 1, (20,))
"""

import sys
import torch
import itertools
import numpy as np
import pandas as pd
import seaborn as sns; sns.set_style("whitegrid")

import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve

from func_utils import utils

from scipy import interp
import warnings; warnings.filterwarnings("ignore")


def get_conMx_roc(model, nb_classes, dataloaders, phase, modelDir, savngDir):
    # Initialize the prediction and label lists(tensors)
    predlist = torch.zeros(0,dtype=torch.long, device='cpu') # used in conf max
    lbllist  = torch.zeros(0,dtype=torch.long, device='cpu') # used in conf max
    
    outputs_list = torch.zeros(0,dtype=torch.long, device='cpu')  # used in plot_roc
    lbl_ohlist  = torch.zeros(0,dtype=torch.long, device='cpu')
    
    # model = _cuda_enabled(model, modelDir)
    if torch.cuda.is_available(): # and torch.cuda.device_count() > 1
        model = utils.load_model_weights(model, modelDir, map_location='gpu') 
    else:
        model = utils.load_model_weights(model, modelDir, map_location='cpu')
    
    model.eval()
    
    i = 0 
    with torch.no_grad():
        for inputs, targets in dataloaders[phase]:
            # inputs shape: N x C x W x H
            # label shape: 1 scalar
            
            i = i + 1
            # ------------------- moving data to gpu, if available
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()   
        
            outputs = model(inputs)
            
            prob = F.softmax(outputs)
            
            y_true_oh = torch.eye(nb_classes)[targets] # one-hot encoded the targets
            preds = torch.argmax(prob, dim=1)
    
            # Append batch prediction results
            predlist = torch.cat([predlist, preds.view(-1).cpu()])
            lbllist = torch.cat([lbllist, targets.view(-1).cpu()])

            outputs_list = torch.cat([outputs_list, prob.cpu()])     # 2 column matrix of probs of all batches
            lbl_ohlist  = torch.cat([lbl_ohlist, y_true_oh.cpu()])   # 2 column matrix of targets of all batches

            print(i, "/", len(dataloaders[phase]), " - training: input size", inputs.size(), ", output_size", outputs.size(), ", label size", targets.size()) 
            
    y_true = lbllist.numpy(); y_preds = predlist.numpy()
    # Confusion matrix
    perClass_accuracy = confusionMatrix(y_true, y_preds, savngDir)
    print(' \n ')
    
    # calculate and plot the ROC curve
    fpr, tpr, roc_auc = plot_roc(nb_classes, lbl_ohlist, outputs_list, perClass_accuracy, savngDir)
    
    average_precision = average_precision_score(lbl_ohlist, outputs_list)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    plot_precision_recall(nb_classes, lbl_ohlist, outputs_list, average_precision, savngDir)


def confusionMatrix(y_true, yhat_preds, savingDir): # yhat_preds: are 1,0 predictions
    # Confusion matrix
    conf_mat = confusion_matrix(y_true, yhat_preds)
    df_cm = pd.DataFrame(conf_mat, range(2), range(2))
    # plt.figure(figsize=(10,7))
    fig = plt.figure()   
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
    plt.ylabel('Ground truth'); plt.xlabel('Prediction')
    # plt.xticks(np.arange(2), ['non-glaucoma', 'glaucoma'])
    # plt.yticks(np.arange(2), ['non-glaucoma', 'glaucoma'], rotation=90)
    plt.tight_layout()
    plt.savefig(savingDir+'_confMX.pdf')
    print('conf_mat: \n ', conf_mat)
    
    # Per-class accuracy
    class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
    print('per-class accuracy [0, 1]:', class_accuracy); print('-'*20)
    
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_true, yhat_preds)
    print('Overall accuracy: %f' % accuracy); print('-'*20)
    
    # precision = Predicted Positive Value (PPV): tp / (tp + fp)
    precision = precision_score(y_true, yhat_preds)
    print('Precision: %f' % precision); print('-'*20)
    
    # recall = sensitivity: tp / (tp + fn)
    recall = recall_score(y_true, yhat_preds)
    print('Recall: %f' % recall); print('-'*20)
    
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_true, yhat_preds)
    print('F1 score: %f' % f1); print('-'*20)  
    
    # kappa
    kappa = cohen_kappa_score(y_true, yhat_preds)
    print('Cohens kappa: %f' % kappa)
    return class_accuracy

def RoC(n_classes, y_test, y_score): # y_score: predicted probablties/logits
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], thresholds = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # print(y_test)
    # print(y_score)    
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.numpy().ravel(), y_score.numpy().ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # macro
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return fpr, tpr, roc_auc
    

def plot_roc(n_classes, true_labls, pred_labls, per_class_accuracy, savingDir):
    fpr, tpr, roc_auc = RoC(n_classes, true_labls, pred_labls)
    
    # Plot all ROC curves
    plt.figure()
    if n_classes>2:
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC AUC (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)
    
        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC AUC (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))
        
    else:
        colors  = cycle(['deeppink'])
        for i, color in zip(range(n_classes-1), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC AUC = {1:0.2f})'
                           ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate: FP/(TN+FP)') # 1 - precision
    plt.ylabel('True Positive Rate: TP/(TP+FN)') # recall/sensitivity
    plt.title('ROC')
    plt.legend(loc="lower right")
    
    plt.text(0.4, 0.22, 'Per-class (0, 1) accuracy= ('+str(np.round(per_class_accuracy[0],1))+', '+str(np.round(per_class_accuracy[1],1))+')', fontsize=13)
    # plt.show()    
    plt.tight_layout()
    plt.savefig(savingDir+'_roc.pdf')
    return fpr, tpr, roc_auc 


def plot_precision_recall(n_classes, y_test, y_score, averag_precision, savingDir):
    plt.figure()
    # precision recall curve
    precision = dict()
    recall = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            y_score[:, i])
        plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))
        
    plt.xlabel("Recall: TP/(TP+FN)")  # recall/sensitivity
    plt.ylabel("Precision: TP/(TP+FP)")  
    plt.legend(loc="best")
    plt.title("Precision vs. recall curve")
    plt.text(0, 0.52, 'Average precision-recall score= '+str(np.round(averag_precision,2)), fontsize=13)
    # plt.show()
    plt.tight_layout()
    plt.savefig(savingDir+'_pr.pdf')
    return precision, recall

    
    


# def plot_confusion_matrix(cm, class_names, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
#     plt.figure(figsize=(10,10))
#     plot_confusion_matrix(cm, class_names)
    
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(class_names))
#     plt.xticks(tick_marks, class_names, rotation=45)
#     plt.yticks(tick_marks, class_names)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')  
