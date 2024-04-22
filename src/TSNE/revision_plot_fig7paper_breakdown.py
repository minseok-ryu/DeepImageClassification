#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 20:27:54 2024

@author: homai
"""

import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from visualize_features import scale_to_01_range
 
def annotateOnFigure_v2(string, colr, x, y, fSiz, edgecolr):
    plt.annotate(string, 
                 color=colr, xy=(x, y), xycoords='axes fraction', xytext=(-10, 10), 
                 textcoords='offset pixels', horizontalalignment='center',
                 verticalalignment='center', fontsize=fSiz,
                 bbox=dict(facecolor='none', edgecolor=edgecolr, pad=4, linewidth =0.5, alpha=0.5) # boxstyle='round,pad=1',
                 )    

def rotation(vector, angle): # angle = np.pi/2
    R_mx = np.zeros((2,2))
    R_mx[0,0] = np.cos(angle)
    R_mx[0,1] = -1*np.sin(angle)
    R_mx[1,0] = np.sin(angle)
    R_mx[1,1] = np.cos(angle)
    
    rot_vect = R_mx @ vector.T
    rot_vect = rot_vect.T
    rot_vect.columns = ['x', 'y']
    return rot_vect

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

    

def tsne_public_breakdown(path, save_dir): # exp_names = ['IODA', 'Public']
    rot = 68
    
    # ============================ read tsne results for RWD model
    df_IODA_model = pd.read_csv(os.path.join(path, 'IODA_IODA_all_RWD_Public_CROPS.csv'))
    # ============================ rotate tsne results
    tsne_res_rot = rotation(df_IODA_model[['x_feat', 'y_feat']], rot); tsne_res_rot.columns = ['x_feat', 'y_feat']
    df_IODA_model     = pd.concat([tsne_res_rot, df_IODA_model.iloc[:, 2:]], axis=1)        
    # ============ scale and move the coordinates so they fit [0; 1] range
    df_IODA_model['x_feat'] = scale_to_01_range(df_IODA_model['x_feat'])
    df_IODA_model['y_feat'] = scale_to_01_range(df_IODA_model['y_feat'])
    # ============================ get train and test sets individually
    IODA_test = df_IODA_model[df_IODA_model['data_split']=='test']
    IODA_train = df_IODA_model[df_IODA_model['data_split']=='train']


    # ============================ read tsne results for public model
    df_Pub_model = pd.read_csv(os.path.join(path, 'Public_Public_all_RWD_Public_CROPS.csv'))
    # ============================ rotate tsne results
    tsne_res_rot = rotation(df_Pub_model[['x_feat', 'y_feat']], rot); tsne_res_rot.columns = ['x_feat', 'y_feat']
    df_Pub_model = pd.concat([tsne_res_rot, df_Pub_model.iloc[:, 2:]], axis=1)        
    # ============ scale and move the coordinates so they fit [0; 1] range
    df_Pub_model['x_feat'] = scale_to_01_range(df_Pub_model['x_feat'])
    df_Pub_model['y_feat'] = scale_to_01_range(df_Pub_model['y_feat'])
    # ============================ get train and test sets individually
    pub_test = df_Pub_model[df_Pub_model['data_split']=='test']
    pub_train = df_Pub_model[df_Pub_model['data_split']=='train']

    # ============================ Using public model: plot Public test versus Public train
    fig2 = plt.figure(figsize=(6.5,5.5))
    plt.scatter(pub_train['x_feat'], pub_train['y_feat'], c = 'green', edgecolor='k', marker='o', s=200, alpha=0.8)
    plt.scatter(pub_test['x_feat'], pub_test['y_feat'], c = 'green', edgecolor='w', marker='X', s=200, alpha=0.8)  
    plt.legend(['$Public_{train}$', '$Public_{test}$'], loc='lower right', fontsize=18)
    annotateOnFigure_v2('A', 'k', 0.11, 0.9, 17, 'k')
    plt.xticks(fontsize=15); plt.yticks(fontsize=15)
    plt.tight_layout(); plt.savefig(save_dir+'fig7_manuscript_breakdown_A.png', dpi = 300, bbox_inches='tight', pad_inches = 0.1)
    # ============================ Using public model: plot Public test versus IODA test + Public train (in gray)
    fig1 = plt.figure(figsize=(6.5,5.5))
    plt.scatter(pub_train['x_feat'], pub_train['y_feat'], c = 'gray', edgecolor='k', marker='o', s=200, alpha=0.6)
    plt.scatter(pub_test['x_feat'], pub_test['y_feat'], c = 'green', edgecolor='w', marker='X', s=200, alpha=0.8)  
    plt.scatter(IODA_test['x_feat'], IODA_test['y_feat'], c = 'darkorange', edgecolor='w', marker='X', s=200, alpha=0.8)  
    plt.legend(['$Public_{train}$', '$Public_{test}$', '$UIC_{test}$'], loc='lower right', fontsize=18)
    annotateOnFigure_v2('B', 'k', 0.11, 0.9, 17, 'k')
    plt.xticks(fontsize=15); plt.yticks(fontsize=15)
    plt.tight_layout(); plt.savefig(save_dir+'fig7_manuscript_breakdown_B.png', dpi = 300, bbox_inches='tight', pad_inches = 0.1)

    # ============================ Using IODA model: plot IODA test versus IODA train
    fig4 = plt.figure(figsize=(6.5,5.5))
    plt.scatter(IODA_train['x_feat'], IODA_train['y_feat'], c = 'darkorange', edgecolor='k', marker='o', s=200, alpha=0.8)
    plt.scatter(IODA_test['x_feat'], IODA_test['y_feat'], c = 'darkorange', edgecolor='w', marker='X', s=200, alpha=0.8)  
    plt.legend(['$UIC_{train}$', '$UIC_{test}$'], loc='lower right', fontsize=18)
    annotateOnFigure_v2('C', 'k', 0.11, 0.9, 17, 'k')
    plt.xticks(fontsize=15); plt.yticks(fontsize=15)
    plt.tight_layout(); plt.savefig(save_dir+'fig7_manuscript_breakdown_C.png', dpi = 300, bbox_inches='tight', pad_inches = 0.1)    
    # ============================ Using IODA model: plot Public test versus IODA test + IODA train (in gray)
    fig3 = plt.figure(figsize=(6.5,5.5))
    plt.scatter(IODA_train['x_feat'], IODA_train['y_feat'], c = 'gray', edgecolor='k', marker='o', s=200, alpha=0.6)
    plt.scatter(IODA_test['x_feat'], IODA_test['y_feat'], c = 'darkorange', edgecolor='w', marker='X', s=200, alpha=0.8)  
    plt.scatter(pub_test['x_feat'], pub_test['y_feat'], c = 'green', edgecolor='w', marker='X', s=200, alpha=0.8)  
    plt.legend(['$UIC_{train}$', '$UIC_{test}$', '$Public_{test}$'], loc='lower right', fontsize=18)
    annotateOnFigure_v2('D', 'k', 0.11, 0.9, 17, 'k')
    plt.xticks(fontsize=15); plt.yticks(fontsize=15)
    plt.tight_layout(); plt.savefig(save_dir+'fig7_manuscript_breakdown_D.png', dpi = 300, bbox_inches='tight', pad_inches = 0.1)


    
def tsne_class_conv(df, col):
    unique_labels = np.unique(df[col])
    print(f'unique_labels = {unique_labels}')
    
    for labl in unique_labels:
        sub_df  = df[df[col] == labl] # cov for what dataset? type? class label?
        cov_mat = np.cov(sub_df[['x_feat', 'y_feat']].to_numpy().T)
        trace   = np.trace(cov_mat)
        
        print(f'Trace of covatiance matrix of T-SNE (x, y) coordinates for {labl} = {trace}')
        
    
def twoD_Wasserstein_dist(dist_1, dist_2, name):
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment    
   
    d = cdist(dist_1, dist_2)
    assignment = linear_sum_assignment(d)
    wasser_dist = d[assignment].sum() / min(dist_1.shape[0], dist_2.shape[0])
    print(f'{name} Wasserstein distance = {wasser_dist}')
    return wasser_dist


def get_tsne_diversity_spread(df, df_name, column):
    tsne_class_conv(df, column)

    if column == 'data_split':
        df_RWD = df[df[column] == 'train'][['x_feat', 'y_feat']].to_numpy()
        df_Pub = df[df[column] == 'test'][['x_feat', 'y_feat']].to_numpy()        
    elif column == 'type':
        df_RWD = df[df[column] == 'RWD'][['x_feat', 'y_feat']].to_numpy()
        df_Pub = df[df[column] == 'Public'][['x_feat', 'y_feat']].to_numpy()    
    

    twoD_Wasserstein_dist(df_RWD, df_Pub, df_name)



if __name__=="__main__":
    
    reading_path = '/Volumes/homa/Homa/Glaucoma_prediction/1.RWD_paper/tsne/revision/'
    saveDir      = '/Volumes/homa/Homa/Glaucoma_prediction/1.RWD_paper/tsne/revision/'
      
    
    # ---------- TSNE plot per dataset    
    tsne_public_breakdown(reading_path, saveDir)
    
