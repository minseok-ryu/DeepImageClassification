#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 21:01:23 2023

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


def annotateOnFigure(string, colr, x, y, fSiz, ax):
    ax.annotate(string, 
                 color=colr, xy=(x, y), xycoords='axes fraction', xytext=(-20, 20), 
                 textcoords='offset pixels', horizontalalignment='center',
                 verticalalignment='center', fontsize=fSiz,
                 # bbox=dict(facecolor='silver', edgecolor='none', boxstyle='round,pad=0.06', alpha=0.3)
                 )
 
def annotateOnFigure_v2(string, colr, x, y, fSiz, edgecolr, ax):
    ax.annotate(string, 
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

def plot_per_class(df, ax):
    gl  = df[df['label']==1]
    ngl = df[df['label']==0]
    
    ax.scatter(gl['x_feat'], gl['y_feat'], c = 'firebrick', edgecolor='k', s=70)
    ax.scatter(ngl['x_feat'], ngl['y_feat'], c = 'darkblue', edgecolor='k', s=70)
    
    

def tsne_per_model(path, save_dir, itit):
    fig, ax       = plt.subplots(2, 2, figsize=(11, 6.5)); ax = ax.flatten()
    rot = 68

    # ============================ read tsne results for RWD model
    df_IODA_model = pd.read_csv(os.path.join(path, 'IODA_IODA_all_RWD_Public_CROPS.csv'))
    # ============================ rotate tsne results
    tsne_res_rot  = rotation(df_IODA_model[['x_feat', 'y_feat']], rot); tsne_res_rot.columns = ['x_feat', 'y_feat']
    df_IODA_model = pd.concat([tsne_res_rot, df_IODA_model.iloc[:, 2:]], axis=1)        
    # ============ scale and move the coordinates so they fit [0; 1] range
    df_IODA_model['x_feat'] = scale_to_01_range(df_IODA_model['x_feat'])
    df_IODA_model['y_feat'] = scale_to_01_range(df_IODA_model['y_feat'])
    # ============================get only test set data
    df_IODA_model = df_IODA_model[df_IODA_model['data_split']=='test']

    # ============================ read tsne results for public model
    df_Pub_model  = pd.read_csv(os.path.join(path, 'Public_Public_all_RWD_Public_CROPS.csv'))
    # ============================ rotate tsne results
    tsne_res_rot  = rotation(df_Pub_model[['x_feat', 'y_feat']], rot); tsne_res_rot.columns = ['x_feat', 'y_feat']
    df_Pub_model  = pd.concat([tsne_res_rot, df_Pub_model.iloc[:, 2:]], axis=1)        
    # ============ scale and move the coordinates so they fit [0; 1] range
    df_Pub_model['x_feat'] = scale_to_01_range(df_Pub_model['x_feat'])
    df_Pub_model['y_feat'] = scale_to_01_range(df_Pub_model['y_feat'])
    # ============================get only test set data
    df_Pub_model  = df_Pub_model[df_Pub_model['data_split']=='test']
    # ============================ 

    pub_RWD = df_Pub_model[df_Pub_model['type'] == 'RWD']
    pub_pub = df_Pub_model[df_Pub_model['type'] == 'Public']

    RwD_RwD = df_IODA_model[df_IODA_model['type'] == 'RWD']    
    RWD_pub = df_IODA_model[df_IODA_model['type'] == 'Public']
    
    plot_per_class(pub_pub, ax[0])
    plot_per_class(RwD_RwD, ax[1])
    plot_per_class(pub_RWD, ax[2])
    plot_per_class(RWD_pub, ax[3])

    # Set xtick labels and fontsize
    xticks      = [0, 0.2, 0.4, 0.6, 0.8, 1]; yticks  = [0.2, 0.4, 0.6, 0.8, 1]
    xtick_labls = ['0', '0.2', '0.4', '0.6', '0.8', '1']; ytick_labls = ['0.2', '0.4', '0.6', '0.8', '1']; 
    
    ax[0].set_xlim([-0.02,1.02]); ax[0].set_ylim([-0.02,1.2])
    ax[1].set_xlim([-0.02,1.02]); ax[1].set_ylim([-0.02,1.2])
    ax[2].set_xlim([-0.02,1.02]); ax[2].set_ylim([-0.02,1.2])
    ax[3].set_xlim([-0.02,1.02]); ax[3].set_ylim([-0.02,1.2])
    
    ax[0].set_xticks(xticks); ax[1].set_xticks(xticks); ax[2].set_xticks(xticks); ax[3].set_xticks(xticks); 
    ax[0].set_yticks(yticks); ax[1].set_yticks(yticks); ax[2].set_yticks(yticks); ax[3].set_yticks(yticks)
    
    ax[0].set_xticklabels(xtick_labls, fontsize=13); ax[0].set_yticklabels(ytick_labls, fontsize=13); 
    ax[1].set_xticklabels(xtick_labls, fontsize=13); ax[1].set_yticklabels(ytick_labls, fontsize=13)
    ax[2].set_xticklabels(xtick_labls, fontsize=13); ax[2].set_yticklabels(ytick_labls, fontsize=13)
    ax[3].set_xticklabels(xtick_labls, fontsize=13); ax[3].set_yticklabels(ytick_labls, fontsize=13)    
    
    annotateOnFigure_v2(f'A: {itit[0]}', 'k', 0.28, 0.90, 13.5 , 'k', ax[0]); #annotateOnFigure_v2(itit[0], 'k', 0.23, 0.06, 11, 'none', ax[0])
    annotateOnFigure_v2(f'B: {itit[1]}', 'k', 0.28, 0.90, 13.5, 'k', ax[1]); #annotateOnFigure_v2(itit[1], 'k', 0.21, 0.06, 11, 'none', ax[1])
    annotateOnFigure_v2(f'C: {itit[2]}', 'k', 0.28, 0.90, 13.5, 'k', ax[2]); #annotateOnFigure_v2(itit[2], 'k', 0.21, 0.06, 11, 'none', ax[2])
    annotateOnFigure_v2(f'D: {itit[3]}', 'k', 0.28, 0.90, 13.5, 'k', ax[3]); #annotateOnFigure_v2(itit[3], 'k', 0.21, 0.06, 11, 'none', ax[3])

    log=[]; log.append('Glaucoma'); log.append('Non-glaucoma')
    ax[3].legend(log, bbox_to_anchor= (0.5, -0.2), ncol=2, fontsize=15, borderaxespad=0, frameon=True) #     

    # Set the borders to a given color...
    for iax in ax:
        # ax.tick_params(color='k', labelcolor='k')
        for spine in iax.spines.values():
            spine.set_edgecolor('grey')
            
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    # plt.savefig(save_dir+'fig4_supp2.png', dpi = 400, bbox_inches='tight', pad_inches = 0.1)
    


def tsne_per_dataset(path, save_dir): # exp_names = ['IODA', 'Public']
    fig, ax       = plt.subplots(1, 2, figsize=(9,4)); ax = ax.flatten()
    rot = 68

    # ============================ read tsne results for RWD model
    df_IODA_model = pd.read_csv(os.path.join(path, 'IODA_IODA_all_RWD_Public_CROPS.csv'))
    # ============================ rotate tsne results
    tsne_res_rot = rotation(df_IODA_model[['x_feat', 'y_feat']], rot); tsne_res_rot.columns = ['x_feat', 'y_feat']
    df_IODA_model     = pd.concat([tsne_res_rot, df_IODA_model.iloc[:, 2:]], axis=1)        
    # ============ scale and move the coordinates so they fit [0; 1] range
    df_IODA_model['x_feat'] = scale_to_01_range(df_IODA_model['x_feat'])
    df_IODA_model['y_feat'] = scale_to_01_range(df_IODA_model['y_feat'])
    # ============================get only test set data
    df_IODA_model = df_IODA_model[df_IODA_model['data_split']=='test']

    # ============================ read tsne results for public model
    df_Pub_model = pd.read_csv(os.path.join(path, 'Public_Public_all_RWD_Public_CROPS.csv'))
    # ============================ rotate tsne results
    tsne_res_rot = rotation(df_Pub_model[['x_feat', 'y_feat']], rot); tsne_res_rot.columns = ['x_feat', 'y_feat']
    df_Pub_model = pd.concat([tsne_res_rot, df_Pub_model.iloc[:, 2:]], axis=1)        
    # ============ scale and move the coordinates so they fit [0; 1] range
    df_Pub_model['x_feat'] = scale_to_01_range(df_Pub_model['x_feat'])
    df_Pub_model['y_feat'] = scale_to_01_range(df_Pub_model['y_feat'])
    # ============================get only test set data
    df_Pub_model = df_Pub_model[df_Pub_model['data_split']=='test']
    # ============================ 

    pub_RWD = df_Pub_model[df_Pub_model['type'] == 'RWD']
    pub_pub = df_Pub_model[df_Pub_model['type'] == 'Public']

    RWD_RWD = df_IODA_model[df_IODA_model['type'] == 'RWD']    
    RWD_pub = df_IODA_model[df_IODA_model['type'] == 'Public']
    
    ax[0].scatter(pub_pub['x_feat'], pub_pub['y_feat'], c = 'darkorange', edgecolor='k', s=60)
    ax[0].scatter(pub_RWD['x_feat'], pub_RWD['y_feat'], c = 'green', edgecolor='k', s=60)

    ax[1].scatter(RWD_pub['x_feat'], RWD_pub['y_feat'], c = 'darkorange', edgecolor='k', s=60)
    ax[1].scatter(RWD_RWD['x_feat'], RWD_RWD['y_feat'], c = 'green', edgecolor='k', s=60)

    ax[0].set_xlim([-0.02,1.02]); ax[0].set_ylim([-0.02,1.02])
    ax[0].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax[0].set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=12); 
    ax[0].set_yticks([0.2, 0.4, 0.6, 0.8, 1.0]); ax[0].set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=12)

    ax[1].set_xlim([-0.02,1.02]); ax[1].set_ylim([-0.02,1.02])
    ax[1].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax[1].set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=12); 
    ax[1].set_yticks([0.2, 0.4, 0.6, 0.8, 1.0]); ax[1].set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=12)

    plt.grid(False)
    
    log=[]; log.append('Real-world data'); log.append('Public data')

    # Create a custom legend
    handles = [plt.Line2D([], [], marker='o', markersize=10, markeredgecolor='k', color=color, linestyle='None') for color in ['darkorange', 'green']]
    plt.legend(handles, log, fontsize=13, bbox_to_anchor=(0.5, -0.15),  ncol=2) 

    # Set the borders to a given color...
    for iax in ax:
        # ax.tick_params(color='k', labelcolor='k')
        for spine in iax.spines.values():
            spine.set_edgecolor('grey')

    # plt.subplots_adjust()
    annotateOnFigure_v2('A', 'k', 0.11, 0.9, 15, 'k', ax[0])
    annotateOnFigure_v2('B', 'k', 0.11, 0.9, 15, 'k', ax[1])

    # plt.tight_layout()
    plt.savefig(save_dir+'fig7_manuscript.png', dpi = 350, bbox_inches='tight', pad_inches = 0.1)
    
    # ---------- calculate Wasserstein distance
    wd_RWD_on_datasets = twoD_Wasserstein_dist(RWD_RWD[['x_feat','y_feat']].to_numpy(), RWD_pub[['x_feat','y_feat']].to_numpy(), 'all')
    wd_Pub_on_datasets = twoD_Wasserstein_dist(pub_RWD[['x_feat','y_feat']].to_numpy(), pub_pub[['x_feat','y_feat']].to_numpy(), 'all')

    print(f'wd_RWD_on_datasets = {wd_RWD_on_datasets}')
    print(f'wd_Pub_on_datasets = {wd_Pub_on_datasets}')
    
      

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
    
    reading_path = '/Volumes/homa/Homa/Glaucoma_prediction/Real_World_Data_paper/tsne/revision/'
    saveDir      = '/Volumes/homa/Homa/Glaucoma_prediction/Real_World_Data_paper/tsne/revision/'
    
    # ---------- TSNE plot per model
    subtitles    = ['Public->Public', 'RWD->RWD', 'Public->RWD', 'RWD->Public']
    tsne_per_model(reading_path, saveDir, subtitles)    
    
    # ---------- TSNE plot per dataset    
    # tsne_per_dataset(reading_path, saveDir)
    
    # ---------- Quantify TSNE features' diversity and similarity:   table 5 in paper/ table 2 in supp
    # tsne_res = pd.read_csv(os.path.join(reading_path, 'randModel_all_RWD_Public_CROPS.csv'))
    # train = tsne_res[tsne_res['data_split'] == 'train'] 
    # test  = tsne_res[tsne_res['data_split'] == 'test'] 
    
    # get_tsne_diversity_spread(train, 'train', 'type')
    # print('='*30)
    # get_tsne_diversity_spread(test, 'test', 'type')    
    # print('='*30)
    # # tsne_class_conv(tsne_res, 'type')
    # # print('='*30)
    
    # RwD = tsne_res[tsne_res['type'] == 'RWD'] 
    # Pub = tsne_res[tsne_res['type'] == 'Public'] 
    
    # get_tsne_diversity_spread(RwD, 'RwD', 'data_split')
    # print('='*30)
    # get_tsne_diversity_spread(Pub, 'Pub', 'data_split')    
    # print('='*30)
    
    
    # ======== table 9 in paper
    # tsne_res = pd.read_csv(os.path.join(reading_path, 'IODA_IODA_all_RWD_Public_CROPS.csv'))
    # test  = tsne_res[tsne_res['data_split'] == 'test'] 
    
    # get_tsne_diversity_spread(test, 'test', 'type')    
    # print('='*30)
    
    # RwD = tsne_res[tsne_res['type'] == 'RWD'] 
    # # Pub = tsne_res[tsne_res['type'] == 'Public'] 
    
    # get_tsne_diversity_spread(RwD, 'RwD', 'data_split')
    # print('='*30)
    # get_tsne_diversity_spread(Pub, 'Pub', 'data_split')    
    # print('='*30)
    
    