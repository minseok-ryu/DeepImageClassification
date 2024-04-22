#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 10:10:01 2022

@author: homai
"""

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
                 bbox=dict(facecolor='none', edgecolor=edgecolr, pad=2, linewidth =0.5, alpha=0.5) # boxstyle='round,pad=1',
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


def tsne_per_model(fig, axs, path, exp_names, label, itit, save_fig):
    for idx, iexp in enumerate(exp_names):
        read_dir = path + iexp + '/prediction/results/'
        
        tsne_gl = pd.read_csv(read_dir + 'tsn_glaucoma.csv')
        tsne_nongl = pd.read_csv(read_dir + 'tsn_non-glaucoma.csv') 

        # ---------- rotate values for a better visualization
        if iexp == 'IODA_IODA' or iexp == 'IODA_Public':
            tsne_gl = rotation(tsne_gl, -np.pi/4.2)
            tsne_nongl = rotation(tsne_nongl, -np.pi/4.2)
        
        elif iexp == 'Public_Public':
            tsne_gl = rotation(tsne_gl, -np.pi/5)
            tsne_nongl = rotation(tsne_nongl, -np.pi/5)

        elif iexp == 'Public_IODA':
            tsne_gl = rotation(tsne_gl, -np.pi/3)
            tsne_nongl = rotation(tsne_nongl, -np.pi/3)            
      
        # ---------- scale values to [0,1] range for a better visualization
        tsne_gl['x'] = scale_to_01_range(tsne_gl['x'])
        tsne_gl['y'] = scale_to_01_range(tsne_gl['y'])
        
        tsne_nongl['x'] = scale_to_01_range(tsne_nongl['x'])
        tsne_nongl['y'] = scale_to_01_range(tsne_nongl['y'])       
    
        # ---------- plot scatter values
        a = axs[idx].scatter(tsne_nongl['x'], tsne_nongl['y'], color='darkblue', edgecolor='k')
        b = axs[idx].scatter(tsne_gl['x'], tsne_gl['y'], color='firebrick', edgecolor='k')

        axs[idx].grid(False)    # Hide grid lines
        axs[idx].axis('on', color='gray', alpha=0.2)
        axs[idx].tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    
        # axs[idx].set_xlim([0,1])
        axs[idx].set_ylim([-0.2,1.05])
        
        annotateOnFigure_v2(label[idx], 'k', 0.14 , 0.055, 13, 'k', axs[idx])
        annotateOnFigure_v2(itit[idx], 'k', 0.6, 0.055, 9, 'none', axs[idx])
        

    log=[]; log.append('Non-glaucoma'); log.append('Glaucoma')
    axs[idx].legend([a,b], log, bbox_to_anchor= (-0.25, -0.05), ncol=2, borderaxespad=0, frameon=True) # 
    
    # Set the borders to a given color...
    for ax in axs:
        # ax.tick_params(color='k', labelcolor='k')
        for spine in ax.spines.values():
            spine.set_edgecolor('grey')
            
    plt.subplots_adjust(wspace=0.08, hspace=0.05)
    if save_fig:
        plt.savefig(path+'TSNE_res_new_v4.png', bbox_inches='tight', pad_inches = 0.1)
    

def tsne_per_dataset(fig, axs, path, exp_names, label, save_fig): # exp_names = ['IODA', 'Public']
    for idx, iexp in enumerate(exp_names):
        iexp1 = iexp + '_IODA'
        iexp2 = iexp + '_Public'
        
        read_dir1       = path + iexp1 + '/prediction/results/'
        tsne_gl_IODA    = pd.read_csv(read_dir1 + 'tsn_glaucoma.csv')
        tsne_nongl_IODA = pd.read_csv(read_dir1 + 'tsn_non-glaucoma.csv')
        
        read_dir2      = path + iexp2 + '/prediction/results/'
        tsne_gl_Pub    = pd.read_csv(read_dir2 + 'tsn_glaucoma.csv')
        tsne_nongl_Pub = pd.read_csv(read_dir2 + 'tsn_non-glaucoma.csv')

        # ---------- rotate values for a better visualization
        if iexp1 == 'IODA_IODA' or iexp2 == 'IODA_IODA':
            tsne_gl_IODA    = rotation(tsne_gl_IODA, -np.pi/4)
            tsne_nongl_IODA = rotation(tsne_nongl_IODA, -np.pi/4)
            
        if iexp1 == 'IODA_Public' or iexp2 == 'IODA_Public':    
            tsne_gl_Pub    = rotation(tsne_gl_Pub, -np.pi/4)
            tsne_nongl_Pub = rotation(tsne_nongl_Pub, -np.pi/4)
            
        if iexp1 == 'Public_Public' or iexp2 == 'Public_Public':
            tsne_gl_Pub = rotation(tsne_gl_Pub, -np.pi/4)
            tsne_nongl_Pub = rotation(tsne_nongl_Pub, -np.pi/4)

        if iexp1 == 'Public_IODA' or iexp2 == 'Public_IODA':
            tsne_gl_IODA = rotation(tsne_gl_IODA, -np.pi/4)
            tsne_nongl_IODA = rotation(tsne_nongl_IODA, -np.pi/4)       
    
        # ---------- scale values to [0,1] range for a better visualization
        # tsne_gl_IODA['x'] = scale_to_01_range(tsne_gl_IODA['x'])
        # tsne_gl_IODA['y'] = scale_to_01_range(tsne_gl_IODA['y'])
        
        # tsne_nongl_IODA['x'] = scale_to_01_range(tsne_nongl_IODA['x'])
        # tsne_nongl_IODA['y'] = scale_to_01_range(tsne_nongl_IODA['y'])       
    

        # tsne_gl_Pub['x'] = scale_to_01_range(tsne_gl_Pub['x'])
        # tsne_gl_Pub['y'] = scale_to_01_range(tsne_gl_Pub['y'])
        
        # tsne_nongl_Pub['x'] = scale_to_01_range(tsne_nongl_Pub['x'])
        # tsne_nongl_Pub['y'] = scale_to_01_range(tsne_nongl_Pub['y'])   

        # ---------- Merge df
        df_IODA = pd.DataFrame()
        df_IODA = df_IODA.append(tsne_gl_IODA)
        df_IODA = df_IODA.append(tsne_nongl_IODA)
        df_IODA_np = df_IODA.to_numpy()
        
        df_Pub = pd.DataFrame()
        df_Pub = df_Pub.append(tsne_gl_Pub)
        df_Pub = df_Pub.append(tsne_nongl_Pub) 
        df_Pub_np = df_Pub.to_numpy()
        # ----------
        # gmm_IODA = GMM(df_IODA_np, 1, 1000)
        # gmm_Pub  = GMM(df_Pub_np, 1, 1000)
        
        # ---------- calculate Wasserstein distance
        twoD_Wasserstein_dist(df_IODA_np, df_Pub_np)

        # ---------- plot scatter values
        axs[idx].scatter(tsne_gl_IODA['x'], tsne_gl_IODA['y'], color='green', edgecolor='k')
        a = axs[idx].scatter(tsne_nongl_IODA['x'], tsne_nongl_IODA['y'], color='green', edgecolor='k')

        axs[idx].scatter(tsne_gl_Pub['x'], tsne_gl_Pub['y'], color='darkorange', edgecolor='k')
        b = axs[idx].scatter(tsne_nongl_Pub['x'], tsne_nongl_Pub['y'], color='darkorange', edgecolor='k')

        axs[idx].set_aspect(aspect='auto', adjustable='datalim') 
        # axs[idx].set_aspect(aspect='equal') 
    
        axs[idx].grid(False)    # Hide grid lines
        # axs[idx].axis('on', color='gray', alpha=0.2)
        axs[idx].tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    
        annotateOnFigure_v2(label[idx], 'k', 0.125, 0.865, 13, 'k', axs[idx])
    
        # print(axs[idx].axis())
    log=[]; log.append('Real-world data'); log.append('Public data')
    axs[idx].legend([a,b], log, bbox_to_anchor= (0.5, -0.1), ncol=2, borderaxespad=0, frameon=True) #     

    # axs[idx].set_ylim([-0.05,1.2])
    
    # Set the borders to a given color...
    for ax in axs:
        # ax.tick_params(color='k', labelcolor='k')
        for spine in ax.spines.values():
            spine.set_edgecolor('grey')

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    # if save_fig:
        # plt.savefig(path+'supp_TSNE_res_new_v2.png', bbox_inches='tight', pad_inches = 0.1)
   

def twoD_Wasserstein_dist(dist_1, dist_2):
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment    
   
    d = cdist(dist_1, dist_2)
    assignment = linear_sum_assignment(d)
    wasser_dist = d[assignment].sum() / len(dist_1)
    print(f'Wasserstein distance version 1 = {wasser_dist}')
    
    # from scipy.stats import wasserstein_distance
    # w_dist = wasserstein_distance(dist_1.reshape(-1), dist_2.reshape(-1))
    # print(f'Wasserstein distance version 2 = {w_dist}')
    # print(' '); print('='*40)


def GMM(X, n_class, n_sample):
    import numpy as np
    from sklearn.mixture import GaussianMixture
    
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    
    # X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    gmm = GaussianMixture(n_components=n_class, random_state=0).fit(X)
    # gmm.means_
    samples = gmm.sample(n_sample)[0]
    return samples




if __name__=="__main__":
    
    reading_path = '/Volumes/homa/Homa/Glaucoma_prediction/Real_World_Data_paper/GL_classification/'
    
    # ---------- TSNE plot per model
    # fig1, axs1       = plt.subplots(1, 4, figsize=(8.5,2.5), sharey=True, sharex=True, dpi = 250); axs1 = axs1.flatten()   # , constrained_layout=False 
    # experiment_names = ['Public_Public', 'IODA_IODA', 'Public_IODA', 'IODA_Public']
    # subtitles        = ['Public->Public', 'RWD->RWD', 'Public->RWD', 'RWD->Public']
    # labels           = ['A',           'B',       'C',         'D']
    # tsne_per_model(fig1, axs1, reading_path, experiment_names, labels, subtitles, False)
    
    
    # ---------- TSNE plot per dataset    
    fig1, axs1       = plt.subplots(1, 2, figsize=(6,3), dpi = 250); axs1 = axs1.flatten()
    experiment_names = ['Public', 'IODA']
    labels           = ['A',  'B']
    tsne_per_dataset(fig1, axs1, reading_path, experiment_names, labels, False)
    
    
    