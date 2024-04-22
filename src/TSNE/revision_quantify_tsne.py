#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 05:43:07 2023

@author: homai
"""

import os
import sys
import argparse
from tqdm import tqdm
from numpy import savetxt
import cv2
import torch
import random
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score


def my_inter_cluster_distance(X, col):
    labels = np.unique(X[col])
    n_clusters = labels.size
    centroids  = pd.DataFrame()
    ic_dist    = pd.DataFrame() #np.zeros((n_clusters, 1))# inter cluster distance
    
    # ================================ calculate the coordinate of centroids
    for label in labels:
        df = X[X['dataset']==label]
        # centroid[i, :] = np.mean(df)
        
        centroids[label] = np.mean(df)
    # ================================ calculate the mean disctance between each cluster centroid to other centroids
    for i, this_label in enumerate(labels):
        distances = np.zeros((1, n_clusters))
        for j, other_labels in enumerate(labels):
            distances[:, j] = np.sqrt(np.sum((centroids[this_label] - centroids[other_labels])**2))
        
        ic_dist.loc[0,this_label] = np.mean(distances)
    return ic_dist


def get_cluster_SD(X, col):
    labels           = np.unique(X[col])
    sd               = pd.DataFrame() 
    # ================================ calculate SD of each cluster
    print(f'Mean standard deviation across (x, y) directions: ')
    for label in labels:
        df = X[X['dataset']==label]
        sd[label] = np.std(df)
        print(f'{label}= {np.mean(sd[label])}')
    print('='*30)
    return sd
        

def k_means(tsne, n_clusters):
    # Fit KMeans clustering on the t-SNE transformed data
    kmeans = KMeans(n_clusters=n_clusters)
    pred_labels = kmeans.fit_predict(tsne)
    return pred_labels


def my_calinski_harabasz_score(X, col):
    labels = np.unique(X[col])
    n_clusters = labels.size
    centroids  = pd.DataFrame()
    overal_centroid = np.mean(X)
    distances = np.zeros((1, n_clusters))
    
    # ================================ calculate the coordinate of centroids
    for i, label in enumerate(labels):
        df = X[X[col]==label]
        # centroid[i, :] = np.mean(df)
        
        centroids[label] = np.mean(df)
    
        distances[:, i] = np.sqrt(np.sum((centroids[label] - overal_centroid)**2))
    return distances


def calinski_harabasz_score(tsne, labels):
    # Compute Calinski-Harabasz Index
    # score = calinski_harabasz_score(tsne[['tsne_x', 'tsne_y']], tsne['labels'])

    # Calculate the Calinski-Harabasz index score
    ch_score = calinski_harabasz_score(tsne, labels)
    print("Calinski-Harabasz Index:", ch_score)


def silhouette_score_per_cluster(data, labels):
    unique_labels = np.unique(labels)
    n_clusters = len(np.unique(labels))
    n_samples = data.shape[0]
    silhouette_scores = []
    for i, labl in enumerate(unique_labels):
        in_cluster = data[labels == labl]
        out_cluster = data[labels != labl]
        avg_intra_cluster_distance = np.mean([np.linalg.norm(x-in_cluster[j]) for j in range(len(in_cluster)) for x in in_cluster])
        
        avg_farthest_cluster_distance = np.mean([max([np.linalg.norm(x-out_cluster[j]) for j in range(len(out_cluster))]) for x in in_cluster])
        avg_nearest_cluster_distance = np.mean([min([np.linalg.norm(x-out_cluster[j]) for j in range(len(out_cluster))]) for x in in_cluster])
        score = (avg_nearest_cluster_distance - avg_intra_cluster_distance) / max(avg_intra_cluster_distance, avg_nearest_cluster_distance)
        silhouette_scores.append(score)
        # print(f'Silhouette= {np.round(score,1)}')
        print(f'{labl}: Mean intra-cluster distance= {np.round(avg_intra_cluster_distance,1)}, Mean farthest cluster distance= {np.round(avg_farthest_cluster_distance,1)}')
    return silhouette_scores


def tsne_class_conv(df, col):
    unique_labels = np.unique(df[col])
    
    for labl in unique_labels:
        sub_df  = df[df[col] == labl]
        cov_mat = np.cov(sub_df[['x_feat', 'y_feat']].to_numpy().T)
        trace   = np.trace(cov_mat)
        
        print(f'Trace of cov matrix of T-SNE (x, y) coordinates for {labl} = {np.round(trace, 2)}')
        


# Trace of cov matrix for BinRushed= 987.66
# Trace of cov matrix for Drishti-GS= 217.08
# Trace of cov matrix for MESSIDOR= 127.76
# Trace of cov matrix for Magrabi= 65.06
# Trace of cov matrix for REFUGE= 269.33
# Trace of cov matrix for RWD= 813.08



def rotation(vector, angle): # angle = np.pi/2
    R_mx = np.zeros((2,2))
    R_mx[0,0] = np.cos(angle)
    R_mx[0,1] = -1 * np.sin(angle)
    R_mx[1,0] = np.sin(angle)
    R_mx[1,1] = np.cos(angle)
    
    rot_vect = R_mx @ vector.T
    rot_vect = rot_vect.T
    # rot_vect.columns = ['x', 'y']
    return rot_vect


def labels_to_int(df):
    if df['dataset'] == 'REFUGE':
        val = 0
    elif df['dataset'] == 'Magrabi':
        val = 1
    elif df['dataset'] == 'MESSIDOR':
        val = 2
    elif df['dataset'] == 'BinRushed':
        val = 3
    elif df['dataset'] == 'Drishti-GS':
        val = 4
    elif df['dataset'] == 'RWD':
        val = 5
    return val



# ========================================
fig = 'only_classification/'
# path   = '/Volumes/homa/Homa/Glaucoma_prediction/Real_World_Data_paper/tsne/'+fig+'used_in_paper/'
# df     = pd.read_csv(path + 'tsne_labeled.csv')

path   = '/Volumes/homa/Homa/Glaucoma_prediction/Real_World_Data_paper/tsne/revision/'
df     = pd.read_csv(path + 'randModel_all_RWD_Public.csv')
colum  = 'type'

# covariance_matrix = np.cov(df[['tsne_x', 'tsne_y']].to_numpy())
tsne_class_conv(df, colum)

sys.exit()

# =========================================
ic_distance = my_inter_cluster_distance(df, colum)
print('Mean inter-cluster distance: \n'); print(ic_distance); print('='*30)

# # =========================================
sd = get_cluster_SD(df, colum)
print('Standard deviation in (x, y) directions: \n'); print(sd); print('='*30)

# =========================================
# calinski_harabasz_score(t_sne, labels)
ch_score = my_calinski_harabasz_score(df, 'dataset')

# =========================================
silhouette_scores = silhouette_score_per_cluster(df[['x_feat', 'y_feat']].to_numpy(), df[colum].to_numpy())

