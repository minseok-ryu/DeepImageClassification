#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 09:01:24 2022

@author: homai
"""

"Link to the generic code: https://github.com/spmallick/learnopencv/tree/master/TSNE"

import os
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

from TSNE.resnet import*


def fix_random_seeds():
    seed = 87
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_features(argpars, fileName, dataloader, num_class, do_rand_model, model):
    # ----------------------------------------
    # initialize our implementation of ResNet
    if argpars.net_archit == 'resnet50':
        model = ResNet50(model, argpars.save_dir_modl+fileName, num_classes=num_class, do_rand_model=do_rand_model, pretrained=True) #argpars.net_archit ,
    elif argpars.net_archit == 'resnet101':
        model = ResNet101(model, argpars.save_dir_modl+fileName, num_classes=num_class, do_rand_model=do_rand_model, pretrained=True)
    # ----------------------------------------
   
    model.eval()

    # we'll store the features as NumPy array of size num_images x feature_size
    features = None

    # we'll also store the image labels and paths to visualize them later
    labels = []
    image_paths = []
    
    # for inputs, targets in dataloader:
    #     images = inputs
    #     labels += targets
    #     # ------------------- moving data to gpu, if available
    #     if torch.cuda.is_available():
    #         model.cuda()
    #         images, labels = inputs.cuda(), targets.cuda()   
    
    for images, label, img_path in tqdm(dataloader, desc='Running the model inference to get features'):
        labels += label
        image_paths += img_path

        #     # ------------------- moving data to gpu, if available
        if torch.cuda.is_available():
            model.cuda()
            images = images.cuda()

        with torch.no_grad():
            output = model.forward(images)

        current_features = output.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, current_features))
        else:
            features = current_features
    return features, labels, image_paths


# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def scale_image(image, max_image_size):
    image_height, image_width, _ = image.shape

    scale = max(1, image_width / max_image_size, image_height / max_image_size)
    image_width = int(image_width / scale)
    image_height = int(image_height / scale)

    image = cv2.resize(image, (image_width, image_height))
    return image


def draw_rectangle_by_class(image, label, colors_per_class):
    image_height, image_width, _ = image.shape

    # get the color corresponding to image class
    color = colors_per_class[label]
    image = cv2.rectangle(image, (0, 0), (image_width - 1, image_height - 1), color=color, thickness=5)

    return image


def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    image_height, image_width, _ = image.shape

    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset

    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset
    # center_y = int(800 * (1 - y)) + offset

    # knowing the image center, compute the coordinates of the top left and bottom right corner
    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)

    br_x = tl_x + image_width
    br_y = tl_y + image_height

    return tl_x, tl_y, br_x, br_y


def visualize_tsne_images(args, tx, ty, images, labels, colors_per_class, plot_size=1000, max_image_size=100):
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    
    # we'll put the image centers in the central area of the plot
    # and use offsets to make sure the images fit the plot
    offset = max_image_size // 2
    image_centers_area_size = plot_size - 2 * offset

    tsne_plot = 255 * np.ones((plot_size, plot_size, 3), np.uint8)
    # tsne_plot = 255 * np.ones((1600, plot_size, 3), np.uint8)
    
    # now we'll put a small copy of every image to its corresponding T-SNE coordinate
    for image_path, label, x, y in tqdm(
            zip(images, labels, tx, ty),
            desc='Building the T-SNE plot',
            total=len(images)
            ):   
        # ======================= ******* change this line if necessary
        image = cv2.imread(image_path) # image_path: is a same file as annotate.csv having path + fileName.png

        # scale the image to put it to the plot
        image = scale_image(image, max_image_size)

        # draw a rectangle with a color corresponding to the image class
        image = draw_rectangle_by_class(image, label, colors_per_class)

        # compute the coordinates of the image on the scaled plot visualization
        tl_x, tl_y, br_x, br_y = compute_plot_coordinates(image, x, y, image_centers_area_size, offset)

        # put the image to its TSNE coordinates using numpy subarray indices
        tsne_plot[tl_y:br_y, tl_x:br_x, :] = image
    
    ax.imshow(tsne_plot[:, :, ::-1])
    
    plt.tight_layout(); #plt.legend(['Non-glaucoma', 'Glaucoma'], loc='upper right')
    plt.xticks([]); plt.yticks([]); plt.grid(False)
    # plt.grid(color='k', linewidth=1, alpha=0.3)
    
    # ax.set_aspect(aspect='auto', adjustable='datalim') 
    ax.set_aspect(aspect='equal') 
    
    plt.savefig(os.path.join(args.save_dir_res, 'TSNE_data_discovery.png'), dpi = 400, bbox_inches='tight', pad_inches = 0.1)
    plt.show()


def annotateOnFigure(string, colr, x, y, fSiz, ax):
    ax.annotate(string, 
             color=colr, xy=(x, y), xycoords='axes fraction', xytext=(-20, 20), 
             textcoords='offset pixels', horizontalalignment='center',
             verticalalignment='center', fontsize=fSiz,
             bbox=dict(facecolor='silver', edgecolor='none', boxstyle='round,pad=0.06', alpha=0.3))
    

def visualize_tsne_points(args, fileName, tx, ty, labels, colors_per_class):
    # initialize matplotlib plot
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)

    # # for every class, we'll add a scatter plot separately
    # for label in colors_per_class:
    #     # find the samples of the current class in the data
    #     indices = [i for i, l in enumerate(labels) if l.item() == int(label)]

    #     # extract the coordinates of the points of this class only
    #     current_tx = np.take(tx, indices)
    #     current_ty = np.take(ty, indices)

    #     # convert the class color to matplotlib format:
    #     # BGR -> RGB, divide by 255, convert to np.array
    #     color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255

    #     # add a scatter plot with the correponding color and label
    #     ax.scatter(current_tx, current_ty, c=color, label=label)

    for i in range(len(tx)):
        print("{}/{}".format(i,len(tx)))
        if labels[i]==0:
            a = plt.scatter(tx[i],ty[i], color=colors_per_class[0]) # , marker='+'
        elif labels[i]==1:
            b = plt.scatter(tx[i],ty[i], color=colors_per_class[1]) # , marker='X'

    annotateOnFigure('(c)', 'k', 0.1, 1, 13, ax)

    # ax.set_xlabel('', fontsize=12);   ax.set_ylabel('',fontsize=12)
    ax.set_xticklabels(['']); ax.set_yticklabels([''])
    # build a legend using the labels we set previously
    log=[]; log.append('non-glaucoma'); log.append('glaucoma')
    ax.legend([a,b], log, bbox_to_anchor= (0.25, 1), ncol=2, borderaxespad=0 , frameon=False) # 
    ax.grid(False)    # Hide grid lines
    ax.axis('off')

    plt.tight_layout(h_pad=0, w_pad=0)
    # plt.savefig(args.save_dir_res+fileName+'_TSNE_v2.png', dpi = 200)
    
    # finally, show the plot
    plt.show()

def save_TSNE_results(args, tsne_x, tsne_y, labels):
    tsne_gl = pd.DataFrame()
    tsne_non_gl = pd.DataFrame()
    count=0; count2=0
    
    for i in range(len(tsne_x)):
        print("{}/{}".format(i,len(tsne_x)))
        if labels[i]==0:
            count = count + 1
            tsne_non_gl.loc[count, 'x'] = tsne_x[i]
            tsne_non_gl.loc[count, 'y'] = tsne_y[i]
            
        elif labels[i]==1:
            count2 = count2 + 1
            tsne_gl.loc[count2, 'x'] = tsne_x[i]
            tsne_gl.loc[count2, 'y'] = tsne_y[i]

    tsne_non_gl.to_csv(args.save_dir_res+'tsn_non-glaucoma.csv', index=False, sep=',')
    tsne_gl.to_csv(args.save_dir_res+'tsn_glaucoma.csv', index=False, sep=',')


def visualize_tsne(args, fileName, tsne, labels, plot_size=1000, max_image_size=100):  # images, 
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    
    save_TSNE_results(args, tx, ty, labels)

    # visualize the plot: samples as colored points
    colors_per_class = ['darkblue', 'firebrick']
    visualize_tsne_points(args, fileName, tx, ty, labels, colors_per_class)
    

def TSNE_visualization(args, fileName, dataloader, model):
    fix_random_seeds()
    features, labels, _ = get_features(args, fileName, dataloader, 2, False, model) # loads a trained model

    tsne = TSNE(n_components=2).fit_transform(features)
    
    visualize_tsne(args, fileName, tsne, labels) # image_paths


def visualize_tsne_points_v2(args, tx, ty, labels, colors_per_class):
    # initialize matplotlib plot
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)

    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255

        # add a scatter plot with the correponding color and label
        ax.scatter(current_tx, current_ty, c=color, label=label, edgecolors = 'k', s = 100)

    # build a legend using the labels we set previously
    plt.legend(loc='upper right', fontsize=14)
    # ax.legend(loc='upper right', bbox_to_anchor= (0.5, 1), ncol=6, borderaxespad=0 , frameon=False) 
    
    plt.tight_layout(); #plt.legend(['Non-glaucoma', 'Glaucoma'], loc='upper right')
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    # plt.grid(color='k', linewidth=1, alpha=0.3)
    
    plt.savefig(os.path.join(args.save_dir_res, 'TSNE_data_discovery_scatter.png'), dpi = 400, bbox_inches='tight', pad_inches = 0.1)
    plt.show()
    
    
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


def save_to_file(featurs, labls, img_pats, t_sne):
    savetxt('features.csv', featurs, delimiter=',')

    df_labels = pd.DataFrame(labls)
    df_labels.to_csv('./labels.csv',index=False, header=None)

    df_image_paths = pd.DataFrame(img_pats)
    df_image_paths.to_csv('./image_paths.csv',index=False, header=None)

    savetxt('./tsne.csv', t_sne, delimiter=',')


from sklearn.metrics import pairwise_distances_argmin
import numpy as np

def inter_cluster_distance(X, labels):
    n_clusters = np.unique(labels).size
    cluster_centers = np.zeros((n_clusters, X.shape[1]))
    for i in range(n_clusters):
        cluster_centers[i] = X[labels == i].mean(axis=0)
    closest, _ = pairwise_distances_argmin(cluster_centers, X, axis=0, metric='euclidean')
    return np.mean([np.linalg.norm(cluster_centers[i] - X[closest[i]]) for i in range(n_clusters)])


    

def TSNE_visualization_data_discovery(args, dataloader, model):
    # ============================ All full-view images per dataset (not devided by tasks)
    # colors_per_class = {
    # 'Drishti-GS' : [254, 202, 87],
    # 'REFUGE' : [255, 107, 107],
    # 'MESSIDOR' : [10, 189, 227],
    # 'BinRushed' : [255, 159, 243],
    # 'Magrabi' : [100, 100, 255],
    # 'RWD' : [16, 172, 132],
    # }
    
    # ============================ Only classification: cropped images per classification dataset
    colors_per_class = {
    'Drishti-GS' : [254, 202, 87],
    'REFUGE' : [255, 107, 107],
    'RIM-ONE-DL' : [52, 31, 151],
    'RWD' : [16, 172, 132],
    }
    # ============================ Only segmentation: cropped images per segmentation dataset
    # colors_per_class = {
    # 'MESSIDOR' : [10, 189, 227],
    # 'BinRushed' : [255, 159, 243],
    # 'Magrabi' : [100, 100, 255],
    # 'RWD' : [16, 172, 132],
    # }
    # ============================
    
    fix_random_seeds()
    # ============================ calculate features
    features, labels, img_paths = get_features(args, '', dataloader, 1000, True, model) # uses random model

    # # ============================ calculate T-SNE projections    
    tsne = TSNE(n_components=2).fit_transform(features)
    
    # # ==========================>>> save features, labels, img_paths, and tsne into file 
    save_to_file(features, labels, img_paths, tsne)
    
    # ==========================>>> instead of computing features, labels, and img_paths, we read them from file
    # df_features = pd.read_csv('./features.csv', header=None)
    # features = df_features.to_numpy()
    
    # df_labels = pd.read_csv('./labels.csv', header=None)
    # labels = df_labels[0].values.tolist()
    
    
    # df_image_paths = pd.read_csv('./image_paths.csv', header=None)
    # img_paths = df_image_paths[0].values.tolist()
    
    # df_tsne = pd.read_csv('./tsne.csv', header=None)
    # tsne = df_tsne.to_numpy()
    # ============================
    
    print(f'Features shape = {features.shape}, labels length = {len(labels)}')
    
    # ============================ Rotate t-sne projections for a better visualization
    tsne = rotation(tsne, np.pi/2.3)

    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]    
    
    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    
    
    # fig, ax = plt.subplots(2,1,figsize=(10,7));
    
    # visualize the plot: samples as image
    visualize_tsne_images(args, tx, ty, img_paths, labels, colors_per_class, plot_size=1800, max_image_size=128)

    visualize_tsne_points_v2(args, tx, ty, labels, colors_per_class)
    
    

