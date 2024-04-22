#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 16:58:50 2023

@author: homai
"""

import os
import argparse
from tqdm import tqdm
from numpy import savetxt
import matplotlib.ticker as mticker
import cv2
import torch
import random
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from resnet import*

import torch
import torchvision
import torchvision.transforms as transforms

from tsne_custom_datasetloader import *
from torch.utils.data import DataLoader
from networks import*

import warnings
warnings.filterwarnings("ignore")

def annotateOnFigure_v2(string, colr, x, y, fSiz, edgecolr, ax):
    ax.annotate(string, 
                 color=colr, xy=(x, y), xycoords='axes fraction', xytext=(-10, 10), 
                 textcoords='offset pixels', horizontalalignment='center',
                 verticalalignment='center', fontsize=fSiz,
                 bbox=dict(facecolor='none', edgecolor=edgecolr, pad=6, linewidth =0.5, alpha=0.5) # boxstyle='round,pad=1',
                 )   

def visualize_tsne_points_v2(tsne_df, colors_per_class, ax, alpha):
    labl_colr = {}
    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(tsne_df['dataset']) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tsne_df['x_feat'], indices)
        current_ty = np.take(tsne_df['y_feat'], indices)

        # convert the class color to matplotlib format: BGR -> RGB, divide by 255, convert to np.array
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255

        labl_colr[label] = colors_per_class[label]
        # add a scatter plot with the correponding color and label
        ax.scatter(current_tx, current_ty, c= color, label=label, edgecolors = 'k', s = 130, alpha= alpha)
    
    # annotateOnFigure_v2(label, 'k', 0.12 , 0.87, 15, 'k', ax)
    ax.set_xlim([-0.02,1.02]); ax.set_ylim([-0.02,1.02])
    # ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=23)
    # ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=23)
    
    x_ticks = np.array([0, 0.2, 0.4, 0.6, 0.8, 1]) 
    y_ticks = np.array([0.2, 0.4, 0.6, 0.8, 1])
    ax.xaxis.set_major_locator(mticker.FixedLocator(x_ticks))
    ax.yaxis.set_major_locator(mticker.FixedLocator(y_ticks))

    # Set x-axis and y-axis tick labels with fixed decimal places
    x_tick_labels = ['0' , '0.2', '0.4', '0.6', '0.8', '1']
    y_tick_labels = ['0.2', '0.4', '0.6', '0.8', '1']
    ax.set_xticklabels(x_tick_labels, fontsize=23)
    ax.set_yticklabels(y_tick_labels, fontsize=23)
    
    plt.grid(False)
    return labl_colr
# =============================================================== # ===============================================================

def scale_image(image, max_image_size):
    image_height, image_width, _ = image.shape

    scale = max(1, image_width / max_image_size, image_height / max_image_size)
    image_width = int(image_width / scale)
    image_height = int(image_height / scale)

    image = cv2.resize(image, (image_width, image_height))
    return image


def draw_rectangle_by_class(image, colr):
    image_height, image_width, _ = image.shape

    # get the color corresponding to image class
    color = colr
    image = cv2.rectangle(image, (0, 0), (image_width - 1, image_height - 1), color=color, thickness=6)
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


def visualize_tsne_images(save_dir_res, df, show_in_color, colors_per_class, ax, blur, plot_size=1000, max_image_size=100):
    if show_in_color == 'train':
        col2 = 'test'
    elif show_in_color == 'test':
        col2 = 'train'
    
    df1 = df[df['data_split']==show_in_color];  
    df2 = df[df['data_split']==col2]
    
    offset = max_image_size // 2 # we'll put the image centers in the central area of the plot and use offsets to make sure the images fit the plot
    image_centers_area_size = plot_size - 2 * offset
    tsne_plot = 255 * np.ones((plot_size, plot_size, 3), np.uint8)
    
    # now we'll put a small copy of every image to its corresponding T-SNE coordinate
    for image_path, label, df2, x, y in tqdm(
            zip(df2['img_dir'], df2['dataset'], df2['data_split'], df2['x_feat'],  df2['y_feat']),
            desc='Building the T-SNE plot',
            total=len(df2['img_dir'])
            ):   
        image = cv2.imread(image_path) # image_path: is a same file as annotate.csv having path + fileName.png

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)        # Convert the image to grayscale while preserving shape
        pale_gray_image = cv2.convertScaleAbs(gray_image, alpha=0.5, beta=100) # Adjust contrast and brightness
        image = np.expand_dims(pale_gray_image, axis=2)    # Expand the grayscale image to 3 channels
        image = np.repeat(image, 3, axis=2)
        # icolr = [192, 192, 192]
        icolr = colors_per_class[label]

        image = scale_image(image, max_image_size) # scale the image to put it to the plot
        image = draw_rectangle_by_class(image, icolr) # draw a rectangle with a color corresponding to the image class

        tl_x, tl_y, br_x, br_y = compute_plot_coordinates(image, x, y, image_centers_area_size, offset) # compute the coordinates of the image on the scaled plot visualization
        tsne_plot[tl_y:br_y, tl_x:br_x, :] = image # put the image to its TSNE coordinates using numpy subarray indices 


    # now we'll put a small copy of every image to its corresponding T-SNE coordinate
    for image_path, label, df1, x, y in tqdm(
            zip(df1['img_dir'], df1['dataset'], df1['data_split'], df1['x_feat'],  df1['y_feat']),
            desc='Building the T-SNE plot',
            total=len(df1['img_dir'])
            ):   
        image = cv2.imread(image_path) # image_path: is a same file as annotate.csv having path + fileName.png
        icolr = colors_per_class[label]

        image = scale_image(image, max_image_size) # scale the image to put it to the plot
        image = draw_rectangle_by_class(image, icolr) # draw a rectangle with a color corresponding to the image class

        tl_x, tl_y, br_x, br_y = compute_plot_coordinates(image, x, y, image_centers_area_size, offset) # compute the coordinates of the image on the scaled plot visualization
        tsne_plot[tl_y:br_y, tl_x:br_x, :] = image # put the image to its TSNE coordinates using numpy subarray indices
    
    # annotateOnFigure_v2(label, 'k', 0.12 , 0.87, 15, 'k', ax)
    ax.imshow(tsne_plot[:, :, ::-1])
    plt.tight_layout()

    # ax.set_xticks([]); ax.set_yticks([]); 
    ax.set_xlim([-36,plot_size+36]); ax.set_ylim([-36,plot_size+36])
    
    x_ticks = np.array([0, 0.2, 0.4, 0.6, 0.8, 1]) * plot_size # np.arange(0, 1.2, 0.2) * plot_size
    y_ticks = np.array([0.2, 0.4, 0.6, 0.8, 1]) * plot_size # np.arange(0, 1.2, 0.2) * plot_size # np.arange(0, 1, 0.2)   * plot_size
    ax.xaxis.set_major_locator(mticker.FixedLocator(x_ticks))
    ax.yaxis.set_major_locator(mticker.FixedLocator(y_ticks))

    # Set x-axis and y-axis tick labels with fixed decimal places
    x_tick_labels = ['0' , '0.2', '0.4', '0.6', '0.8', '1']
    y_tick_labels = ['0.2', '0.4', '0.6', '0.8', '1'] # np.arange(0, 1.2, 0.2) # np.arange(1, 0, -0.2)
    # x_tick_labels_formatted = [f"{label:.1f}" for label in x_tick_labels]
    # y_tick_labels_formatted = [f"{label:.1f}" for label in y_tick_labels]
    ax.set_xticklabels(x_tick_labels, fontsize=23)
    ax.set_yticklabels(y_tick_labels, fontsize=23)
    plt.grid(False)


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


def horizontal_flip(vector):
    flip_mx = np.array([[-1, 0], [0, 1]])
    flipped_vect = flip_mx @ vector.T
    flipped_vect = flipped_vect.T
    return flipped_vect

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

def create_legend(label_colors_dict, num_columns=2, **kwargs):
    # Extract the labels and colors from the dictionary
    labels = list(label_colors_dict.keys())
    # colors = [(color*255).astype(int).tolist()[0][::-1] for color in label_colors_dict.values()]
    colors = [np.array(colr[::-1], dtype=float) / 255 for colr in label_colors_dict.values()]

    # Create a custom legend
    handles = [plt.Line2D([], [], marker='o', markersize=17, markeredgecolor='k', color=color, linestyle='None') for color in colors]
    # legend = plt.legend(handles, labels, fontsize=15, bbox_to_anchor=(1.15, -0.1),  ncol=num_columns, **kwargs)
    legend = plt.legend(handles, labels, fontsize=23, bbox_to_anchor=(0.5, -0.1),  ncol=num_columns, **kwargs) 
    return legend


def args(args):
    parser = argparse.ArgumentParser(description = "TSNE results for glaucoma classification.")
    
    # parser.add_argument("--csv_dirc", dest="csv_dir", type=str, default='/Volumes/homa/Homa/Glaucoma_prediction/Real_World_Data_paper/tsne/revision/') # '../../saveDir/dataset/'    #  '../../train_test_val_dirs' 
    # parser.add_argument("--save_dir", dest="save_dir", type=str, default= '/Volumes/homa/Homa/Glaucoma_prediction/Real_World_Data_paper/tsne/revision/') 

    parser.add_argument("--csv_dirc", dest="csv_dir", type=str, default= '../../../saveDir/revision/') 
    parser.add_argument("--save_dir", dest="save_dir", type=str, default= '../../../saveDir/revision/') 
    
    parser.add_argument("--csv_fileName", dest="csv_fileName", type=str, default='randModel_all_RWD_Public.csv') # fileName should have *.csv at the end
    parser.add_argument("--image_resize", dest="img_size", type=int, default=128)
    
    parser.add_argument("--rot_tsne", dest="rot", type=int, default= 0)
    parser.add_argument("--hflip_tsne", dest="hflip", action='store_true') 
    
    parser.add_argument("--apply_grayScale", dest="grayScale", action='store_true') 
    
    return parser.parse_args(args[1:])
    


if __name__ == "__main__":
    
    args = args(sys.argv)  
    
    saveDir = os.path.join(args.save_dir, args.csv_fileName.replace('.csv',''))
    
    tsne_res = pd.read_csv(os.path.join(args.save_dir, f'{args.csv_fileName}'))
    
    if args.rot != 0:
        print('Rotating tsne coordinates.')
        tsne_res_rot = rotation(tsne_res[['x_feat', 'y_feat']], args.rot); tsne_res_rot.columns = ['x_feat', 'y_feat']
        tsne_res     = pd.concat([tsne_res_rot, tsne_res.iloc[:, 2:]], axis=1)        
        # ============================ scale and move the coordinates so they fit [0; 1] range
        tsne_res['x_feat'] = scale_to_01_range(tsne_res['x_feat'])
        tsne_res['y_feat'] = scale_to_01_range(tsne_res['y_feat'])
        
    
    if args.hflip:
        print('Flipping tsne coordinates horizontally.')
        vec          = tsne_res[['x_feat', 'y_feat']].values
        tsne_flip    = pd.DataFrame(horizontal_flip(vec), columns=['x_feat', 'y_feat'])
        tsne_res     = pd.concat([tsne_flip, tsne_res.iloc[:, 2:]], axis=1)
        # ============================ scale and move the coordinates so they fit [0; 1] range
        tsne_res['x_feat'] = scale_to_01_range(tsne_res['x_feat'])
        tsne_res['y_feat'] = scale_to_01_range(tsne_res['y_feat'])
        
    fig, ax  = plt.subplots(2, 2, figsize=(18,18)); ax = ax.flatten()

    # ============================ All full-view images per dataset (not devided by tasks)
    colors_sharp = {
    'RWD'        : [16, 172, 132],
    'Drishti-GS' : [254, 202, 87],
    'REFUGE'     : [255, 107, 107],
    'MESSIDOR'   : [10, 189, 227],
    'BinRushed'  : [255, 159, 243],
    'Magrabi'    : [100, 100, 255],
    }
    
    colors_pale = {
    'RWD'        : [192, 192, 192],
    'Drishti-GS' : [192, 192, 192],
    'REFUGE'     : [192, 192, 192],
    'MESSIDOR'   : [192, 192, 192],
    'BinRushed'  : [192, 192, 192],
    'Magrabi'    : [192, 192, 192],
    }
    # ============================ Only classification: cropped images per classification dataset
    # colors_sharp = {
    # 'Drishti-GS' : [254, 202, 87],
    # 'REFUGE'     : [255, 107, 107],
    # 'RIM-ONE DL' : [52, 31, 151],
    # 'RWD'        : [16, 172, 132],
    # }
    
    # colors_pale = {
    # 'Drishti-GS' : [192, 192, 192],
    # 'REFUGE'     : [192, 192, 192],
    # 'RIM-ONE DL' : [192, 192, 192],
    # 'RWD'        : [192, 192, 192],
    # }
    # ============================ get train and test tsne results separately
    # tsne_res = tsne_res.iloc[100:300]
    
    train = tsne_res[tsne_res['data_split'] == 'train'] 
    test  = tsne_res[tsne_res['data_split'] == 'test'] 

    # ============================ visualize image space plots for the train and test tsne results separately    
    visualize_tsne_images(saveDir, tsne_res, 'train', colors_sharp, ax[0], False, plot_size=1800, max_image_size=128)
    visualize_tsne_images(saveDir, tsne_res,  'test', colors_sharp, ax[1], False, plot_size=1800, max_image_size=128)

    # ============================ visualize scatter plots for the train and test tsne results separately    
    visualize_tsne_points_v2(test, colors_pale, ax[2], 0.6)
    labl_colr = visualize_tsne_points_v2(train, colors_sharp,  ax[2], 1)
    
    visualize_tsne_points_v2(train, colors_pale, ax[3], 0.6)
    visualize_tsne_points_v2(test, colors_sharp, ax[3], 1)

    labl_colr['Train\Test'] = [192, 192, 192] 
    plt.tight_layout(w_pad=4, h_pad=4)

    # Create the legend using the function
    create_legend(labl_colr, 3)
    plt.savefig(saveDir+'_final.png', dpi = 350, bbox_inches='tight', pad_inches = 0.2)
    
    
    
    
    
    
    
