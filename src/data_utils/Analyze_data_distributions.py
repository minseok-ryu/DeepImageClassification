#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 23:24:02 2021

@author: homai
"""


import os
import sys
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from PIL import Image
from skimage import io

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from src.data_utils.custom_datasetloader import preprocess_CLAHE


def get_transforms(aument, apply_CLAHE, apply_grayscale, img_size):
    if aument:
        data_transform = transforms.Compose([
                                              transforms.Resize(img_size),
                                              transforms.Grayscale(num_output_channels=1),
                                              # transforms.functional.rgb_to_grayscale(num_output_channels: int = 3),
                                              preprocess_CLAHE(apply_CLAHE),
                                              transforms.RandomHorizontalFlip(),
                                              # transforms.RandomVerticalFlip(),
                                              transforms.GaussianBlur(kernel_size =(5,5), sigma=(0.1, 1.0)),
                                              transforms.RandomAffine(degrees=(0,0),translate=(0.10, 0.10)),# scale=(0.6,0.6)
                                              # transforms.RandomCrop(32, padding=4),
                                              transforms.ToTensor(),
                                              # normalize(mean, std)
                                              ])
    else:
        data_transform = transforms.Compose([
                                                transforms.Resize(img_size),
                                                transforms.ToTensor()])
    return data_transform


def getHistInfo_Data(root_dir, annotations_dir, img_size, transform):
    annotations = pd.read_csv(annotations_dir)
    all_images = []
    
    for idx in range(len(annotations)):
        img_path = os.path.join(root_dir, annotations.loc[idx, 'fileName'])    
        # img = io.imread(img_path) # transform does not work with this data structure ===> needs PIL object:
        img = Image.open(img_path).convert('RGB') #fig =plt.figure(); plt.imshow(img); plt.title('raw image: '+str(img.size)); trans_img_raw = self.transform(img); fig1 =plt.figure(); plt.hist(trans_img_raw[0,:,:], bins=50)
        # img.thumbnail((256,256,3), Image.ANTIALIAS)
        img= img.resize((img_size,img_size)) # fig1 =plt.figure(); plt.imshow(img); plt.title('Resized image to '+ str(img_size)+'x'+str(img_size))
        
        trans_img = transform(img) # fig2 =plt.figure(); plt.imshow(trans_img.permute(1, 2, 0)); plt.title('Transformed image')
        # fig = plt.figure(); plt.imshow(trans_img[1,:,:]);
        trans_img = trans_img[1,:,:].data.numpy().flatten()
    
        all_images.append(trans_img)
    
        if idx%200 == 0:
            print(idx, ' image!')
# bins, freqs, ptchs = plt.hist(trans_img[:,0,:])        
    return all_images



os.system('mount_smbfs //hrashi4:H19_O94_M19_ay@10.157.80.76/aio /Users/homai/Desktop/mntpoint')

root_dir = '/Users/homai/Desktop/mntpoint/Datasets/FundusDiseases/Resized/'
annotations_dir = root_dir + 'test_annotations.csv'

img_size = 224


transform = get_transforms(True, False, True , img_size)
all_images = getHistInfo_Data(root_dir, annotations_dir, img_size, transform)

all_images_flat = np.concatenate(all_images).ravel()

fig = plt.figure(); 
bins, freqs, ptchs = plt.hist(all_images_flat, alpha=0.5, color= 'blue')  
plt.title(annotations_dir.replace(root_dir,''))













