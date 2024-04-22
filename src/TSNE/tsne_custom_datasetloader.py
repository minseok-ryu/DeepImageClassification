#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 21:02:21 2021

@author: homai
"""

import os
import sys
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage import io

import torch
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import ToTensor


class CustomDataset(Dataset):
    def __init__(self, annot_fileName: str, img_size, applyCLAHE:bool, grayScale:bool, transform = None):
        self.annotations = annot_fileName
        self.img_size = img_size
        self.applyCLAHE = applyCLAHE
        self.grayScale = grayScale
        self.transform = transform

    def preprocess_HistEqualization(self, img): 
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab_img) # b, g, r
        
        equ = cv2.equalizeHist(l) 
        # --------------- Combine CLAHE enhanced L-channel back with A and B channels
        updated_lab_img1 = cv2.merge((equ, a, b))
        
        # --------------- Convert LAB image back to color (RGB)
        hist_eq_img = cv2.cvtColor(updated_lab_img1, cv2.COLOR_LAB2BGR)         
        return hist_eq_img

    def preprocess_CLAHE_getitem(self, img):
        # don't expand default
        if(self.applyCLAHE == False):
            return img
        
        img = np.array(img) # img.convert('RGB')
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab_img) # L – Lightness ( Intensity ). a – color component ranging from Greb – color component ranging from Blue to Yellow.en to Magenta. 
        
        # --------------- Apply CLAHE to l channel
        clahe = cv2.createCLAHE(clipLimit = 3.0, tileGridSize = (8,8))
        clahe_img = clahe.apply(l)
        updated_lab_img2 = cv2.merge((clahe_img, a, b))
        # --------------- Convert LAB image back to color (RGB)
        # --------------- Combine CLAHE enhanced L-channel back with A and B channels
        CLAHE_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)   
        CLAHE_img = Image.fromarray(CLAHE_img)
        return CLAHE_img


    def __len__(self):
        return len(self.annotations)
     
    
    def __getitem__(self, idx):
       img_path = os.path.join(self.annotations.loc[idx, 'img_dir'], self.annotations.loc[idx, 'fileName'])     
       # folder_name = str(self.img_size)+'x'+str(self.img_size)
       # img_path = os.path.join(self.annotations.loc[idx, 'img_dir'], folder_name, self.annotations.loc[idx, 'fileName'])     
       # -------------------------- loed image
       if self.applyCLAHE==False and self.grayScale ==False:
           img = Image.open(img_path).convert('RGB') 
           img = img.resize((self.img_size,self.img_size))
     
       elif self.applyCLAHE and self.grayScale == False:
           img = Image.open(img_path).convert('RGB') # fig = plt.figure(); plt.imshow(img); plt.title('raw image: '+str(img.size))
           img = self.preprocess_CLAHE_getitem(img) 
           img = img.resize((self.img_size,self.img_size)) 
    
       elif self.applyCLAHE==False and self.grayScale: 
           img = Image.open(img_path).convert('L')   
           img = img.resize((self.img_size,self.img_size))
      
       elif self.applyCLAHE and self.grayScale:        
           img = Image.open(img_path).convert('RGB') 
           img = self.preprocess_CLAHE_getitem(img) 
           img = img.convert('L')
           img = img.resize((self.img_size,self.img_size))   
       
       if self.transform:
           img = self.transform(img) 
           
       y_label = self.annotations.loc[idx, 'label']
     
       return img, y_label, img_path, self.annotations.loc[idx, 'type'], self.annotations.loc[idx, 'data_split'], self.annotations.loc[idx, 'dataset']

      
class preprocess_CLAHE(object):
    def __init__(self, applyCLAHE):
        self.applyCLAHE = applyCLAHE

    def __call__(self, img):
        # don't expand default
        if(self.applyCLAHE == False):
            return img
        
        img = np.array(img) # img.convert('RGB')
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab_img) # L – Lightness ( Intensity ). a – color component ranging from Greb – color component ranging from Blue to Yellow.en to Magenta. 
        
        # --------------- Apply CLAHE to l channel
        clahe = cv2.createCLAHE(clipLimit = 3.0, tileGridSize = (8,8))
        clahe_img = clahe.apply(l)
        updated_lab_img2 = cv2.merge((clahe_img, a, b))
        # --------------- Convert LAB image back to color (RGB)
        # --------------- Combine CLAHE enhanced L-channel back with A and B channels
        CLAHE_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)   
        CLAHE_img = Image.fromarray(CLAHE_img)
        return CLAHE_img






    
    