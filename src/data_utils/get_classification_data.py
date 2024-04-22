#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 19:25:00 2021

@author: homai
"""

import os
import copy
import glob
import argparse
import itertools
import numpy as np
import pandas as pd
from pytz import timezone
from datetime import datetime
from tzlocal import get_localzone
from os import listdir,mkdir,rmdir
from os.path import join,isdir,isfile
import warnings; warnings.filterwarnings("ignore")

from src.data_utils.dataspliter import*


args = sys.argv

# Reading command line arguments into parser.
parser = argparse.ArgumentParser(description = "Glaucoma prediction via classification.")

# Filepaths
os.system('mount_smbfs //hrashi4:H19_O94_M19_ay@10.157.80.76/aio /Users/homai/Desktop/mntpoint')
parser.add_argument("--pData", dest="data_dir", type=str, default= '/Volumes/Homa/Homa/Glaucoma_segmentation/label_prop_filt_crop/6_data/filter_crop_margin90/') # '/Users/homai/Desktop/mntpoint/Datasets/FundusDiseases/Resized/'
parser.add_argument("--pAnnot", dest="annot_dir", type=str, default='/Volumes/Homa/Homa/Glaucoma_segmentation/label_prop_filt_crop/6_data/filter_crop_margin90/') # '/Users/homai/Desktop/mntpoint/Datasets/FundusDiseases/Resized/'

# parser.add_argument("--pData", dest="data_dir", type=str, default='../data/cropped_margin100/') #'../data/FundusDiseases/Resized/'
# parser.add_argument("--pAnnot", dest="annot_dir", type=str, default='../data/cropped_margin100/') #'../data/GlaucomaPrediction/OpticDisc/'

parser.add_argument("--validation_ratio", dest="val_Ratio", type=np.float32, default=0.083)   
parser.add_argument("--test_ratio", dest="test_Ratio", type=np.float32, default=0.1)    



num_classes=2
in_channels =3 # because I am using grayscale transform

net_number = 50 # ResNet50
valid_acc_max = 0

apply_CLAHE = False
apply_grayScale = False

# Creating Parser Object
opts = parser.parse_args(args[1:])
  
# # ------------------------ split data into train/val/test sets
train_annotations, val_annotations, test_annotations = Data_splitter_cropped(opts)
saveDataInFile(opts.data_dir, train_annotations, 'train')
saveDataInFile(opts.data_dir, val_annotations, 'val')
saveDataInFile(opts.data_dir, test_annotations, 'test')
