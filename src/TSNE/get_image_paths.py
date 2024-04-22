#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 19:51:05 2023

@author: homai
"""


import os
import sys
import cv2

import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = 'IODA'

annot_path  = '/Volumes/homa/Homa/Glaucoma_prediction/Real_World_Data_paper/GL_classification/'+data+'_'+data+'/dataset/'
root_new_df = '/Volumes/homa/Homa/Glaucoma_prediction/Real_World_Data_paper/RW_data_inventory/classification/original/'+data+'/' 
saving_path = os.path.join(annot_path , 'img_paths_TSNE.csv')

# annot_path  = '../../../dataset/'
# root_new_df   = '../../../original/IODA/' 


df     = pd.read_csv(annot_path + 'annotations.csv')
new_df = pd.DataFrame()

for idx in range(len(df)):
    fileName = df.loc[idx, 'fileName']
    
    if df.loc[idx, 'label'] == 1: 
        string = os.path.join(root_new_df, 'glaucoma', fileName)
        print(string)
        new_df.loc[idx, 'dirs'] = string
    
    elif df.loc[idx, 'label'] == 0:    
        string = os.path.join(root_new_df, 'non-glaucoma', fileName)
        print(string)
        new_df.loc[idx, 'dirs'] = string

new_df.to_csv(saving_path, index=False, header=False)



# with open(saving_path, 'w') as f:
      
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#     write.writerow(alist)










