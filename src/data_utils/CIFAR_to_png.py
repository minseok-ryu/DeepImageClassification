#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 19:02:40 2021

@author: homai
"""

from __future__ import print_function
import os 
import cv2
import pickle
# import _pickle as cPickle
import glob
import numpy as np
import pandas as pd

# ============================================================================*
# reading_dir = '/Users/homai/Desktop/AIO_Darvin/3.Classification_PyTorch/data/cifar-10-batches-py/'
# saving_dir = '/Users/homai/Desktop/AIO_Darvin/0.Research/data/'

# X = load_cifar_pickle(reading_dir+'data_batch_1')
# height, width = 32, 32
# X = np.shape(height, width, 3)
# io.imwrite(saving_dir+'/file.png', X)
# ============================================================================*

def load_cifar_pickle(path, file):
    f = open(os.path.join(path, file), 'rb')
    # dict = pickle.load(f)
    # f.close()
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    dict = u.load()
    
    images = dict['data']
    images = np.reshape(images, (10000, 3, 32, 32))
    labels = np.array(dict['labels'])
    print("Loaded {} labelled images.".format(images.shape[0]))
    return images, labels 


def load_cifar_categories(path, file):
    f = open(os.path.join(path, file), 'rb')
    dict = pickle.load(f)
    return dict['label_names']

def save_cifar_image(array, path):
    # array is 3x32x32. cv2 needs 32x32x3
    array = array.transpose(1,2,0)
    # array is RGB. cv2 needs BGR
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    # save to PNG file
    return cv2.imwrite(path, array)


def get_annotation_file(list_of_classes, pathId): # pathId = '../.././data/cifat10/'
    annot = pd.DataFrame(); idx = -1
    for iclass in list_of_classes:
        idx = idx + 1
        counter = -1
        this_class_annot = pd.DataFrame()
        for infile in glob.glob(pathId+iclass+'/' + '*.png'): # 
            counter = counter + 1
            this_class_annot.loc[counter, 'fileName'] = infile.replace(pathId,'')
            this_class_annot.loc[counter, 'label']    = idx
        
        print('Annotations for class: '+ iclass+ ' is done with a total #images = '+str(len(this_class_annot)))    
        annot = annot.append(this_class_annot, ignore_index=True)
    return annot

if __name__ == '__main__':
    base_dir= '../.././data/'
    # base_dir = '/Users/homai/Desktop/AIO_Darvin/3.Classification_PyTorch/data/'
    picke_name = 'data_batch_5'
    n_imgs = 10000

    images, labels = load_cifar_pickle(os.path.join(base_dir, 'cifar-10-batches-py'), picke_name)
    categories = load_cifar_categories(os.path.join(base_dir, 'cifar-10-batches-py'), "batches.meta")
    print(categories)
    
    # ---------------------- convert CIFAR10 to png and write each class in a separate file
    for i in range(0,n_imgs):
        cat = categories[labels[i]]
        out_dir = os.path.join(base_dir, 'cifar10', cat)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        save_cifar_image(images[i], os.path.join(out_dir, 'image_{}.png'.format(i)))        

    # ---------------------- get annotations file for all classes in a single .csv file
    # annotations = get_annotation_file(categories, base_dir+'cifar10/')
    # annotations.to_csv(base_dir+'cifar10/annotations.csv', sep=',', index=False)










