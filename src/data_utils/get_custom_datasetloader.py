#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 16:10:31 2021

@author: homai
"""


"a[:4] means only the first 4 elemnets!"
"a[4:] means from index=4 to the end of dataframe!"
"a[-4:] means only the last 4 indices!"

import sys
import os
import glob

import random
import numpy as np 
import pandas as pd
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from math import floor
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GroupShuffleSplit

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from data_utils.custom_datasetloader import CustomDataset
from data_utils.custom_datasetloader import preprocess_CLAHE

import warnings
warnings.filterwarnings("ignore")




"The main difference between this version of the code from the other previous version (get_custom_datasetloade.py) is that"
"here we split train/validation/test datasets based on patients to avoid information leakage between train and test."


def normalize(mean, std):
    return transforms.Normalize(mean= mean, std= std)


def get_transforms_needFolders(argparsr):
    # test_transform = transforms.Compose([transforms.Resize((argparsr.img_size, argparsr.img_size))]) # converts PILImage a FloatTensor of shape (C x H x W) with range [0.0, 1.0]
    test_transform = transforms.Compose([])                         
                    
    if argparsr.augnment:
        train_transform = transforms.Compose([
                                              # transforms.Resize((argparsr.img_size, argparsr.img_size)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomAffine(degrees=(0,0), translate=(0.2, 0.2)), # scale=(0.3,0.3)
                                              ]) 
    elif argparsr.augnment==False:
    #     train_transform = transforms.Compose([transforms.Resize((argparsr.img_size, argparsr.img_size))])
        train_transform = transforms.Compose([])    
    
    if argparsr.CLAHE:     
        train_transform.transforms.append(preprocess_CLAHE(argparsr.CLAHE))
        test_transform.transforms.append(preprocess_CLAHE(argparsr.CLAHE))
        
    if argparsr.gaussianBlur:
        train_transform.transforms.append(transforms.GaussianBlur(kernel_size =(3,3), sigma=(0.1, 1.0)))
        test_transform.transforms.append(transforms.GaussianBlur(kernel_size =(3,3), sigma=(0.1, 1.0)))
        
    if argparsr.grayScale:
        train_transform.transforms.append(transforms.Grayscale(num_output_channels=1))
        test_transform.transforms.append(transforms.Grayscale(num_output_channels=1))

    train_transform.transforms.append(transforms.ToTensor())
    test_transform.transforms.append(transforms.ToTensor())
    
    return train_transform, test_transform



def get_transforms(argparsr):
    test_transform = transforms.Compose([])
                                             
    if argparsr.augnment:
        train_transform = transforms.Compose([
                                              # transforms.Resize((argparsr.img_size, argparsr.img_size)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomAffine(degrees=(90,90), translate=(0, 0)), # scale=(0.3,0.3)
                                              ]) 
    elif argparsr.augnment==False:
        train_transform = transforms.Compose([])

    if argparsr.gaussianBlur:
        train_transform.transforms.append(transforms.GaussianBlur(kernel_size =(3,3), sigma=(0.1, 1.0)))
        test_transform.transforms.append(transforms.GaussianBlur(kernel_size =(3,3), sigma=(0.1, 1.0)))

    train_transform.transforms.append(transforms.ToTensor())
    test_transform.transforms.append(transforms.ToTensor())
    
    return train_transform, test_transform


def printClassRatios(train_annotations, val_annotations, test_annotations):
    print('Ratio of Glaucoma (label=1) images in train = ', str(np.round((len(train_annotations[(train_annotations['label'] == 1)])/len(train_annotations))*100,3)), '%')
    print('Ratio of Glaucoma (label=1) images in validation = ', str(np.round((len(val_annotations[(val_annotations['label'] == 1)])/len(val_annotations))*100,3)), '%')
    print('Ratio of Glaucoma (label=1) images in test = ', str(np.round((len(test_annotations[(test_annotations['label'] == 1)])/len(test_annotations))*100,3)), '%')
    
def dropColumns(train_annotations, val_annotations, test_annotations):
    train_annotations = train_annotations.drop(columns=['disease','type', 'validity','splitfileName']); train_annotations= train_annotations.reset_index(drop=True)
    val_annotations = val_annotations.drop(columns=['disease','type', 'validity','splitfileName']); val_annotations= val_annotations.reset_index(drop=True)    
    test_annotations = test_annotations.drop(columns=['disease','type', 'validity','splitfileName']); test_annotations= test_annotations.reset_index(drop=True)    
    return train_annotations, val_annotations, test_annotations


def getStatOnData(argpars, ntrain_loader, nvalid_loader, ntest_loader):
    train_annotations = pd.read_csv(argpars.annot_dir+'train_annotations.csv')
    val_annotations = pd.read_csv(argpars.annot_dir+'val_annotations.csv')
    test_annotations = pd.read_csv(argpars.annot_dir+'test_annotations.csv')

    printClassRatios(train_annotations, val_annotations, test_annotations)
    # train_annotations, val_annotations, test_annotations = dropColumns(train_annotations, val_annotations, test_annotations)

    print('train: batch size= ', str(argpars.batch_size_train), ', # batch= ', str(ntrain_loader), ', total # train data= ', str(len(train_annotations)))
    print('valid: batch size= ', str(argpars.batch_size_valid), ', # batch= ', str(nvalid_loader), ', total # valid data= ', str(len(val_annotations)))
    print('test: batch size= ', str(argpars.batch_size_test), ', # batch= ', str(ntest_loader), ', total # test data= ', str(len(test_annotations)))


def get_dataloaders_needFolders(argpars):
    train_transform, val_test_transform = get_transforms_needFolders(argpars)
    print(train_transform); print(val_test_transform)

    # ================= remove any file that its name starts with a dot
    train_files = glob.glob(os.path.join(argpars.annot_dir, "train", "glaucoma", '.*.png'))
    train_files += glob.glob(os.path.join(argpars.annot_dir, "train", "non_glaucoma", '.*.png'))
    
    val_files   = glob.glob(os.path.join(argpars.annot_dir, "val", "glaucoma", '.*.png'))
    val_files   += glob.glob(os.path.join(argpars.annot_dir, "val", "non_glaucoma", '.*.png'))

    test_files  = glob.glob(os.path.join(argpars.annot_dir, "test", "glaucoma", '.*.png'))    
    test_files  += glob.glob(os.path.join(argpars.annot_dir, "test", "non_glaucoma", '.*.png'))
    
    # Iterate over the image files and remove them
    for train_file in train_files:
        os.remove(train_file)

    for val_file in val_files:
        os.remove(val_file)

    for test_file in test_files:
        os.remove(test_file)
        
    # ================= 
    train_dataset = torchvision.datasets.ImageFolder(os.path.join(argpars.annot_dir, "train"), train_transform)
    valid_dataset = torchvision.datasets.ImageFolder(os.path.join(argpars.annot_dir, "val"), val_test_transform)
    test_dataset  = torchvision.datasets.ImageFolder(os.path.join(argpars.annot_dir, "test"), val_test_transform)

    # ------------------------ fixing unbalance data in classes by weighting them differently
    train_classes = train_dataset.classes
    train_classes_count = [train_dataset.targets.count(train_dataset.class_to_idx[train_class]) for train_class in train_classes]
    train_classes_ratio = (1 / torch.Tensor(train_classes_count)).double()
    train_classes_weights = np.array([train_classes_ratio[target] for target in train_dataset.targets])
    sampler = torch.utils.data.sampler.WeightedRandomSampler(train_classes_weights, len(train_classes_weights))

    # ------------------------ Creating dataloader object for train, val and test sets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=argpars.batch_size_train, sampler=sampler, num_workers=argpars.nworkers, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=argpars.batch_size_valid, shuffle=False, num_workers=argpars.nworkers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=argpars.batch_size_test, shuffle=False, num_workers=argpars.nworkers)

    getStatOnData(argpars, len(train_loader), len(valid_loader), len(test_loader))
    return {"train":train_loader, "val": valid_loader, "test": test_loader}
  
def get_dataloaders_test_needFolders(argpars):
    _, val_test_transform = get_transforms_needFolders(argpars)
    print(val_test_transform)
    
    test_dataset = torchvision.datasets.ImageFolder(os.path.join(argpars.annot_dir, "test_IODA"), val_test_transform)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=argpars.batch_size_test, shuffle=False, num_workers=argpars.nworkers)
    return {"test": test_loader}
  

def get_dataloaders_test(argpars):
    _, val_test_transform = get_transforms(argpars)
    print(val_test_transform)
    
    test_annotations = pd.read_csv(argpars.annot_dir+'test_annotations.csv')
    
    test_dataset  = CustomDataset(annot_fileName= test_annotations, img_size=argpars.img_size, applyCLAHE = argpars.CLAHE, grayScale=argpars.grayScale, transform= val_test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=argpars.batch_size_test, shuffle=False, num_workers=argpars.nworkers)
    return {"test": test_loader}


def get_dataloaders(argpars):
    train_transform, val_test_transform = get_transforms(argpars)
    print(train_transform); print(val_test_transform)

    train_annotations = pd.read_csv(argpars.annot_dir+'train_annotations.csv')
    val_annotations = pd.read_csv(argpars.annot_dir+'val_annotations.csv')
    test_annotations = pd.read_csv(argpars.annot_dir+'test_annotations.csv')
    
    # ------------------------ Repeat training data to increase the batch size
    train_annotations = pd.concat([train_annotations]*argpars.repeat_train_data)
    train_annotations = train_annotations.reset_index(drop=True)
    
    print('-'*50); print('train:', train_annotations.loc[0:10,'fileName'])
    print('test:', test_annotations.loc[0:10,'fileName']); print('-'*50)
    # ------------------------ get my custom dataset: u can either use my custom dataset or use pytorch
    train_dataset = CustomDataset(annot_fileName= train_annotations, img_size=argpars.img_size, applyCLAHE = argpars.CLAHE, grayScale=argpars.grayScale, transform= train_transform)
    valid_dataset = CustomDataset(annot_fileName= val_annotations, img_size=argpars.img_size, applyCLAHE = argpars.CLAHE, grayScale=argpars.grayScale, transform= val_test_transform)
    test_dataset  = CustomDataset(annot_fileName= test_annotations, img_size=argpars.img_size, applyCLAHE = argpars.CLAHE, grayScale=argpars.grayScale, transform= val_test_transform)

    # ------------------------ Creating dataloader object for train, val and test sets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=argpars.batch_size_train, num_workers=argpars.nworkers, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=argpars.batch_size_valid, shuffle=False, num_workers=argpars.nworkers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=argpars.batch_size_test, shuffle=False, num_workers=argpars.nworkers)

    getStatOnData(argpars, len(train_loader), len(valid_loader), len(test_loader))
    return {"train":train_loader, "val": valid_loader, "test": test_loader}



class aDataset(Dataset):
    def __init__(self, path, img_size, transform= None): #  duplicate=False,
        self.transform   = transform; print(self.transform)
        self.path        = path
        self.img_size    = img_size
        # self.duplicate   = duplicate
        self.image_paths = []
        self.targets     = []
        self.class_names = []
        
        all_files = os.listdir(path)
        all_files = [item for item in all_files if not item.startswith('.')]
        # all_files = ['non_glaucoma', 'glaucoma']

        for target, class_name in enumerate(all_files):
            if not class_name.startswith('.'):
                
                class_path = os.path.join(path, class_name)
                for image_name in os.listdir(class_path):
                    # ignores some database files created on windows. can later change so it supports only .png or .jpg etc.
                    if image_name != "Thumbs.db" and not image_name.startswith('.'): 
                        image_path = os.path.join(class_path, image_name)
                        self.image_paths.append(image_path)
                        self.targets.append(target)
                        self.class_names.append(class_name)
                        
        print(f'labels = {all_files}, labels length = {len(self.class_names)}, image paths length = {len(self.image_paths)}')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # can't use torchvision.io.read_image because it currently only supports rbg images
        # can't use opencv because pytorch transforms only takes PIL image or tensor as input 
        img = Image.open(self.image_paths[index]) 
        image = img.convert('RGB')
        image = image.resize((self.img_size, self.img_size))
        
        image = self.transform(image)
        target = self.targets[index]
        class_name = self.class_names[index]
        return image, class_name, self.image_paths[index]


def get_dataloaders_inference(argpars, data_dir): 
    test_transform = transforms.Compose([transforms.ToTensor()])
    
    data       = aDataset(data_dir, argpars.img_size, test_transform) 
    dataloader = DataLoader(data, batch_size=200, shuffle=False, num_workers=0)
    return {"infer":dataloader}
        
        
        
   # os.path.join(argpars.annot_dir, "train")     
        
        
        
        
        
        
        
        
        
        
        
        


    