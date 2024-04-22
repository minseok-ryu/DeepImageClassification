#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 12:21:08 2023

@author: homai
"""

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

from resnet import*

import torch
import torchvision
import torchvision.transforms as transforms

from tsne_custom_datasetloader import *
from torch.utils.data import DataLoader
from networks import*

import warnings
warnings.filterwarnings("ignore")


def get_dataloaders(arg, data_dir): 
    test_transf = transforms.Compose([transforms.ToTensor()])
    
    df = pd.read_csv(data_dir)
    
    data = CustomDataset(annot_fileName= df, img_size=arg.img_size, applyCLAHE=arg.CLAHE, grayScale = arg.grayScale, transform= test_transf)
    dataloader = DataLoader(data, batch_size=300, shuffle=False, num_workers=0)
    return dataloader


# =============================================================== # ===============================================================
def get_features(net_archit, model_dir, dataloader, in_channels, num_class, do_rand_model):
    # ----------------------------------------
    # initialize our implementation of ResNet
    if net_archit == 'resnet50':
        model = resnet50(in_channels=in_channels, out_features=num_class)
        model = ResNet50(model, model_dir, num_classes=num_class, do_rand_model=do_rand_model, pretrained=True) #argpars.net_archit ,
    elif net_archit == 'resnet101':
        model = resnet101(in_channels=in_channels, out_features=num_class)
        model = ResNet101(model, model_dir, num_classes=num_class, do_rand_model=do_rand_model, pretrained=True)
    # ----------------------------------------
   
    model.eval()

    # we'll store the features as NumPy array of size num_images x feature_size
    features = None; labels = []; image_paths = []; types = []; data_split = []; datasets = []
    
    for images, label, img_path, typ, set_name, data in tqdm(dataloader, desc='Running the model inference to get features'):
        labels += label.tolist()
        types  += typ
        image_paths += img_path
        data_split += set_name
        datasets   += data
        # ------------------- moving data to gpu, if available
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
    return features, labels, types, image_paths, data_split, datasets

# =============================================================== # ===============================================================
# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

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

def fix_random_seeds():
    seed = 87
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def save_res_datafram(saveDir, x_feat, y_feat, labls, typs, img_dirs, set_name, data):
    df = pd.DataFrame(columns = ['x_feat', 'y_feat', 'label', 'type', 'img_dir'])
    df['x_feat']  = x_feat
    df['y_feat']  = y_feat
    df['label']   = labls
    df['type']    = typs
    df['data_split'] = set_name
    df['dataset'] = data
    df['img_dir'] = img_dirs    
    df.to_csv(saveDir, sep=',', index=False)
# =============================================================== # ===============================================================
    
    
    
def args(args):
    parser = argparse.ArgumentParser(description = "TSNE results for glaucoma classification.")
    
    # parser.add_argument("--csv_dirc", dest="csv_dir", type=str, default='/Volumes/homa/Homa/Glaucoma_prediction/Real_World_Data_paper/GL_classification/Public_Public/dataset/') # '../../saveDir/dataset/'    #  '../../train_test_val_dirs' 
    # parser.add_argument("--save_dir", dest="save_dir", type=str, default= '/Volumes/homa/Homa/Glaucoma_prediction/Real_World_Data_paper/tsne/revision/comment_1/') 
    # parser.add_argument("--model_dir", dest="model_dir", type=str, default='/Volumes/homa/Homa/Glaucoma_prediction/Real_World_Data_paper/GL_classification/Public_Public/') # '../../saveDir/dataset/'    #  '../../train_test_val_dirs' 

    parser.add_argument("--csv_dirc", dest="csv_dir", type=str, default= '../../../tsne/acsv_files/') 
    parser.add_argument("--save_dir", dest="save_dir", type=str, default= '../../../saveDir/revision/') 
    parser.add_argument("--model_dir", dest="model_dir", type=str, default= '../../../tsne/acsv_files/models/') 
    
    parser.add_argument("--rand_model", dest="rand_model", action='store_false')
    
    parser.add_argument("--csv_fileName", dest="csv_fileName", type=str, default='.csv') # fileName should have *.csv at the end
    parser.add_argument("--image_resize", dest="img_size", type=int, default=128)
    parser.add_argument("--net_archit", dest="net_archit", type=str, default='resnet50')  
    
    parser.add_argument("--apply_CLAHE", dest="CLAHE", action='store_true')
    parser.add_argument("--apply_grayScale", dest="grayScale", action='store_true') 
    
    return parser.parse_args(args[1:])
    


if __name__ == "__main__":
    
    args = args(sys.argv)  
    
    print('=*'*50); print('The default is to use the off-the-shell models like resnet50 or resnet101.'); 
    print(f'=======>>>>>>>>> rand_model == {args.rand_model}'); print('=*'*50);  print()
    
    if args.grayScale:
        in_channels = 1
    else:
        in_channels = 3
    
    data_dir    = os.path.join(args.csv_dir, args.csv_fileName)
    dataloaders = get_dataloaders(args, data_dir)
    
    fix_random_seeds()
    # # ============================ calculate features
    if args.rand_model:
        print(f'rand_model = {args.rand_model} ==> we do not use trained models in the paper.')
        features, labels, types, img_paths, split_name, dataset = get_features(args.net_archit, args.model_dir, dataloaders, in_channels, 1000, True)
        out_f_name = 'randModel'
        
    elif args.rand_model == False:
        model_name = input("Enter the file name of the model that you want to test the model on with extension: ")
        model_dir  = os.path.join(args.model_dir, model_name)
        print(f'rand_model = {args.rand_model} ==> we will use trained models in the paper.')
        features, labels, types, img_paths, split_name, dataset = get_features(args.net_archit, model_dir, dataloaders, in_channels, 2, False)
        out_f_name = input('what experiment is this for (e.g., IODA_IODA)? ')
        
    # =========================== calculate T-SNE projections    
    tsne = TSNE(n_components=2).fit_transform(features)
    
    print(f'Features shape = {features.shape}, labels length = {len(labels)}')
    
    # ============================ Rotate t-sne projections for a better visualization
    # tsne = rotation(tsne, np.pi/2.3)

    # ============================ extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]    
    
    # ============================ scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    
    # ============================>>>>>> save the dataframe of results into file
    saveDir = os.path.join(args.save_dir, f'{out_f_name}_{args.csv_fileName}')
    save_res_datafram(saveDir, tx, ty, labels, types, img_paths, split_name, dataset)    



 





        


