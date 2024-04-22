#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 11:04:56 2021

@author: homai
"""

import torch
import numpy as np 
import matplotlib.pyplot as plt 

# PyTroch version

SMOOTH = 1e-6 # epsilon

def iou(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.int()
    labels = labels.int()
    
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    labels  = labels.squeeze(1)
    
    outputs = outputs.detach()
    labels = labels.detach()
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    return iou.cpu().data.numpy().mean() # Or thresholded.mean() if you are interested in average across the batch


def sensitivity(outputs: torch.Tensor, labels: torch.Tensor):
    outputs = outputs.int()
    labels = labels.int()
    
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    labels  = labels.squeeze(1)
    
    outputs = outputs.detach()
    labels = labels.detach()
    
    tp = (outputs & labels).float().sum((1, 2)) # intersection == true positive
    fn = labels.float().sum((1, 2)) - tp  

    sensitivity = (tp + SMOOTH)/(tp+fn + SMOOTH)
    return sensitivity.data.numpy().mean()
    

def precision(outputs: torch.Tensor, labels: torch.Tensor):
    outputs = outputs.int()
    labels = labels.int()
    
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    labels  = labels.squeeze(1)
    
    outputs = outputs.detach()
    labels = labels.detach()
    
    tp = (outputs & labels).float().sum((1, 2)) # intersection == true positive
    fp = outputs.float().sum((1, 2)) - tp   

    precision = (tp + SMOOTH)/(tp+fp + SMOOTH)    
    return precision.data.numpy().mean()

    
def dice_score(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.int()
    labels = labels.int()
    
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    labels  = labels.squeeze(1)
    
    outputs = outputs.detach()
    labels = labels.detach()
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    mask_sum = union + intersection
    dice = 2 * (intersection + SMOOTH)/(mask_sum + SMOOTH)
    return dice.cpu().data.numpy().mean() # Or thresholded.mean() if you are interested in average across the batch
    
    
## -- in this function, we do not get the mean across the batch, but computing IoU per image
def get_iou_perImg(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.int()
    labels = labels.int()
    
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    labels  = labels.squeeze(1)
    
    outputs = outputs.detach()
    labels = labels.detach()
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    return iou.cpu().data.numpy() # Or thresholded.mean() if you are interested in average across the batch
        

def get_sensitivity_perImg(outputs: torch.Tensor, labels: torch.Tensor):
    outputs = outputs.int()
    labels = labels.int()
    
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    labels  = labels.squeeze(1)
    
    outputs = outputs.detach()
    labels = labels.detach()
    
    tp = (outputs & labels).float().sum((1, 2)) # intersection == true positive
    fn = labels.float().sum((1, 2)) - tp  

    sensitivity = (tp + SMOOTH)/(tp+fn + SMOOTH)
    return sensitivity.data.numpy()


def get_precision_perImg(outputs: torch.Tensor, labels: torch.Tensor):
    outputs = outputs.int()
    labels = labels.int()
    
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    labels  = labels.squeeze(1)
    
    outputs = outputs.detach()
    labels = labels.detach()
    
    tp = (outputs & labels).float().sum((1, 2)) # intersection == true positive
    fp = outputs.float().sum((1, 2)) - tp   

    precision = (tp + SMOOTH)/(tp+fp + SMOOTH)    
    return precision.data.numpy()

