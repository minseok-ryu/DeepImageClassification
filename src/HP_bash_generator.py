#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 15:57:38 2023

@author: homai
"""

import csv
import random
import numpy as np


def get_Glaucoma_bash_script(instance):
    learning_rate   = [0.001, 0.003, 0.0001]
    epochs          = 2000
    early_stop      = 500
    img_size        = 128
    batch_train     = [16, 32, 64, 128]
    model_name      = ['resnet50', 'resnet101']
    wd              = [0, 0.0005, 0.0001, 0.00005, 0.00001]
    phase           = 'train'
    
    str_list = []
    with open(instance+'.sh', 'w', newline='') as csvfile:
      spamwriter = csv.writer(csvfile, delimiter=',')  
        
      for ii in range(15):
        lr             = np.random.choice(learning_rate)
        num_epochs     = epochs
        early_stopping = early_stop
        image_resize   = img_size
        bs_train       = np.random.choice(batch_train)
        
        p_CLAHE              = random.random()
        p_apply_grayScale    = random.random()
        p_apply_gaussianBlur = random.random()
        p_augmnt             = random.random()

        net_archit     = np.random.choice(model_name)
        weight_decay   = np.random.choice(wd)
        phase          = phase
     
        string_out = 'python main.py --lr {} --num_epochs {} --early_stopping {} --image_resize {} --bs_train {} --net_archit {} --weight_decay {} --phase {}'.format(lr, num_epochs, early_stopping, image_resize, bs_train, net_archit, weight_decay, phase)
     
        if p_CLAHE > 0.7:
            apply_CLAHE = ' --apply_CLAHE'
            string_out += apply_CLAHE 
            
        if p_apply_grayScale > 0.7:
            apply_grayScale = ' --apply_grayScale'
            string_out += apply_grayScale
            
        if p_apply_gaussianBlur > 0.7:
            apply_gaussianBlur = ' --apply_gaussianBlur'
            string_out += apply_gaussianBlur
            
        if p_augmnt > 0.5:
            augmnt = ' --augmnt'
            string_out += augmnt

        str_list.append(string_out)
    
      for line in str_list:
          csvfile.write(line)
          csvfile.write('\n')

    
if __name__ == "__main__":    
    intsance = 'PublicIODA_Public_exps'
    get_Glaucoma_bash_script(intsance)
        


