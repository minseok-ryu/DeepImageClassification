import os
import sys
import csv
import cv2
import copy
import time
import torch
#torch.backends.cudnn.benchmark = True
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F

from func_utils import utils

def accuracy(predictions, labels):
    # _, preds = torch.max(predictions, dim=1)
    return (torch.tensor(torch.sum(predictions == labels).item())) #/ len(preds)))*100


# def _cuda_enabled(model):
#     if torch.cuda.is_available(): #and torch.cuda.device_count() ==1
#         print("Let's use", torch.cuda.device_count(), "GPU!")
#         model = model.cuda()
        
#     elif torch.cuda.device_count() > 1:
#         print("Let's use", torch.cuda.device_count(), "GPUs!")
#         # model = model.cuda()
#         model = nn.DataParallel(model)
#     else:
#         return model
#     return model


def call_load_model(model, model_dir):
    if torch.cuda.is_available(): # and torch.cuda.device_count() > 1
        model = utils.load_model_weights(model, model_dir, map_location='gpu') 
    else:
        model = utils.load_model_weights(model, model_dir, map_location='cpu')
    return model


def train_validation_phase(dataloaders, model, model_name, criterion, optimizer, argpars, loading_model, best_acc, metrics, plot=False, write_csv=True):

    field_names = ['epoch', 'train_losses', 'val_losses', 'train_accuracies', 'val_accuracies'] + [f'train_{m}' for m in metrics.keys()] + [f'val_{m}' for m in metrics.keys()] + ['best_model_epoch_number', 'min_train_loss', 'min_val_loss', 'max_train_acc', 'max_val_acc', 'nEpoch', 'early_stopping', 'batch_size_train', 'lr', 'weight_decay', 'augment', 'img_size','CLAHE', 'grayscale', 'GaussianBlur','loss_func', 'opt_solver', 'net_archit', 'repeat_train_data',]

    with open(argpars.save_dir_res+model_name+'_v2.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        
        
    best_loss = float("inf")
    # best_acc = 0.0
    best_acc_threshold = 0
    best_acc_train = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    unimproved_epochs = 0

    min_epochs = 100

    print('Start training the model!')
    # graph plotting variables
    train_accuracies, val_accuracies = [], []
    train_losses, val_losses = [], []
    best_model_epoch_number = 0

    start_time = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); print(device)

    if loading_model:
        print('========= In '+argpars.phase+': Loading model: '+ model_name+' =========')
        model = call_load_model(model, argpars.save_dir_modl+model_name)
        # model.load_state_dict(torch.load(argpars.save_dir_modl+model_name))
    
    for epoch in range(argpars.num_epochs):
        print("\nEpoch {}/{}: ".format(epoch+1, argpars.num_epochs), end=""); print('\n')
        
        batch_summary = {a: [] for a in field_names}

        # each epoch has a training and validation phase
        for phase in ["train", "val"]:
            # setting model phase
            model.train()
            if phase == "val": model.eval()

            # batch loss and accuracy 
            batch_loss = 0.0
            batch_accuracy = 0
            inputs_size = 0
            i =0
            
            for inputs, targets in dataloaders[phase]:
                i = i +1
                
                # ------------------- moving data to gpu, if available
                if torch.cuda.is_available():
                    inputs, targets = inputs.cuda(), targets.cuda()
                    # plt.imshow(inputs)
                # ------------------- reshaping targets to match model output
                # inputs = inputs.float()
                # targets = targets.float()
                # targets = targets.unsqueeze(1)
                
                # ------------------- clearning gradients
                optimizer.zero_grad()
                
                # ------------------- track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    # computing output and loss
                    outputs = model(inputs)
                    # outputs = torch.sigmoid(outputs)
                    loss = criterion(outputs, targets)
                    # predictions = torch.round(outputs)
                                        
                    prob = F.softmax(outputs)
                    y_pred = prob.data.cpu()
                    # y_pred[y_pred>=0.5] = 1; y_pred[y_pred<0.5] = 0       
                    y_pred = torch.argmax(y_pred, dim=1)

                    for name, metric in metrics.items():
                        batch_summary[f'{phase}_{name}'].append(metric(targets.data.cpu().numpy(), y_pred.data.cpu().numpy()))

                    if i%60==0:
                        print('Read new batch of data!')
                        print(i, "/", len(dataloaders[phase]), " - training: input size", inputs.size(), ", output_size", outputs.size(), ", label size", targets.size()) #, 'label unsqueeze:', labels.unsqueeze(1).size()
                        
                # ------------------- backward pass + optimize (only for training phase)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # ------------------- batch loss and accuracy
                batch_loss += loss.item() * inputs.size(0)
                batch_accuracy += accuracy(y_pred, targets.cpu()) #torch.sum(torch.max(outputs, dim=1)[1] == targets).item()
                inputs_size += inputs.size(0)

            # ------------------- epoch loss and accuracy
            epoch_loss = batch_loss/inputs_size
            epoch_accuracy = batch_accuracy.double()/inputs_size
            # print(f"{phase} loss: {epoch_loss:.3f} acc: {epoch_accuracy:.3f}. ", end=""); print(' ')
            # print(f"Time taken: {time.time() - start_time:.3f} seconds")

            batch_summary['epoch'] = epoch
            batch_summary[f'{phase}_losses'] = epoch_loss 
            batch_summary[f'{phase}_accuracies'] = epoch_accuracy.item() 

            # model checkpoint. mointoring best accuracy
            if phase == "val" and epoch_accuracy >= best_acc:
                print(" *----------------> saving model.", end="")
                best_acc = epoch_accuracy
                #best_acc_threshold = best_batch_threshold
                best_acc_train = train_accuracies[-1]
                best_loss = epoch_loss
                best_model_epoch_number = epoch
                best_model_wts = copy.deepcopy(model)
                unimproved_epochs = 0
                
                if epoch % 200 == 0:
                    utils.save_model(best_model_wts, argpars.save_dir_modl+model_name)
                # torch.save(best_model_wts, argpars.save_dir_modl+model_name)    # change line 155 to best_model_wts = copy.deepcopy(model.state_dict()): Saving the trained model parameters 

            elif phase == "val" and not (epoch_accuracy > best_acc):
                unimproved_epochs += 1
            
            # values for generating training graphs
            if(phase == "train"):
                train_accuracies.append(epoch_accuracy)
                train_losses.append(epoch_loss)
            else:
                val_accuracies.append(epoch_accuracy)
                val_losses.append(epoch_loss)
                
            
        for field in field_names[5:]:
            batch_summary[field] = np.mean(batch_summary[field])      
        
        # Writing train results to csv file
        if(write_csv): 
            batch_summary['best_model_epoch_number'] = best_model_epoch_number
            
            batch_summary['min_train_loss'] = np.min(train_losses)
            batch_summary['min_val_loss'] = np.min(val_losses)
            batch_summary['max_train_acc'] = np.max(np.array(best_acc_train))
            batch_summary['max_val_acc'] = np.max(np.array(best_acc))

            batch_summary['nEpoch'] = argpars.num_epochs
            batch_summary['early_stopping'] = argpars.early_stopping
            batch_summary['batch_size_train'] = argpars.batch_size_train
            batch_summary['lr']= argpars.lr
            batch_summary['weight_decay'] = argpars.weight_decay
            batch_summary['augment'] = argpars.augnment
            batch_summary['img_size'] = argpars.img_size
            batch_summary['CLAHE'] = argpars.CLAHE
            batch_summary['grayscale'] = argpars.grayScale
            batch_summary['GaussianBlur'] = argpars.gaussianBlur
            batch_summary['loss_func'] = 'CrossEntropy'
            batch_summary['opt_solver'] = 'Adam'
            batch_summary['net_archit'] = argpars.net_archit
            batch_summary['repeat_train_data'] = argpars.repeat_train_data    
            
            # results.to_csv(argpars.save_dir_res+model_name+'.csv', sep=',', index=False)

        if epoch % 200 == 0:
            with open(argpars.save_dir_res+model_name+'_v2.csv', 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=field_names)
                writer.writerow(batch_summary)

        # early stopping
        if(unimproved_epochs >= argpars.early_stopping and epoch > min_epochs):  
            print(f"\nEarly stopping. val loss not improved for {argpars.early_stopping} epochs")
            return 0
        # unimproved_epochs += 1
        print('unimproved_epochs= ', unimproved_epochs)

    print(f"Time taken: {time.time() - start_time:.3f} seconds")
    print(f"Val: loss: {best_loss:.3f} acc: {best_acc:.3f}.")
    

    with open(argpars.save_dir_res+model_name+'_v2.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writerow(batch_summary)
    utils.save_model(best_model_wts, argpars.save_dir_modl+model_name)
    # save_model(best_model_wts, argpars.save_dir_modl, model_name)



