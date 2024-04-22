import os
import sys
import csv
import cv2
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from func_utils import utils
from sklearn.metrics import auc, roc_curve
# from func_utils.performance_funcs import*


def accuracy(predictions, labels):
    # _, preds = torch.max(predictions, dim=1)
    return (torch.tensor(torch.sum(predictions == labels).item())) #/ len(preds)))*100

# def _cuda_enabled(model, model_dir):
#     if not torch.cuda.is_available():
#         print("Let's use CPU!")
#         # model.load_state_dict(torch.load(model_dir, map_location=lambda storage, loc: storage))
#         model.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))

#     else:
#         print('Loading the model on GPU')
#         model.load_state_dict(torch.load(model_dir))    
#     return model


def test_phase(argpars, dataloaders, model, model_name_dir, fileName, criterion, metrics, write_csv=False):

    print(' '); print('--------- Testing phase ---------'); print('='*60); 
    phase = "test"
    print(phase)
    
    # Loading trained model 
    model_dir = model_name_dir
    # model = _cuda_enabled(model, model_name_dir)

    if torch.cuda.is_available(): # and torch.cuda.device_count() > 1
        model = utils.load_model_weights(model, model_name_dir, map_location='gpu') 
    else:
        model = utils.load_model_weights(model, model_name_dir, map_location='cpu')

    print('loaded the model!')
    model.eval()


    # batch loss and accuracy 
    batch_loss = 0.0
    batch_wighted_ncorrect = 0
    inputs_size = 0
    counter = 0 
    
    field_names = [f'{phase}_losses', f'{phase}_accuracies'] + [f'{phase}_{m}' for m in metrics.keys()]
    batch_summary = {a: [] for a in field_names}

    print(field_names)
    
    result = pd.DataFrame()

    with open(os.path.join(argpars.save_dir_res,fileName+'_cleanTest.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        
    with torch.no_grad():
        for inputs, targets in dataloaders[phase]:
            counter = counter + 1
            # ------------------- moving data to gpu, if available
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()                
            
            outputs = model(inputs)
            
            # print(outputs)
            loss = criterion(outputs, targets)
            
            prob = F.softmax(outputs)
            y_pred = prob.data.cpu()
            # y_pred[y_pred>=0.5] = 1; y_pred[y_pred<0.5] = 0       
            y_pred = torch.argmax(y_pred, dim=1)
            
            # result['prob_col1'] = prob[:,0].cpu().numpy()
            # result['prob_col2'] = prob[:,1].cpu().numpy()
            # result['prediction'] = y_pred.numpy()
            # result['target'] = targets.cpu().numpy()
            # ------------------- batch loss and accuracy
            batch_loss += loss.item() * inputs.size(0)
            batch_wighted_ncorrect += accuracy(y_pred, targets.cpu())  #torch.sum(torch.max(outputs, dim=1)[1] == targets).item()
           
            print(counter, ': out of ', inputs.size(0), ' images! => batch acc = ', accuracy(y_pred, targets.cpu()).item())
            inputs_size += inputs.size(0)
                    
            # _, y_pred = torch.max(outputs, dim=1)
                    
            for name, metric in metrics.items():
                batch_summary[f'{phase}_{name}'].append(metric(targets.data.cpu().numpy(), y_pred.data.cpu().numpy()))
                if name == 'auroc':
                    # fpr, tpr, thresholds = roc_curve(targets.data.cpu().numpy(), prob.data.cpu().numpy().max(axis=1))
                    # AUROC = auc(fpr, tpr)
                    # batch_summary[f'{phase}_{name}'].append(AUROC)
                    batch_summary[f'{phase}_{name}'].append(metric(targets.data.cpu().numpy(), prob.data.cpu().numpy()[:,1]))
                    
                    # batch_summary[f'{phase}_{name}'].append(metric(targets.data.cpu().numpy(), prob.data.cpu().numpy().max(axis=1)))
            # -------------------------------testing conf matrix code 
            # confusionMatrix(targets.data.cpu().numpy(), y_pred.data.cpu().numpy(), argpars.save_dir_res+fileName)        
    
    # ------------------- epoch loss and accuracy
    test_loss = float(batch_loss/inputs_size)
    test_acc = float(batch_wighted_ncorrect.double()/inputs_size)
    
    batch_summary[f'{phase}_losses'] = test_loss 
    batch_summary[f'{phase}_accuracies'] = test_acc
    
    for field in field_names[2:]:
        batch_summary[field] = np.mean(batch_summary[field])
    
    print(batch_summary)
    print("Test: loss: {:.3f} acc: {:.3f}.".format(test_loss, test_acc))

    # result.to_csv(os.path.join(argpars.save_dir_res,fileName+'_for_CI_calculation_cleanTest.csv'), sep=',', index=False)
    with open(os.path.join(argpars.save_dir_res,fileName+'_cleanTest.csv'), 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writerow(batch_summary)




        