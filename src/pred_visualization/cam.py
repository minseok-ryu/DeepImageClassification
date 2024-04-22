import os
import sys
import cv2
import glob
import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision import models
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
import torchvision.transforms as transforms


SCRIPT_DIR = os.path.dirname(os.path.abspath('.'))
os.chdir(SCRIPT_DIR); print(SCRIPT_DIR)
print(os.getcwd()); print('='*60)
sys.path.append(SCRIPT_DIR)

from func_utils.utils import*
from architecture.networks import*

# from data_utils.get_custom_datasetloader import*


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')

    # parser.add_argument("--pAnnot", dest="annot_dir", type=str, default='/Volumes/Homa/Homa/Glaucoma_prediction/Real_World_Data_paper/GL_classification/IODA_IODA/dataset/')  
    # parser.add_argument("--pSaving_model", dest="save_dir_modl", type=str, default='/Volumes/Homa/Homa/Glaucoma_prediction/Real_World_Data_paper/GL_classification/IODA_IODA/prediction/models/') #'/Users/homai/Desktop/mntpoint/Projects/homa/Glaucoma_prediction/entire_images/models/7.grayScale/'
    # parser.add_argument("--pSaving_res", dest="save_dir_res", type=str, default='/Volumes/Homa/Homa/Glaucoma_prediction/Real_World_Data_paper/GL_classification/IODA_IODA/visualization_CAM/on_test_set/') 
    
    parser.add_argument("--pAnnot", dest="annot_dir", type=str, default='../../saveDir/dataset/') #    '../../saveDir/dataset/'
    parser.add_argument("--pSaving_model", dest="save_dir_modl", type=str, default='../../saveDir/prediction/models/') 
    parser.add_argument("--pSaving_res", dest="save_dir_res", type=str, default='../../saveDir/visualization_CAM/on_test_set/') 
    
    parser.add_argument('--aug_smooth', action='store_true',help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth',action='store_true',help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')
    
    parser.add_argument("--image_resize", dest="img_size", type=int, default=224)
    parser.add_argument("--apply_CLAHE", dest="CLAHE", action='store_true')
    parser.add_argument("--apply_grayScale", dest="grayScale", action='store_true') 
    parser.add_argument("--apply_gaussianBlur", dest="gaussianBlur", action='store_true')

    parser.add_argument("--separate_plot", dest="separate_plot", action='store_true')  
    parser.add_argument("--pred_mask", dest="pred_mask", action='store_true')    
    parser.add_argument("--phase", dest="phase", type=str, default='test')
    parser.add_argument("--net_archit", dest="net_archit", type=str, default='resnet50')  
    
    
    args = parser.parse_args()
    if torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.use_cuda =False
    print(args.use_cuda); 
    print(torch.cuda.is_available())
    print('='*40)
    
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')
        
    print(args)
    return args


def call_load_model(model, model_dir):
    if torch.cuda.is_available(): # and torch.cuda.device_count() > 1
        model = load_model_weights(model, model_dir, map_location='gpu') 
    else:
        model = load_model_weights(model, model_dir, map_location='cpu')
    return model

def str_to_class(str):
    return getattr(sys.modules[__name__], str)


if __name__ == '__main__':
    """ python cam.py -annot_dir <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    num_classes = 2
    in_channels =3 # because I am using RGB images
    net_number = 50
    class_names = ['glaucoma/', 'non_glaucoma/']  

    if args.grayScale:
        in_channels = 1
    print('in_channels: ================== ', in_channels)  

    net_archit = str_to_class(args.net_archit)
    model = net_archit(in_channels=in_channels, out_features=num_classes)
    # ----------------------- get the name of the model you want to load
    fileName = input("Enter the name of the model you want to load for visualization: "); print(fileName)
    
    model = call_load_model(model, args.save_dir_modl+fileName)
    print(model)
    
    
    # Choose the target layer you want to compute the visualization for. Usually this will be the last convolutional layer in the model.
    # Some common choices can be: Resnet18 and 50: model.layer4[-1], VGG, densenet161: model.features[-1], mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer. You can pass a list with several target layers, in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    
    if isinstance(model, (nn.DataParallel)):
        target_layers = [model.module.layer4[-1]]
    else:
        target_layers = [model.layer4[-1]]    
        

    # dataloaders = get_dataloaders_test(args)
    # for input_tensor, targets in dataloaders[args.phase]:
    idx = -1
    for iclass in class_names:
        for infile in glob.glob(args.annot_dir+ args.phase+'/'+iclass+ '*.png'):  
            idx = idx + 1
            img_name = infile.replace(args.annot_dir+ args.phase+'/'+iclass, '') 
        # ------------------------------------------===========================================================
            if args.grayScale:
                img = cv2.imread(infile, 0)  #[:, :, ::-1]
            else:
                img = cv2.imread(infile, 1)[:, :, ::-1]
        
            img = cv2.resize(img, (args.img_size,args.img_size))    
            img = np.float32(img) / 255
            
            if args.grayScale:
                input_tensor = preprocess_image(img,
                                                mean=[0.5],
                                                std=[0.2])                
            else:
                input_tensor = preprocess_image(img,
                                                mean=[0.5, 0.5, 0.5],
                                                std=[0.2, 0.2, 0.2])
            # ------------------------------------------===========================================================
            if iclass == 'non_glaucoma/':
                label = 0
            elif iclass == 'glaucoma/':
                label = 1
    
            # If None, returns the map for the highest scoring category.
            # Otherwise, targets the requested category.
            target_category = [label] #[1 for _ in range(len(input_tensor))] # None #[25] 
            if idx%500 ==0:
                print('Getting CAM for image ', idx, ': ', img_name, ', target_category=', target_category)
        
            # Using the with statement ensures the context is freed, and you can
            # recreate different CAM objects in a loop.
            cam_algorithm = methods[args.method]
            with cam_algorithm(model=model,
                               target_layers=target_layers,
                               use_cuda=args.use_cuda) as cam:
        
                # AblationCAM and ScoreCAM have batched implementations.
                # You can override the internal batch size for faster computation.
                cam.batch_size = 32
        
                grayscale_cam = cam(input_tensor=input_tensor,
                                    target_category=target_category,
                                    aug_smooth=args.aug_smooth,
                                    eigen_smooth=args.eigen_smooth)
        
                # Here grayscale_cam has only one image in the batch
                grayscale_cam = grayscale_cam[0, :]
                
                if args.grayScale:
                    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=False)
                else:
                    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        
                # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
                cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        
            gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
            gb = gb_model(input_tensor, target_category=target_category)
        
            cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
            cam_gb = deprocess_image(cam_mask * gb)
            gb = deprocess_image(gb)
        
        
            # -------------------------------------- Sanity check for 
            an = cam_mask; an =np.moveaxis(an, 2, 0)
            an_diff1 = (an[0,:,:] - an[1,:,:])
            an_diff2 = (an[0,:,:] - an[2,:,:])
            an_diff3 = (an[1,:,:] - an[2,:,:])
            if (np.unique(an_diff1) == 0) and (np.unique(an_diff2) == 0) and (np.unique(an_diff3) == 0):
                print('All 3 channels of cam_mask are equal!'); print('='*60)
            else:
                print('DANGEEEEEEEEEEERRRRRRRR: Do not save one channel of cam_mask! They are different.')

        
            saving_dir = args.save_dir_res+args.method+'/'
            if args.aug_smooth and not args.eigen_smooth:
                saving_dir = saving_dir +'aug_smooth/'
            elif args.eigen_smooth and not args.aug_smooth:
                saving_dir = saving_dir +'eigen_smooth/'
            elif args.aug_smooth and args.eigen_smooth:
                saving_dir = saving_dir +'aug_eigen_smooth/'
            else: 
                saving_dir = saving_dir +'none/'
                
            if idx%500==0:
                print(saving_dir)
                
            if args.separate_plot:
                fig = plt.figure(); plt.imshow(img); plt.axis('off'); plt.tight_layout(); plt.savefig(saving_dir+'few_random_image/'+img_name)
                fig = plt.figure(); plt.imshow(cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)); plt.axis('off'); plt.tight_layout(); plt.savefig(saving_dir+'few_random_image/'+args.method+'_'+img_name)
                fig = plt.figure(); plt.imshow(cv2.cvtColor(cam_gb, cv2.COLOR_BGR2RGB)); plt.axis('off'); plt.tight_layout(); plt.savefig(saving_dir+'few_random_image/'+'cam_gb_'+img_name)
                plt.close("all")
                
            elif args.pred_mask:
                ret,thresh = cv2.threshold(cam_mask, 0.5, 255, cv2.THRESH_BINARY)
                converted_cam_mask = Image.fromarray(thresh[:,:,1]*255).convert('L')
                plt.imshow(converted_cam_mask)
                converted_cam_mask.save(saving_dir+'masks/'+img_name)
                plt.close("all")
                # print('hvbdjhvbf')
            
            else:
                fig, ax1 = plt.subplots(1, 4, sharex=True, sharey=True)
                ax1[0].imshow(img); ax1[0].set_title('Original', fontsize=10)
                ax1[0].text(10, 20, str(label), color='red', fontsize=11, fontweight='bold')
                ax1[1].imshow(cv2.cvtColor(gb, cv2.COLOR_BGR2RGB)); ax1[1].set_title('Guided backprop', fontsize=10)
                ax1[2].imshow(cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)); ax1[2].set_title(args.method, fontsize=10)
                ax1[3].imshow(cv2.cvtColor(cam_gb, cv2.COLOR_BGR2RGB)); ax1[3].set_title('Guided '+args.method, fontsize=10)
            
                ax1[0].axis('off'); ax1[1].axis('off'); ax1[2].axis('off'); ax1[3].axis('off')
                plt.tight_layout(); plt.savefig(saving_dir+img_name, bbox_inches='tight'); plt.close("all")
