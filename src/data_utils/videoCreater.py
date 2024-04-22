#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 18:05:02 2021

@author: homai
"""


import cv2
import argparse
import time
import os

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
ap.add_argument("-o", "--output", required=False, default='Public_Public_aug_eigen_smooth.mp4', help="output video file")
args = vars(ap.parse_args())

# Arguments

# os.system('mount_smbfs //hrashi4:H19_O94_M19_ay@10.157.80.76/aio /Users/homai/Desktop/mntpoint')
# dir_path = '/Users/homai/Desktop/mntpoint/Datasets/GlaucomaPrediction/OpticDisc/Cirrus_OIS/'
dir_path = '/Volumes/Homa/Homa/Glaucoma_prediction/Real_World_Data_paper/zOLD_experiments/Public_Public/visualization_CAM/on_train_set/gradcam/aug_eigen_smooth/'


im_size = 512



# dir_path = '.'
ext = args['extension']
output = args['output']

images = []
for f in os.listdir(dir_path):
    if f.endswith(ext):
        images.append(f)

# Determine the width and height from the first image
image_path = os.path.join(dir_path, images[0])
frame = cv2.imread(image_path)
# frame = cv2.resize(frame, (im_size,im_size))
cv2.imshow('video',frame)
height, width, channels = frame.shape


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 5.0, (width, height), True) #2.5

counter = 0
for image in images:
    image_path = os.path.join(dir_path, image)
    frame = cv2.imread(image_path)
    # frame = cv2.resize(frame, (im_size,im_size))
    counter = counter + 1
    print('Added image '+str(counter)+' :)!')

    out.write(frame) # Write out frame to video

    cv2.imshow('frame',frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

print("The output video is {}".format(output))
print("Total number of images = ", len(images))











