import cv2
import os
from PIL import Image 

image_folder = '/Volumes/Homa/Homa/Glaucoma_segmentation/Datasets/crops/Drishti-GS/image/'
video_name = 'mygeneratedvideo.avi'


# Checking the current directory path
print(os.getcwd())

# Folder which contains all the images
# from which video is to be generated
os.chdir(image_folder)
path = image_folder


# Video Generating function
def generate_video(image_folder,video_name):
	os.chdir(image_folder)
	
	images = [img for img in os.listdir(image_folder)
			if img.endswith(".jpg") or
				img.endswith(".jpeg") or
				img.endswith("png")]
	
	# Array images should only consider
	# the image files ignoring others if any
	print(images)

	frame = cv2.imread(os.path.join(image_folder, images[0])); 	frame = cv2.resize(frame, (200,200))

	# setting the frame width, height width
	# the width, height of first image
	height, width, layers = frame.shape; video = cv2.VideoWriter(video_name, 0, 0.1, (200,200))

	# Appending the images to the video one by one
	for image in images:
		video.write(cv2.imread(os.path.join(image_folder, image)))
	
	# Deallocating memories taken for window creation
	cv2.destroyAllWindows()
	video.release() # releasing the video generated


# Calling the generate_video function
generate_video(image_folder,video_name)

