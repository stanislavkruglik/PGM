import numpy as np
import cv2
import os


def generate_data(images_paths, masks_paths, save_path='Images'):
	images = [cv2.imread(path) for path in images_paths]
	images = [np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for image in images]

	masks = [cv2.imread(path, 0) for path in masks_paths]
	masks = [cv2.threshold(mask,222,255,cv2.THRESH_BINARY_INV)[1] for mask in masks]
	masks= [mask.astype(np.bool) for mask in masks]

	if save_path[-1] != '/':
		save_path += '/'
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	np.save(save_path + 'images',images)
	np.save(save_path + 'masks',masks)


