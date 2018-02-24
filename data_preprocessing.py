import cv2
import numpy as np
import os

class DataAugmentation:
	def __init__(self, RESIZE_DIM):
		self.RESIZE_DIM = RESIZE_DIM

	def resize(self, image):
		pass


	def flip_90(self, image):
		pass


	def flip_180(self, image):
		pass


	def flip_270(self, image):
		pass


	def random_rotate(self, image):
		pass


	def random_zoom(self, image):
		pass


	def random_lightning(self, image):
		pass


	


class DataPreprocessing(DataAugmentation):

	def __init__(self, DATA_DIR, RESIZE_DIM):
		DataAugmentation.__init__(RESIZE_DIM)
		self.DATA_DIR = DATA_DIR

	def augment(self, augmentation_factor, flip_90=True, flip_180=True, flip_270=True, random_rotate=True, random_zoom=True, random_lightning=True):
		for folder_name in os.listdir(DATA_DIR):
			pass


def main():
	DATA_DIR = 'G:/DL/satellite_imagery_land_classification/data/dataset'
	RESIZE_DIM = 224


if __name__ == '__main__':
	main()
