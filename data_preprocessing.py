import cv2
import numpy as np
import os

from PIL import Image

class DataAugmentation:
	def __init__(self, RESIZE_DIM):
		self.RESIZE_DIM = RESIZE_DIM
		

	def flip_90(self, image):
		new_image = np.rot90(image)
		new_image = new_image.reshape((3, self.RESIZE_DIM, self.RESIZE_DIM))
		return new_image


	def flip_180(self, image):
		new_image = np.rot90(image)
		new_image = np.rot90(new_image)
		new_image = new_image.reshape((3, self.RESIZE_DIM, self.RESIZE_DIM))
		return new_image


	def flip_270(self, image):
		new_image = np.rot90(image)
		new_image = np.rot90(new_image)
		new_image = np.rot90(new_image)
		new_image = new_image.reshape((3, self.RESIZE_DIM, self.RESIZE_DIM))
		return new_image


	def flip_vertically(self, image):
		new_image = cv2.flip(image, 1)
		new_image = new_image.reshape((3, self.RESIZE_DIM, self.RESIZE_DIM))
		return new_image


	def flip_horizontally(self, image):
		new_image = cv2.flip(image, 0)
		new_image = new_image.reshape((3, self.RESIZE_DIM, self.RESIZE_DIM))
		return new_image


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


	def augment_and_convert_to_hdf5(self, augmentation_factor=1, flip_90=True, flip_180=True, flip_270=True, flip_vertically=True, flip_horizontally=True, random_rotate=True, random_zoom=True, random_lightning=True):
		images_dataset = []
		labels_dataset = []

		for folder_name in os.listdir(self.DATA_DIR):
			for file_name in os.listdir(self.DATA_DIR + os.sep + folder_name):
				image = None
				label = None
				

				if '.jpg' in file_name:
					image = cv2.imread(self.DATA_DIR + os.sep + folder_name + os.sep + file_name)
				elif '.tif' in file_name:
					image = np.array(Image.open(self.DATA_DIR + os.sep + folder_name + os.sep + file_name))

				image = cv2.resize(image, (self.RESIZE_DIM, self.RESIZE_DIM))
				new_image = image.copy()
				new_image = new_image.reshape((3, self.RESIZE_DIM, self.RESIZE_DIM))
				label = folder_name
				images_dataset.append(new_image)
				labels_dataset.append(label)

				if flip_90:
					new_image = self.flip_90(image.copy())
					images_dataset.append(new_image)
					labels_dataset.append(label)

				if flip_180:
					new_image = self.flip_180(image.copy())
					images_dataset.append(new_image)
					labels_dataset.append(label)

				if flip_270:
					new_image = self.flip_270(image.copy())
					images_dataset.append(new_image)
					labels_dataset.append(label)

				if flip_vertically:
					new_image = self.flip_vertically(image.copy())
					images_dataset.append(new_image)
					labels_dataset.append(label)

				if flip_horizontally:
					new_image = self.flip_horizontally(image.copy())
					images_dataset.append(new_image)
					labels_dataset.append(label)

				



			


def main():
	DATA_DIR = 'G:/DL/satellite_imagery_land_classification/data/dataset'
	RESIZE_DIM = 224


if __name__ == '__main__':
	main()
