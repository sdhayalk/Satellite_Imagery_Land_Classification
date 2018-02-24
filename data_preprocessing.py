import cv2
import numpy as np
import os
import h5py

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


	def augment_and_convert_to_hdf5(self, train_validation_counter, hdf5_train_filename, hdf5_validation_filename, augmentation_factor=1, flip_90=True, flip_180=True, flip_270=True, flip_vertically=True, flip_horizontally=True, random_rotate=True, random_zoom=True, random_lightning=True):
		images_train_dataset = []
		labels_train_dataset = []
		images_validation_dataset = []
		labels_validation_dataset = []
		counter = 0

		for folder_name in os.listdir(self.DATA_DIR):
			print("In folder", folder_name)
			for file_name in os.listdir(self.DATA_DIR + os.sep + folder_name):
				image = None
				label = None
				images_train_dataset_temp = []
				labels_train_dataset_temp = []

				if '.jpg' in file_name:
					image = cv2.imread(self.DATA_DIR + os.sep + folder_name + os.sep + file_name)
				elif '.tif' in file_name:
					image = np.array(Image.open(self.DATA_DIR + os.sep + folder_name + os.sep + file_name))

				image = cv2.resize(image, (self.RESIZE_DIM, self.RESIZE_DIM))
				new_image = image.copy()
				new_image = new_image.reshape((3, self.RESIZE_DIM, self.RESIZE_DIM))
				label = folder_name
				images_train_dataset_temp.append(new_image)
				labels_train_dataset_temp.append(label)

				if flip_90:
					new_image = self.flip_90(image.copy())
					images_train_dataset_temp.append(new_image)
					labels_train_dataset_temp.append(label)

				if flip_180:
					new_image = self.flip_180(image.copy())
					images_train_dataset_temp.append(new_image)
					labels_train_dataset_temp.append(label)

				if flip_270:
					new_image = self.flip_270(image.copy())
					images_train_dataset_temp.append(new_image)
					labels_train_dataset_temp.append(label)

				if flip_vertically:
					new_image = self.flip_vertically(image.copy())
					images_train_dataset_temp.append(new_image)
					labels_train_dataset_temp.append(label)

				if flip_horizontally:
					new_image = self.flip_horizontally(image.copy())
					images_train_dataset_temp.append(new_image)
					labels_train_dataset_temp.append(label)

				
				if counter == train_validation_counter:
					images_train_dataset.extend(images_train_dataset_temp)
					labels_train_dataset.extend(labels_train_dataset_temp)
					counter += 1
				else:
					images_validation_dataset.extend(images_train_dataset_temp)
					labels_validation_dataset.extend(labels_train_dataset_temp)
					counter = 0

		with h5py.File(hdf5_train_filename, 'w') as f:
			f['data'] = images_train_dataset
			f['label'] = labels_train_dataset

		with h5py.File(hdf5_validation_filename, 'w') as f:
			f['data'] = images_validation_dataset
			f['label'] = labels_validation_dataset

		print("Saved as HDF5")
			


def main():
	DATA_DIR = 'G:/DL/satellite_imagery_land_classification/data/mydataset'
	RESIZE_DIM = 224
	hdf5_train_filename = 'G:/DL/satellite_imagery_land_classification/data/dataset_train.hdf5'
	hdf5_validation_filename = 'G:/DL/satellite_imagery_land_classification/data/dataset_validation.hdf5'
	train_validation_counter = 15


	data_preprocessing_instance = DataPreprocessing(DATA_DIR, RESIZE_DIM)
	data_preprocessing_instance.augment_and_convert_to_hdf5(train_validation_counter, hdf5_train_filename, hdf5_validation_filename)

if __name__ == '__main__':
	main()
