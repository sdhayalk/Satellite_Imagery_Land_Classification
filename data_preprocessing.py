import cv2
import numpy as np
import os
import h5py
import copy

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
		DataAugmentation.__init__(self, RESIZE_DIM)
		self.DATA_DIR = DATA_DIR


	def augment_and_convert_to_hdf5(self, train_validation_counter, hdf5_train_filename, hdf5_validation_filename, augmentation_factor=1, flip_90=True, flip_180=True, flip_270=True, flip_vertically=True, flip_horizontally=True, random_rotate=True, random_zoom=True, random_lightning=True):
		images_train_dataset = []
		labels_train_dataset = []
		images_validation_dataset = []
		labels_validation_dataset = []
		f_train = h5py.File(hdf5_train_filename, 'w')
		f_test = h5py.File(hdf5_validation_filename, 'w')

		counter = 0
		folder_count = -1
		first_flag = True

		for folder_name in os.listdir(self.DATA_DIR):
			print("In folder", folder_name)
			folder_count += 1
			for file_name in os.listdir(self.DATA_DIR + os.sep + folder_name):
				print(file_name)
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
				label = folder_count
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

				
				if counter != train_validation_counter:
					images_train_dataset.extend(images_train_dataset_temp)
					labels_train_dataset.extend(labels_train_dataset_temp)
					counter += 1
				else:
					images_validation_dataset.extend(images_train_dataset_temp)
					labels_validation_dataset.extend(labels_train_dataset_temp)
					counter = 0


				images_train_dataset_np = np.array(images_train_dataset, dtype='float')
				labels_train_dataset_np = copy.deepcopy(labels_train_dataset)
				images_validation_dataset_np = np.array(images_validation_dataset, dtype='float')
				labels_validation_dataset_np = copy.deepcopy(labels_validation_dataset)

				images_train_dataset_np = images_train_dataset_np / 255.0
				images_validation_dataset_np = images_validation_dataset_np / 255.0

				print('images_train_dataset_np.shape', images_train_dataset_np.shape, 'labels_train_dataset_np.shape:', len(labels_train_dataset_np))
				print('images_validation_dataset_np.shape', images_validation_dataset_np.shape, 'labels_validation_dataset_np.shape:', len(labels_validation_dataset_np))

				num_images = 6 #* len(os.listdir(self.DATA_DIR + os.sep + folder_name))

				if first_flag:
					f_train_data = f_train.create_dataset("data", (num_images,3,224,224), maxshape=(None,3,224,224), chunks=(num_images,3,224,224))
					f_train_label = f_train.create_dataset("label", (num_images,), maxshape=(None,), chunks=(num_images,))
					f_test_data = f_test.create_dataset("data", (num_images,3,224,224), maxshape=(None,3,224,224), chunks=(num_images,3,224,224))
					f_test_label = f_test.create_dataset("label", (num_images,), maxshape=(None,), chunks=(num_images,))

					try:
						f_train_data[:] = images_train_dataset_np
						f_train_label[:] = labels_train_dataset_np
						f_test_data[:] = images_validation_dataset_np
						f_test_label[:] = labels_validation_dataset_np
					except:
						pass
					
				else:
					if counter != train_validation_counter:
						f_train_data.resize(f_train_data.shape[0] + num_images, axis=0)
						f_train_label.resize(f_train_label.shape[0] + num_images, axis=0)
						f_train_data[-num_images:] = images_train_dataset_np[-1]
						f_train_label[-num_images:] = labels_train_dataset_np[-1]
						counter += 1
					
					else:
						f_test_data.resize(f_test_data.shape[0] + num_images, axis=0)
						f_test_label.resize(f_test_label.shape[0] + num_images, axis=0)
						f_test_data[-num_images:] = images_validation_dataset_np[-1]
						f_test_label[-num_images:] = labels_validation_dataset_np[-1]
						counter = 0

				print("Saved", folder_name,  "as HDF5")
				first_flag = False
			


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
