import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar100,mnist,cifar10,fashion_mnist
from scipy.io import loadmat

import numpy as onp #original numpy
import jax.numpy as jnp #jax numpy
import itertools
#import custom_datasets


# TODO: Setup this function to take in a string for the data set
def setupMNIST():
	classes = 10
	subtract_pixel_mean = True

	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	#for MNIST
	x_train = onp.expand_dims(x_train,axis=3)
	x_test = onp.expand_dims(x_test,axis=3)


	y_train = y_train.reshape([-1])
	y_test = y_test.reshape([-1])

	# Input image dimensions.
	input_shape = x_train.shape[1:]

	# Normalize data.
	x_train = x_train.astype('float32') / 255
	x_test = x_test.astype('float32') / 255

	# If subtract pixel mean is enabled
	if subtract_pixel_mean:
		x_train_mean = onp.mean(x_train, axis=0)
		x_train -= x_train_mean
		x_test -= x_train_mean

	orig_x_train = onp.array(x_train)
	orig_y_train = onp.array(y_train)

	datagen = ImageDataGenerator(
			# set input mean to 0 over the dataset
			featurewise_center=False,
			# set each sample mean to 0
			samplewise_center=False,
			# divide inputs by std of dataset
			featurewise_std_normalization=False,
			# divide each input by its std
			samplewise_std_normalization=False,
			# apply ZCA whitening
			zca_whitening=False,
			# epsilon for ZCA whitening
			zca_epsilon=1e-06,
			# randomly rotate images in the range (deg 0 to 180)
			rotation_range=0,
			# randomly shift images horizontally
			width_shift_range=0.1,
			# randomly shift images vertically
			height_shift_range=0.1,
			# set range for random shear
			shear_range=0.,
			# set range for random zoom
			zoom_range=0.,
			# set range for random channel shifts
			channel_shift_range=0.,
			# set mode for filling points outside the input boundaries
			fill_mode='nearest',
			# value used for fill_mode = "constant"
			cval=0.,
			# randomly flip images
			horizontal_flip=True,
			# randomly flip images
			vertical_flip=False,
			# set rescaling factor (applied before any other transformation)
			rescale=None,
			# set function that will be applied on each input
			preprocessing_function=None,
			# image data format, either "channels_first" or "channels_last"
			data_format=None,
			# fraction of images reserved for validation (strictly between 0 and 1)
			validation_split=0.0)

	datagen = ImageDataGenerator()


	datagen.fit(x_train)
	train_flow = datagen.flow(x_train, y_train, batch_size=128)

	train_ds = map(lambda x: {'image': x[0].astype(onp.float32),
							 'label': x[1].astype(onp.int32)},train_flow)


	test_ds = {'image': x_test.astype(jnp.float32),
			'label': y_test.astype(jnp.int32)}
	full_train_ds = {'image': x_train.astype(jnp.float32),
			'label': y_train.astype(jnp.int32)}

	return x_train, full_train_ds, train_ds, test_ds, classes

def setupFashionMNIST():
	classes = 10
	subtract_pixel_mean = True

	(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


	# Add 3rd dimension to correspond to color in RGB images
	x_train = onp.expand_dims(x_train,axis=3)
	x_test = onp.expand_dims(x_test,axis=3)

	y_train = y_train.reshape([-1])
	y_test = y_test.reshape([-1])

	# Input image dimensions.
	input_shape = x_train.shape[1:]

	# Normalize data.
	x_train = x_train.astype('float32') / 255
	x_test = x_test.astype('float32') / 255

	# If subtract pixel mean is enabled
	if subtract_pixel_mean:
		x_train_mean = onp.mean(x_train, axis=0)
		x_train -= x_train_mean
		x_test -= x_train_mean

	orig_x_train = onp.array(x_train)
	orig_y_train = onp.array(y_train)

	datagen = ImageDataGenerator()


	datagen.fit(x_train)
	train_flow = datagen.flow(x_train, y_train, batch_size=128)

	train_ds = map(lambda x: {'image': x[0].astype(onp.float32),
							 'label': x[1].astype(onp.int32)},train_flow)
	full_train_ds = {'image': x_train.astype(jnp.float32),
			'label': y_train.astype(jnp.int32)}

	test_ds = {'image': x_test.astype(jnp.float32),
			'label': y_test.astype(jnp.int32)}
	return x_train, full_train_ds, train_ds, test_ds, classes


def setupCIFAR10():
	classes = 10
	subtract_pixel_mean = True
	
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()

	y_train = y_train.reshape([-1])
	y_test = y_test.reshape([-1])

	# Input image dimensions.
	input_shape = x_train.shape[1:]

	# Normalize data.
	x_train = x_train.astype('float32') / 255
	x_test = x_test.astype('float32') / 255

	# If subtract pixel mean is enabled
	if subtract_pixel_mean:
		x_train_mean = onp.mean(x_train, axis=0)
		x_train -= x_train_mean
		x_test -= x_train_mean

	orig_x_train = onp.array(x_train)
	orig_y_train = onp.array(y_train)

	datagen = ImageDataGenerator()


	datagen.fit(x_train)
	train_flow = datagen.flow(x_train, y_train, batch_size=128)

	train_ds = map(lambda x: {'image': x[0].astype(onp.float32),
							 'label': x[1].astype(onp.int32)},train_flow)
	full_train_ds = {'image': x_train.astype(jnp.float32),
			'label': y_train.astype(jnp.int32)}

	test_ds = {'image': x_test.astype(jnp.float32),
			'label': y_test.astype(jnp.int32)}

	return x_train, full_train_ds, train_ds, test_ds, classes


def setupCIFAR100():
	classes = 100
	subtract_pixel_mean = True

	(x_train, y_train), (x_test, y_test) = cifar100.load_data()

	y_train = y_train.reshape([-1])
	y_test = y_test.reshape([-1])

	# Input image dimensions.
	input_shape = x_train.shape[1:]

	# Normalize data.
	x_train = x_train.astype('float32') / 255
	x_test = x_test.astype('float32') / 255

	# If subtract pixel mean is enabled
	if subtract_pixel_mean:
		x_train_mean = onp.mean(x_train, axis=0)
		x_train -= x_train_mean
		x_test -= x_train_mean

	orig_x_train = onp.array(x_train)
	orig_y_train = onp.array(y_train)

	datagen = ImageDataGenerator()


	datagen.fit(x_train)
	train_flow = datagen.flow(x_train, y_train, batch_size=128)

	train_ds = map(lambda x: {'image': x[0].astype(onp.float32),
							 'label': x[1].astype(onp.int32)},train_flow)
	full_train_ds = {'image': x_train.astype(jnp.float32),
			'label': y_train.astype(jnp.int32)}

	test_ds = {'image': x_test.astype(jnp.float32),
			'label': y_test.astype(jnp.int32)}

	return x_train, full_train_ds, train_ds, test_ds, classes


def setupSVHN():
	classes = 10
	subtract_pixel_mean = True

	
	def load_data(path):
		""" Helper function for loading a MAT-File"""
		data = loadmat(path)
		return data['X'], data['y']

	x_train, y_train = load_data('train_32x32.mat')
	x_test, y_test = load_data('test_32x32.mat')

	# Gets rid of the extra dimension on the training labels
	y_train = y_train.reshape([-1])
	y_test = y_test.reshape([-1])

	x_train = onp.moveaxis(x_train, -1, 0)
	x_test = onp.moveaxis(x_test, -1, 0)

	# Input image dimensions.
	input_shape = x_train.shape[1:]

	# Normalize data.
	x_train = x_train.astype('float32') / 255
	x_test = x_test.astype('float32') / 255

	# If subtract pixel mean is enabled
	if subtract_pixel_mean:
		x_train_mean = onp.mean(x_train, axis=0)
		x_train -= x_train_mean
		x_test -= x_train_mean

	orig_x_train = onp.array(x_train)
	orig_y_train = onp.array(y_train)

	datagen = ImageDataGenerator()


	datagen.fit(x_train)
	train_flow = datagen.flow(x_train, y_train, batch_size=128)

	train_ds = map(lambda x: {'image': x[0].astype(onp.float32),
							 'label': x[1].astype(onp.int32)},train_flow)
	full_train_ds = {'image': x_train.astype(jnp.float32),
			'label': y_train.astype(jnp.int32)}

	test_ds = {'image': x_test.astype(jnp.float32),
			'label': y_test.astype(jnp.int32)}

	return x_train, full_train_ds, train_ds, test_ds, classes


def setupTinyImageNet():
	classes = 200
	subtract_pixel_mean = True

	dataset = custom_datasets.TINYIMAGENET('Data', train=True, download=True)

	print(dataset)
	#(x_train, y_train), (x_test, y_test) = cifar100.load_data()

	y_train = y_train.reshape([-1])
	y_test = y_test.reshape([-1])

	# Input image dimensions.
	input_shape = x_train.shape[1:]

	# Normalize data.
	x_train = x_train.astype('float32') / 255
	x_test = x_test.astype('float32') / 255

	# If subtract pixel mean is enabled
	if subtract_pixel_mean:
		x_train_mean = onp.mean(x_train, axis=0)
		x_train -= x_train_mean
		x_test -= x_train_mean

	orig_x_train = onp.array(x_train)
	orig_y_train = onp.array(y_train)

	datagen = ImageDataGenerator()


	datagen.fit(x_train)
	train_flow = datagen.flow(x_train, y_train, batch_size=128)

	train_ds = map(lambda x: {'image': x[0].astype(onp.float32),
							 'label': x[1].astype(onp.int32)},train_flow)
	full_train_ds = {'image': x_train.astype(jnp.float32),
			'label': y_train.astype(jnp.int32)}

	test_ds = {'image': x_test.astype(jnp.float32),
			'label': y_test.astype(jnp.int32)}

	return x_train, full_train_ds, train_ds, test_ds, classes

