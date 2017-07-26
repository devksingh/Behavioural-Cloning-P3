import csv
import cv2
import random
import numpy as np
import math
import sklearn
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Read the csv file
samples = []
with open('./driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader, None)  # skip the header
	str_adjust = 0.23
	for line in reader:
		samples.append(('C',line[0],float(line[3])))
		samples.append(('L',line[1],(float(line[3])+str_adjust)))
		samples.append(('R',line[2],(float(line[3])-str_adjust)))
#print(len(samples))

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Resize the image, crop top 50 and bottom 25 pixel and resize it to 66x200x3 images
def image_resize(img):
	return cv2.resize(img[51:135, 0:319], (200,66), interpolation = cv2.INTER_AREA)

# Random horizontal shift in image to make it look like car is drifting left or right and calcuate steering angle to correct the direction. I have used .004 per pixel correction in steering angle
# This images will help model to learn how to recover when car is drifting in left or right direction
def image_shift(img,str):
	rows = 160
	columns = 320
	# Randomly shift image between -70 to 70 pixel.
	shift = round((np.random.uniform() - 0.5)*140)
	M = np.float32([[1,0,shift],[0,1,0]])
	return (cv2.warpAffine(img,M,(columns,rows))), (shift*.004+str)

#Flip the image because Track-1 is left bias. It will give some data to make the model learn right turn as well
def image_flip(img,str):
	return cv2.flip(img,1), -str
#Consolidate the data with data augmentation 	

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			images = []
			angles = []
			for batch_sample in batch_samples:
				name = batch_sample[1].strip()
				#print(batch_sample[0])
				#print('Name:'+name)
				if (batch_sample[0] == 'C'):
					image = cv2.imread(name)
					#print(len(image))
					angle = float(batch_sample[2])
					images.append(image_resize(image))
					angles.append(angle)
					shifted_image,shifted_angle = image_shift(image,angle)
					images.append(image_resize(shifted_image))
					angles.append(shifted_angle)
					if abs(angle) > .02:
						flip_image,flip_angle = image_flip(image,angle)
						images.append(image_resize(flip_image))
						angles.append(flip_angle)
				if (batch_sample[0] == 'L'):
					image = cv2.imread(name)
					angle = float(batch_sample[2])
					shifted_image,shifted_angle = image_shift(image,angle)
					images.append(image_resize(shifted_image))
					angles.append(shifted_angle)
					if abs(angle) > .02:
						flip_image,flip_angle = image_flip(image,angle)
						images.append(image_resize(flip_image))
						angles.append(flip_angle)
				if (batch_sample[0] == 'R'):
					image = cv2.imread(name)
					angle = float(batch_sample[2])
					shifted_image,shifted_angle = image_shift(image,angle)
					images.append(image_resize(shifted_image))
					angles.append(shifted_angle)
					if abs(angle) > .02:
						flip_image,flip_angle = image_flip(image,angle)
						images.append(image_resize(flip_image))
						angles.append(flip_angle)
            # trim image to only see section with road
			X_train = np.array(images)
			y_train = np.array(angles)
			#print(X_train[0].shape)
			#print(y_train.shape)
			yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

#define model
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(66,200,3)))

 # Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
# Add two 3x3 convolution layers (output depth 64, and 64)
model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
# Add a flatten layer
model.add(Flatten())
# Add three fully connected layers (depth 100, 50, 10), tanh activation (and dropouts)
model.add(Dense(100, W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Dense(50, W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Dense(10, W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Dropout(0.5))

# Add a fully connected output layer
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= 42000, validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=7)
model.save('model_n.h5')

