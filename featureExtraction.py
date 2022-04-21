import tensorflow as tf
from keras import Model
import numpy as np
from keras.models import load_model
import os
import gdown
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, ZeroPadding2D, AveragePooling2D, LocallyConnected2D
from keras.layers import Convolution2D, MaxPool2D, Input, Lambda, concatenate
from keras import Sequential
from keras import backend as K
import zipfile
from keras.layers import add, Concatenate, Add, PReLU
from tensorflow.python.keras.engine import training
import cv2


# input: url to download model weights
# output: model vgg face
# description: create vgg16 model based on keras -> load pretrained model weights -> remove softmax layer
def load_model_(url = 'https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5'):
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))
	model.add(Convolution2D(4096, (7, 7), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(4096, (1, 1), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(2622, (1, 1)))
	model.add(Flatten())
	model.add(Activation('softmax'))
	output = r'storage/model/feature_extraction_model/vgg_face_weights.h5'
	if not os.path.exists('storage/model/feature_extraction_model'):
		os.makedirs('storage/model/feature_extraction_model')
	if os.path.isfile(output) != True:
		print("vgg_face_weights.h5 will be downloaded...")
		gdown.download(url, output, quiet=False)
	model.load_weights(output)
	vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
	return vgg_face_descriptor


model = load_model_()
input_shape = model.layers[0].input_shape
if type(input_shape) == list:
	input_shape = input_shape[0][1:3]
else:
	input_shape = input_shape[1:3]
target_size = input_shape

# input: array of cropped face image
# output: array of normalized cropped face image
# description: resized image to (224,224) px, pad other side if image not square -> normalized image
def img_normalize(face_pixels):
	face_pixels = face_pixels.astype('float64')
	if face_pixels.shape[0] == 0 or face_pixels.shape[1] == 0:
		raise ValueError("Detected face shape is ", face_pixels.shape,". Consider to set enforce_detection argument to False.")
	if face_pixels.shape[0] > 0 and face_pixels.shape[1] > 0:
		factor_0 = target_size[0] / face_pixels.shape[0]
		factor_1 = target_size[1] / face_pixels.shape[1]
		factor = min(factor_0, factor_1)
		dsize = (int(face_pixels.shape[1] * factor), int(face_pixels.shape[0] * factor))
		face_pixels = cv2.resize(face_pixels, dsize)
		# Then pad the other side to the target size by adding black pixels
		diff_0 = target_size[0] - face_pixels.shape[0]
		diff_1 = target_size[1] - face_pixels.shape[1]
		face_pixels = np.pad(face_pixels, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
	if face_pixels.shape[0:2] != target_size:
		face_pixels = cv2.resize(face_pixels, target_size)
	face_pixels[...,0] -= 93.5940
	face_pixels[...,1] -= 104.7624
	face_pixels[...,2] -= 129.1863
	return face_pixels

# input: array of cropped face image
# output: array (2622)
# description: normalized image -> predicted with vggface model
def feature_extraction(face_pixels):
    face_pixels = img_normalize(face_pixels)
    samples = np.expand_dims(face_pixels,axis=0)
    yhat = model.predict(samples)
    embedding = yhat[0]
    return embedding