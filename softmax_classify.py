import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential,Model
from keras.layers import ZeroPadding2D,Convolution2D,MaxPooling2D
from keras.layers import Dense,Dropout,Softmax,Flatten,Activation,BatchNormalization
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.imagenet_utils import preprocess_input
import keras.backend as K
from keras import losses


# Softmax regressor to classify images based on encoding 
classifier_model=Sequential()
classifier_model.add(Dense(units=100,input_dim=2622,kernel_initializer='glorot_uniform'))
classifier_model.add(BatchNormalization())
classifier_model.add(Activation('tanh'))
classifier_model.add(Dropout(0.3))
classifier_model.add(Dense(units=10,kernel_initializer='glorot_uniform'))
classifier_model.add(BatchNormalization())
classifier_model.add(Activation('tanh'))
classifier_model.add(Dropout(0.2))
classifier_model.add(Dense(units=6,kernel_initializer='he_uniform'))
classifier_model.add(Activation('softmax'))
classifier_model.compile(loss=losses.SparseCategoricalCrossentropy(),optimizer='nadam',metrics=['accuracy'])

# classifier_model.fit(x_train,y_train,epochs=100,validation_data=(x_test,y_test))
