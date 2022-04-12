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
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.optimizer_v2 import nadam
import os
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder


# Softmax regressor to classify images based on encoding
def softmax_classifier_model(input_dim=2622,classes=1000):
    classifier_model=Sequential()
    classifier_model.add(Dense(units=100,input_dim=input_dim,kernel_initializer='glorot_uniform'))
    classifier_model.add(BatchNormalization())
    classifier_model.add(Activation('tanh'))
    classifier_model.add(Dropout(0.3))
    classifier_model.add(Dense(units=10,kernel_initializer='glorot_uniform'))
    classifier_model.add(BatchNormalization())
    classifier_model.add(Activation('tanh'))
    classifier_model.add(Dropout(0.2))
    classifier_model.add(Dense(units=classes,kernel_initializer='he_uniform'))
    classifier_model.add(Activation('softmax'))
    return classifier_model


dataset_path = 'storage/dataset.npz'
save_path = 'storage/classifier_model/classifier_model.h5'


# train softmax model function
def train_softmax(epochs=100, batch_size=24, lr=0.001):
    if os.path.isfile(dataset_path):
        dataset = np.load(dataset_path)
    feature = dataset['feature_ds']
    id = dataset['id']
    le = LabelEncoder()
    y = le.fit_transform(id)
    classes = len(list(dict.fromkeys(y)))
    x, y = shuffle(x, y)
    classifier_model = softmax_classifier_model()
    optimizer = nadam(learning_rate=0.001)
    classifier_model.compile(loss=losses.SparseCategoricalCrossentropy(),optimizer=optimizer,metrics=['accuracy'])
    classifier_model.fit(x,y,epochs=epochs,batch_size=batch_size)
    classifier_model.save(save_path)


# predict with sofmax model
def softmax_classifier(feature):
    if os.path.isfile(dataset_path):
        dataset = np.load(dataset_path)
    classifier_model = load_model(save_path)
    predict = classifier_model.predict(feature)
    id_encode = LabelEncoder()
    id_encode.classes_ = dataset['id']
    max_prob = np.max(predict[0])
    max_index = np.where(predict[0] == max_prob)
    id = id_encode.inverse_transform(max_index)
    return  id, max_prob

