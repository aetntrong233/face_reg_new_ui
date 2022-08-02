import tensorflow as tf
import keras
import os
import sys
sys.path.append(r'D:\sw\face_rec\face_reg_new_ui')
from models.feature_extraction_model.inceptionresnetv2 import get_train_model
from keras.models import Model, load_model
from train.arcface_metrics import ArcFace
from keras.layers import Input
import numpy as np
import cv2

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

EPOCHS = 200
BATCH_SIZE = 8
DATA_DIR = r''
RESULTS_DIR = 'train/results'

print('INFO: TRAIN MODEL')

print('INFO: Dataset pre_processing')
print('Please wait...')

num_class = len(os.listdir(DATA_DIR))

x_train = []
labels = []
train_classes = []
for i, fld_name in enumerate(os.listdir(DATA_DIR)):
    fld_path = os.path.join(DATA_DIR, fld_name)
    for f_name in os.listdir(fld_path):
        f_path = os.path.join(fld_path, f_name)
        x_train.append(cv2.resize(cv2.imread(f_path), (112, 112))/255.0)
        labels.append(i)
labels = np.array(labels)
y_train = tf.keras.utils.to_categorical(labels, num_classes=num_class, dtype='int32')
del labels
x_train = np.array(x_train)

# import random
# temp = list(zip(x_train, y_train))
# random.shuffle(temp)
# x_train , y_train = zip(*temp)

print('INFO: Create model')
# model = get_train_model(num_class)

# 
inputs = Input(shape=(112, 112, 3))
labels = Input(shape=(num_class,))

base_model = get_train_model(num_class)
extract_model = Model(base_model.inputs, base_model.layers[-2].output)
x = extract_model(inputs)
output = ArcFace(n_classes=num_class)([x, labels])
model = Model([inputs, labels], output)

print('INFO: Compile model')
model.compile(loss='categorical_crossentropy',optimizer='Adam', metrics=['acc'])

cp = keras.callbacks.ModelCheckpoint(os.path.join(RESULTS_DIR, "check_point_arcface.h5"), monitor='loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min'),

callbacks = [
    cp,
]

model.summary()

history = model.fit(
    [x_train, y_train],
    y_train,
    batch_size = BATCH_SIZE,
    epochs = EPOCHS,
    callbacks = callbacks,
    # validation_data = ([x_train, y_train], y_train),
    shuffle=True,
)

model.save(os.path.join(RESULTS_DIR, "final_arcface.h5"))


# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# arcface_model = load_model(os.path.join(RESULTS_DIR, "check_point_arcface_fei.h5"), custom_objects={'ArcFace': ArcFace})
# arcface_model = arcface_model.get_layer('model_1')
# # arcface_model = Model(inputs=arcface_model.input, outputs=arcface_model.layers[-2].output)
# arcface_model.summary()
# arcface_model.save(r'models\feature_extraction_model\arcface_fei_.h5')
# # arcface_features = arcface_model.predict(x_train, verbose=1)
# # arcface_features /= np.linalg.norm(arcface_features, axis=1, keepdims=True)
# # # plot
# # fig1 = plt.figure()
# # ax1 = Axes3D(fig1)
# # for c in range(len(np.unique(y_train))):
# #     ax1.plot(arcface_features[y_train==c, 0], arcface_features[y_train==c, 1], arcface_features[y_train==c, 2], '.', alpha=0.1)
# # plt.title('ArcFace')
# # plt.show()