from imp import load_compiled
from json import load
from tabnanny import verbose
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras.metrics import CategoricalAccuracy
from keras.preprocessing.image import ImageDataGenerator
import os
import sys
sys.path.append(r'D:\sw\face_rec\face_reg_new_ui')
from models.feature_extraction_model.inceptionresnetv2 import get_train_model
from keras.models import Model
from train.arcface_metrics import ArcFace
from keras.layers import Input
import numpy as np
import math


EPOCHS = 1500
BATCH_SIZE = 256
DATA_DIR = 'dataset/celeb_vn/croped'
RESULTS_DIR = 'train/results'
LR = 0.0001

print('INFO: TRAIN MODEL')

print('INFO: Dataset pre_processing')
print('Please wait...')

num_class = len(os.listdir(DATA_DIR))

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    label_mode = 'categorical',
    image_size=(160, 160),
    batch_size=BATCH_SIZE,
    shuffle=True,
    # subset = "training",
    # validation_split = 0.2,
    # seed = 23,
)

# val_ds = tf.keras.utils.image_dataset_from_directory(
#     DATA_DIR,
#     label_mode = 'categorical',
#     image_size=(299,299),
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     subset = "validation",
#     validation_split = 0.2,
#     seed = 23,
# )

train_ds = train_ds.prefetch(buffer_size=BATCH_SIZE)
# val_ds = val_ds.prefetch(buffer_size=BATCH_SIZE)

# data_augmentation = keras.layers.RandomZoom(0.2)
# augmented_train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
normalization_layer = keras.layers.Rescaling(1./255)
normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

print('INFO: Create model')
# model = get_train_model(num_class)

# print('INFO: Compile model')
# model.compile(loss='categorical_crossentropy',optimizer='Adam', metrics=[CategoricalAccuracy()])

model = keras.models.load_model('train/results/check_point.h5')
score = model.evaluate(normalized_train_ds, batch_size=BATCH_SIZE)

initial_learning_rate = 0.001

def lr_exp_decay(epoch, lr):
    k = 0.1
    return LR
    return initial_learning_rate * math.exp(-k*epoch)

cp1 = keras.callbacks.ModelCheckpoint(os.path.join(RESULTS_DIR, "check_point.h5"), monitor='loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')
cp1.best = score[0]
cp1.best = 6.0

callbacks = [
    keras.callbacks.LearningRateScheduler(lr_exp_decay, verbose=1),
    cp1,
]

model.summary()

history = model.fit(
    normalized_train_ds,
    epochs = EPOCHS,
    verbose = 1,
    callbacks = callbacks,
    # validation_data = normalized_val_ds,
    # shuffle=True,
    # initial_epoch = 390,
)


model.save(os.path.join(RESULTS_DIR, "final.h5"))

model = keras.models.load_model('train/results/final.h5')
score = model.evaluate(normalized_train_ds, batch_size=BATCH_SIZE)

# epochs = [i for i in range(1, len(history.history['loss'])+1)]

# plt.figure(1)
# plt.plot(epochs, history.history['categorical_accuracy'], color='blue', label="training_accuracy")
# plt.legend(loc='best')
# plt.title('training')
# plt.xlabel('epoch')
# plt.savefig(os.path.join(RESULTS_DIR, "acc.png"), bbox_inches='tight')
# plt.show()
# plt.figure(2)
# plt.plot(epochs, history.history['val_categorical_accuracy'], color='blue', label="training_val_accuracy")
# plt.legend(loc='best')
# plt.title('training')
# plt.xlabel('epoch')
# plt.savefig(os.path.join(RESULTS_DIR, "val_acc.png"), bbox_inches='tight')
# plt.show()
# plt.figure(3)
# plt.plot(epochs, history.history['loss'], color='red', label="training_loss")
# plt.legend(loc='best')
# plt.title('training')
# plt.xlabel('epoch')
# plt.savefig(os.path.join(RESULTS_DIR, "loss.png"), bbox_inches='tight')
# plt.show()