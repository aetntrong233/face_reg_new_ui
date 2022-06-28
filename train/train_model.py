import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import math
from keras.metrics import CategoricalAccuracy
from keras.preprocessing.image import ImageDataGenerator
import os
from keras.models import load_model
import sys
sys.path.append(r'D:\sw\face_rec\face_reg_new_ui')
from inceptionresnetv2 import get_train_model


EPOCHS = 500
BATCH_SIZE = 128

callbacks = [
    keras.callbacks.ModelCheckpoint("train/check_point.h5", monitor='categorical_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='max'),
]

train_data_dir = "D:/sw/face_rec/data/CASIA-WebFace_aligned"

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(299, 299),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    ) 

num_class = len(os.listdir(train_data_dir))

model = get_train_model(num_class)

model.compile(loss='categorical_crossentropy',optimizer='Adam', metrics=[CategoricalAccuracy()])

# model = load_model('train/200_09602_32119/check_point.h5')

history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // BATCH_SIZE,
    epochs = EPOCHS,
    callbacks=callbacks,
    )

model.save_weights('train/finished_weights.h5')


# Training plots
epochs = [i for i in range(1, len(history.history['loss'])+1)]

plt.figure(1)
plt.plot(epochs, history.history['categorical_accuracy'], color='blue', label="training_accuracy")
plt.legend(loc='best')
plt.title('training')
plt.xlabel('epoch')
plt.savefig('train/acc.png', bbox_inches='tight')
plt.show()
plt.figure(2)
plt.plot(epochs, history.history['loss'], color='red', label="training_loss")
plt.legend(loc='best')
plt.title('training')
plt.xlabel('epoch')
plt.savefig('train/loss.png', bbox_inches='tight')
plt.show()