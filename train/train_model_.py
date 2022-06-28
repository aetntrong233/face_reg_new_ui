from cmath import e
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


EPOCHS = 200
BATCH_SIZE = 128
initial_learning_rate = 0.001

def lr_exp_decay(epoch, lr):
    k = 0.1
    return initial_learning_rate * math.exp(-k*epoch)

callbacks = [
    # keras.callbacks.LearningRateScheduler(lr_exp_decay, verbose=1),
    keras.callbacks.ModelCheckpoint("train/check_point.h5", monitor='categorical_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='max'),
    # keras.callbacks.ModelCheckpoint("train/check_point_val.h5", monitor='val_categorical_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='max'),
]

data_dir = r"D:\sw\face_rec\data\fei\fei_both"

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    label_mode = 'categorical',
    image_size=(299,299),
    batch_size=BATCH_SIZE,
    shuffle=True,
    # subset = "training",
    # validation_split = 0.2,
    # seed = 23,
)

# val_ds = tf.keras.utils.image_dataset_from_directory(
#     data_dir,
#     label_mode = 'categorical',
#     image_size=(299,299),
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     subset = "validation",
#     validation_split = 0.2,
#     seed = 23,
# )

# num_class = len(train_ds.class_names)
num_class = len(os.listdir(data_dir))

train_ds = train_ds.prefetch(buffer_size=128)
# val_ds = val_ds.prefetch(buffer_size=128)

# data_augmentation = keras.layers.RandomZoom(0.2)
# augmented_train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
normalization_layer = keras.layers.Rescaling(1./255)
normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

model = get_train_model(num_class)

# opt = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
# model.compile(optimizer=opt, 
#             loss='categorical_crossentropy',
#             metrics=['accuracy'])
model.compile(loss='categorical_crossentropy',optimizer='Adam', metrics=[CategoricalAccuracy()])

history = model.fit(
    normalized_train_ds,
    epochs = EPOCHS,
    callbacks = callbacks,
    # validation_data = normalized_val_ds,
)

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