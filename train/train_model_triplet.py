from cmath import e
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import keras
import matplotlib.pyplot as plt
import math
from keras.metrics import CategoricalAccuracy
from keras.preprocessing.image import ImageDataGenerator
import os
from keras.models import load_model
import io
import numpy as np
import sys
sys.path.append(r'D:\sw\face_rec\face_reg_new_ui')
from inceptionresnetv2 import get_train_model_triplet


EPOCHS = 100
BATCH_SIZE = 56

callbacks = [
    keras.callbacks.ModelCheckpoint("train/check_point.h5", monitor='loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min'),
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

# train_ds = 

# num_class = len(train_ds.class_names)
num_class = len(os.listdir(data_dir))

train_ds = train_ds.prefetch(buffer_size=BATCH_SIZE)

# data_augmentation = keras.layers.RandomZoom(0.2)
# augmented_train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
normalization_layer = keras.layers.Rescaling(1./255)
normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

model = get_train_model_triplet()

loss = tfa.losses.TripletSemiHardLoss()
model.compile(loss=loss, optimizer='Adam')

history = model.fit(
    normalized_train_ds,
    callbacks = callbacks,
    # validation_data = normalized_val_ds,
)

model.save('train/final.h5')

results = model.predict(normalized_train_ds)

np.savetxt("vecs.tsv", results, delimiter='\t')

out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for img, labels in tfds.as_numpy(normalized_train_ds):
    [out_m.write(str(x) + "\n") for x in labels]
out_m.close()


epochs = [i for i in range(1, len(history.history['loss'])+1)]

plt.figure(1)
plt.plot(epochs, history.history['loss'], color='red', label="training_loss")
plt.legend(loc='best')
plt.title('training')
plt.xlabel('epoch')
plt.savefig('train/loss.png', bbox_inches='tight')
plt.show()