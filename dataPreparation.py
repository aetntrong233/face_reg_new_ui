from keras import Sequential, layers
import numpy as np
import os
import tensorflow as tf
import PIL
import shutil
from  addMask2Face import add_mask_2_face
import cv2


imageBase_dir = r'storage/imageBase'
imageTrain_dir = r'storage/imageTrain'
if not os.path.exists(imageBase_dir):
    os.makedirs(imageBase_dir)
if not os.path.exists(imageTrain_dir):
    os.makedirs(imageTrain_dir)


def data_train_enriched_mask():   
    data_augmentation = Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.1),])
    image_ds = []
    fld_list = []
    for label in os.listdir(imageBase_dir):
        sub_dir = os.path.join(imageBase_dir,label)
        for j, image in enumerate(os.listdir(sub_dir)):
            image_dir = os.path.join(sub_dir,image)
            base_image = cv2.imread(image_dir)
            # base_image = cv2.resize(base_image, (224, 224))
            # cv2.imshow('x',base_image)
            # cv2.waitKey()
            image_ds.append(base_image)
            fld_list.append(label)
            base_image_mask = base_image.copy()
            base_image_mask = add_mask_2_face(base_image_mask)
            image_ds.append(base_image_mask)
            fld_list.append(label)
            for i in range(1, 10):
                augmented_image = data_augmentation(tf.expand_dims(base_image, 0), training=True)
                augmented_image = np.array(augmented_image[0]).astype('uint8')
                image_ds.append(augmented_image)
                fld_list.append(label)
                # augmented_image_mask = data_augmentation(tf.expand_dims(base_image_mask, 0), training=True)
                # augmented_image_mask = np.array(augmented_image_mask[0]).astype('uint8')
                # image_ds.append(augmented_image_mask)
                # fld_list.append(label)
    return image_ds, fld_list

# x,y=data_train_enriched_mask()
# for i in x:
#     cv2.imshow('x',i)
#     cv2.waitKey()
