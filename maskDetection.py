from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import load_model
import cv2
import numpy as np

mask_net = load_model('storage/model/mask_detection_model/mask_detector.model')

def mask_detector(pixels):
    img = pixels.astype('float32')
    img = cv2.resize(img,(224,224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    pred = mask_net.predict(img)
    (mask, without_mask) = pred[0]
    if mask >= without_mask:
        is_masked = True
    else:
        is_masked = False
    return is_masked, (mask, without_mask)



# a=cv2.imread(r'C:\Trong\python\rasp_test\ds_face\002\100.png')
# print(mask_detector(a))