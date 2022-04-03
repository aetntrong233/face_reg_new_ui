import tensorflow as tf
import gdown
import os
import time
import cv2
import numpy as np


if not os.path.exists('storage/model/landmark_model'):
    os.makedirs('storage/model/landmark_model')
landmark_model_path = r'storage/model/landmark_model/face_landmark.tflite'
landmark_model_url = 'https://drive.google.com/u/1/uc?id=1mtuMCbn2RjkMdzx94lD88jmWT9tniu-P&export=download'
if os.path.isfile(landmark_model_path) != True:
		print("face_landmark.tflite will be downloaded...")
		gdown.download(url=landmark_model_url, output=landmark_model_path, quiet=False)
interpreter = tf.lite.Interpreter(model_path=landmark_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]['index']
input_shape = input_details[0]['shape']
landmark_index = output_details[0]['index']
score_index = output_details[1]['index']

height, width = input_shape[1:3]


def get_landmark(pixels):
    image = pixels
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height))
    # image_array = ((image_resized-127.5)/127.5).astype('float32')
    image_array = (image_resized/255.0).astype('float32')
    input_data = np.expand_dims(image_array, axis=0)
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    landmark = interpreter.get_tensor(landmark_index)[0].reshape(468, 3) / 192
    score = interpreter.get_tensor(score_index)[0]
    return landmark, score

def add_mask_2_face(pixels):
    base_width, base_height = pixels.shape[1], pixels.shape[0]
    landmark, score = get_landmark(pixels)
    left_x, left_y = int(landmark[234][0]*base_width), int(landmark[234][1]*base_height)
    right_x, right_y = int(landmark[454][0]*base_width), int(landmark[454][1]*base_height)
    bottom_x, bottom_y = int(landmark[152][0]*base_width), int(landmark[152][1]*base_height)
    mask_w = abs(left_x-right_x)
    mask_h = max(abs(left_y-bottom_y),abs(right_y-bottom_y))
    mask = cv2.imread('storage/masks/blue-mask.png')
    mask_resized = cv2.resize(mask, (mask_w,mask_h))
    y1=min(left_y,right_y)
    y2=bottom_y
    x1=left_x
    x2=right_x
    # pixels[y1:y2, x1:x2] = mask_resized  
    alpha_s = mask_resized / 255.0
    alpha_l = 1.0 - alpha_s
    pixels[y1:y2, x1:x2] = (alpha_s * mask_resized[:, :] + alpha_l * pixels[y1:y2, x1:x2])
    face_masked =  pixels
    return face_masked


# img = cv2.imread(r'storage/imageBase/29-03-22-17-31-49/img-29-03-22-17-31-57.jpg')
# img = cv2.resize(img,(200,200))
# a = add_mask_2_face(img)
# # for point in a:
# #     x1 = int(point[0]*img.shape[1])
# #     y1 = int(point[1]*img.shape[0])
# #     cv2.circle(img, (x1, y1),2, (0, 255, 128), -1)
# # # print(a,b)
# cv2.imshow('x',a)
# cv2.waitKey()