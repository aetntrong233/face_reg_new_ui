# references:
#   https://google.github.io/mediapipe/solutions/models.html


import tensorflow as tf
import gdown
import os
import time
import cv2
import numpy as np


# kiểm tra và load model landmark
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


# summary: xác định tọa độ landmark
# params:
# 	init
# 		pixels: ảnh (array)
# 	return
# 		landmark: 468 3D landmarks flattened into a 1D tensor: (x1, y1, z1), (x2, y2, z2), ...
#       score: khả năng xuất hiện của khuôn mặt trong ảnh
def get_landmark(pixels):
    image = pixels
    # chuẩn hóa ảnh ngõ vào
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    factor = min(height/pixels.shape[0],width/pixels.shape[1])
    interpolation = cv2.INTER_CUBIC if factor > 1 else cv2.INTER_AREA
    image_resized = cv2.resize(image_rgb, (width, height), interpolation=interpolation)
    # image_array = ((image_resized-127.5)/127.5).astype('float32')
    image_array = (image_resized/255.0).astype('float32')
    input_data = np.expand_dims(image_array, axis=0)
    # xác định tọa độ landmark
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    landmark = interpreter.get_tensor(landmark_index)[0].reshape(468, 3) / 192
    score = interpreter.get_tensor(score_index)[0]
    return landmark, score


# from faceAngle import get_face_angle
# image = cv2.imread(r'C:\Users\TrongTN\Downloads\musk.jpg')
# loc = (312, 67, 245, 245)
# face = image.copy()[loc[1]:loc[1]+loc[3],loc[0]:loc[0]+loc[2]]
# landmark = get_landmark(face)[0][:, [0, 1]]
# landmark_ = []
# for point in landmark:
#     point_x = int(point[0]*245)
#     point_y = int(point[1]*245)
#     face = cv2.circle(face, (point_x,point_y), 1, (0, 0, 255), 1)
#     landmark_.append((point_x,point_y))
# angle = get_face_angle(landmark_)
# def rotate_image(image, angle):
#     image_center = tuple(np.array(image.shape[1::-1]) / 2)
#     rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
#     result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
#     return result
# new_image = rotate_image(image.copy(), angle[0])
# from faceDetection import face_detector
# loc = face_detector(new_image)[1][0]
# new_face = new_image.copy()[loc[1]:loc[1]+loc[3],loc[0]:loc[0]+loc[2]]
# new_landmark = get_landmark(new_face)[0]
# new_landmark_ = []
# (x,y,w,h) = loc
# for point in new_landmark:
#     point_x = int(x+point[0]*new_face.shape[1])
#     point_y = int(y+point[1]*new_face.shape[0])
#     point_z = int(y+point[2]*new_face.shape[1])
#     new_landmark_.append((point_x,point_y,point_z))
# from faceDivider import face_divider
# face_parts = face_divider(new_image, new_landmark_, loc)
# cv2.imwrite(r'C:\Users\TrongTN\Downloads\musk_remove_mask.png', face_parts[2])
# # cv2.imwrite(r'C:\Users\TrongTN\Downloads\musk_rotate.png', new_image)
# # cv2.imwrite(r'C:\Users\TrongTN\Downloads\musk_face_rotate.png', new_face)