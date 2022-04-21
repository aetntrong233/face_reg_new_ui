import numpy as np
import cv2


HALF_LOW_LEFT = [148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 227, 116, 117, 119, 47, 174, 196, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11 ,12, 13, 14, 15, 16, 17, 18, 200, 199, 175, 152]
HALF_LOW_RIGHT = [377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 447, 345, 346, 277, 399, 419, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11 ,12, 13, 14, 15, 16, 17, 18, 200, 199, 175, 152]
HALF_UP_LEFT = [109, 67, 103, 54, 21, 162, 127, 234, 227, 116, 117, 119, 47, 174, 196, 197, 6, 168, 8, 9, 151, 10]
HALF_UP_RIGHT = [338, 297, 332, 284, 251, 389, 356, 454, 447, 345, 346, 277, 399, 419, 197, 6, 168, 8, 9, 151, 10]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
FACE_HALF_RIGHT = [338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 175, 199, 200, 18, 17, 16, 15, 14, 13, 12, 11, 0, 164, 2, 94, 19, 1, 4, 5, 195, 197, 6, 168, 8, 9, 151, 10]
FACE_HALF_LEFT = [109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 175, 199, 200, 18, 17, 16, 15, 14, 13, 12, 11, 0, 164, 2, 94, 19, 1, 4, 5, 195, 197, 6, 168, 8, 9, 151, 10] 
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]

# # face_parts = ['base_img','face_part','hide_low_part','low_part','hide_up_part','up_part','eye_part']
# def face_divider(pixels, landmark):
#     face_parts = []
#     # base image
#     base_img = pixels.copy()
#     face_parts.append(base_img)
#     # face part
#     face_part = pixels.copy()
#     points = []
#     stencil = np.zeros(face_part.shape).astype(face_part.dtype)
#     for i in FACE_HALF_LEFT:
#         points.append([landmark[i][0],landmark[i][1]])
#     points = np.asarray(points)
#     cv2.fillPoly(stencil, [points], [255, 255, 255])
#     points = []
#     for i in FACE_HALF_RIGHT:
#         points.append([landmark[i][0],landmark[i][1]])
#     points = np.asarray(points)
#     cv2.fillPoly(stencil, [points], [255, 255, 255])
#     face_part = cv2.bitwise_and(face_part, stencil)
#     face_parts.append(face_part)
#     # delete half low face + low face part
#     hide_low_part = pixels.copy()
#     low_face_part = pixels.copy()
#     stencil = np.zeros(hide_low_part.shape).astype(hide_low_part.dtype)
#     points = []
#     for i in HALF_LOW_LEFT:
#         points.append([landmark[i][0],landmark[i][1]])
#     points = np.asarray(points)
#     cv2.fillPoly(hide_low_part, [points], [0,0,0])
#     cv2.fillPoly(stencil, [points], [255, 255, 255])
#     points = []
#     for i in HALF_LOW_RIGHT:
#         points.append([landmark[i][0],landmark[i][1]])
#     points = np.asarray(points)
#     cv2.fillPoly(hide_low_part, [points], [0,0,0])
#     cv2.fillPoly(stencil, [points], [255, 255, 255])
#     face_parts.append(hide_low_part)
#     low_face_part = cv2.bitwise_and(low_face_part, stencil)
#     face_parts.append(low_face_part)
#     # delete half up face + up face part
#     hide_up_part = pixels.copy()
#     up_face_part = pixels.copy()
#     stencil = np.zeros(hide_up_part.shape).astype(hide_up_part.dtype)
#     points = []
#     for i in HALF_UP_LEFT:
#         points.append([landmark[i][0],landmark[i][1]])
#     points = np.asarray(points)
#     cv2.fillPoly(hide_up_part, [points], [0,0,0])
#     cv2.fillPoly(stencil, [points], [255, 255, 255])
#     points = []
#     for i in HALF_UP_RIGHT:
#         points.append([landmark[i][0],landmark[i][1]])
#     points = np.asarray(points)
#     cv2.fillPoly(hide_up_part, [points], [0,0,0])
#     cv2.fillPoly(stencil, [points], [255, 255, 255])
#     face_parts.append(hide_up_part)
#     up_face_part = cv2.bitwise_and(up_face_part, stencil)
#     face_parts.append(up_face_part)
#     # eyes part
#     # left eye part
#     eye_part = pixels.copy()
#     stencil = np.zeros(eye_part.shape).astype(eye_part.dtype)
#     points = []
#     for i in LEFT_EYE:
#         points.append([landmark[i][0],landmark[i][1]])
#     points = np.asarray(points)
#     cv2.fillPoly(stencil, [points], [255, 255, 255])
#     # right eye part
#     points = []
#     for i in RIGHT_EYE:
#         points.append([landmark[i][0],landmark[i][1]])
#     points = np.asarray(points)
#     cv2.fillPoly(stencil, [points], [255, 255, 255])
#     eye_part = cv2.bitwise_and(eye_part, stencil)
#     face_parts.append(eye_part)
#     return face_parts


face_parts = ['base_img','hide_low_part']
def face_divider(pixels, landmark):
    face_parts = []
    # base image
    base_img = pixels.copy()
    face_parts.append(base_img)
    # delete half low face
    hide_low_part = pixels.copy()
    stencil = np.zeros(hide_low_part.shape).astype(hide_low_part.dtype)
    points = []
    for i in HALF_LOW_LEFT:
        points.append([landmark[i][0],landmark[i][1]])
    points = np.asarray(points)
    cv2.fillPoly(hide_low_part, [points], [0,0,0])
    cv2.fillPoly(stencil, [points], [255, 255, 255])
    points = []
    for i in HALF_LOW_RIGHT:
        points.append([landmark[i][0],landmark[i][1]])
    points = np.asarray(points)
    cv2.fillPoly(hide_low_part, [points], [0,0,0])
    cv2.fillPoly(stencil, [points], [255, 255, 255])
    face_parts.append(hide_low_part)
    return face_parts