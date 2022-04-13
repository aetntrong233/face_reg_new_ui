import numpy as np
import cv2


HALF_LOW_LEFT_FACE_OVAL = [148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 227, 116, 117, 119, 47, 174, 196, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11 ,12, 13, 14, 15, 16, 17, 18, 200, 199, 175, 152]
HALF_LOW_RIGHT_FACE_OVAL = [377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 447, 345, 346, 277, 399, 419, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11 ,12, 13, 14, 15, 16, 17, 18, 200, 199, 175, 152]


def face_divider(pixels, landmark):
    del_mask_img = pixels.copy()
    points = []
    for i in HALF_LOW_LEFT_FACE_OVAL:
        points.append([landmark[i][0],landmark[i][1]])
    points = np.asarray(points)
    cv2.drawContours(del_mask_img,[points],0,(0,0,0),2)
    cv2.fillPoly(del_mask_img, [points], [0,0,0])
    points = []
    for i in HALF_LOW_RIGHT_FACE_OVAL:
        points.append([landmark[i][0],landmark[i][1]])
    points = np.asarray(points)
    cv2.drawContours(del_mask_img,[points],0,(0,0,0),2)
    cv2.fillPoly(del_mask_img, [points], [0,0,0])
    return del_mask_img