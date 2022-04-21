import os
import cv2
import numpy as np
import tensorflow as tf
import torch
from keras.models import load_model
import gdown


MIN_SCORE = 0.5
MIN_FACE_SIZE = 100
REQUIRE_SIZE = 224


prototxtPath = 'storage/model/face_detection_model/deploy.prototxt'
weightsPath = 'storage/model/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel'
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)


# load caffe model with cv2
def face_detector_caffe(pixels):
    blob = cv2.dnn.blobFromImage(pixels, 1.0, (224, 224),(104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    locs = []
    scores = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > MIN_SCORE:
            box = detections[0, 0, i, 3:7]
            (xmin, ymin, xmax, ymax) = box
            locs.append((xmin, ymin, xmax, ymax))
            scores.append(confidence)
    return locs, scores


# detect face using ssd resnet caffe model
def face_detector(pixels):
    image = pixels
    base_width, base_height = pixels.shape[1], pixels.shape[0]
    bboxes, scores=face_detector_caffe(image)
    faces = []
    faces_location = []
    for box in bboxes:
        xmin = int(max(1,(box[0] * base_width)))
        ymin = int(max(1,(box[1] * base_height)))
        xmax = int(min(base_width,(box[2] * base_width)))
        ymax = int(min(base_height,(box[3] * base_height)))
        bb_width = xmax-xmin
        bb_height = ymax-ymin
        offset_x = 0
        offset_y = 0
        if bb_width > bb_height:
            offset_y = int((bb_width - bb_height)/2)
            bb_height = bb_width
        elif bb_width < bb_height:
            offset_x = int((bb_height - bb_width)/2)
            bb_width = bb_height
        margin_x = int(bb_width*0.25)
        margin_y = int(bb_height*0.25)
        offset_x = int(offset_x+margin_x/2)
        offset_y = int(offset_y+margin_y/2)
        bb_width += margin_x
        bb_height += margin_y
        if (bb_height>=MIN_FACE_SIZE) and (bb_width>=MIN_FACE_SIZE) and ((ymin-offset_y)>=0) and ((ymin-offset_y+bb_height)<=base_height) and ((xmin-offset_x)>=0) and ((xmin-offset_x+bb_width)<=base_width):
            face = pixels[ymin-offset_y:ymin-offset_y+bb_height,xmin-offset_x:xmin-offset_x+bb_width]
            face = cv2.resize(face, (REQUIRE_SIZE, REQUIRE_SIZE))
            faces.append(face)
            faces_location.append((xmin-offset_x,ymin-offset_y,bb_width,bb_height))
    return faces, faces_location
