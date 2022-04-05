import os
import cv2
import numpy as np
import time
import tensorflow as tf
import torch
from keras.models import load_model
import gdown


def nms(P: torch.tensor, thresh_iou: float):
    x1 = P[:, 0]
    y1 = P[:, 1]
    x2 = P[:, 2]
    y2 = P[:, 3]
    scores = P[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()
    keep = []
    while len(order) > 0:
        idx = order[-1]
        keep.append(P[idx])
        order = order[:-1]
        xx1 = torch.index_select(x1, dim=0, index=order)
        xx2 = torch.index_select(x2, dim=0, index=order)
        yy1 = torch.index_select(y1, dim=0, index=order)
        yy2 = torch.index_select(y2, dim=0, index=order)
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])
        w = xx2 - xx1
        h = yy2 - yy1
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        rem_areas = torch.index_select(areas, dim = 0, index = order)
        union = (rem_areas - inter) + areas[idx]
        IoU = inter / union
        mask = IoU < thresh_iou
        order = order[mask]
    return keep

face_detect_model_path = r'storage/model/face_detection_model/face_detection_short_range.tflite'
face_detect_model_url = 'https://drive.google.com/u/0/uc?id=1U6s_ayAMMPo1HArdwgfbJrXSXBHloeiQ&export=download'
if not os.path.exists('storage/model/face_detection_model/'):
    os.makedirs('storage/model/face_detection_model/')
if os.path.isfile(face_detect_model_path) != True:
		print("face_detection_short_range.tflite will be downloaded...")
		gdown.download(url=face_detect_model_url, output=face_detect_model_path, quiet=False)
interpreter = tf.lite.Interpreter(model_path=face_detect_model_path)
interpreter.allocate_tensors()
# mask_detect_model_path = r'storage/model/extended_model/mask_detector.h5'
# mask_detect_model_url = 'https://drive.google.com/u/0/uc?id=1AKn_xhcLf2A4xcEzM9hYY844ZnlFTbwx&export=download'
# if os.path.isfile(mask_detect_model_path) != True:
# 		print("mask_detector.h5 will be downloaded...")
# 		gdown.download(url=mask_detect_model_url, output=mask_detect_model_path, quiet=False)
# mask_detector = load_model(mask_detect_model_path)

# (reference: modules/face_detection/face_detection_short_range_common.pbtxt)
SSD_OPTIONS_SHORT = {
    'num_layers': 4,
    'input_size_height': 128,
    'input_size_width': 128,
    'anchor_offset_x': 0.5,
    'anchor_offset_y': 0.5,
    'strides': [8, 16, 16, 16],
    'interpolated_scale_aspect_ratio': 1.0
}

def ssd_generate_anchors(opts: dict) -> np.ndarray:
    """This is a trimmed down version of the C++ code; all irrelevant parts
    have been removed.
    (reference: mediapipe/calculators/tflite/ssd_anchors_calculator.cc)
    """
    layer_id = 0
    num_layers = opts['num_layers']
    strides = opts['strides']
    assert len(strides) == num_layers
    input_height = opts['input_size_height']
    input_width = opts['input_size_width']
    anchor_offset_x = opts['anchor_offset_x']
    anchor_offset_y = opts['anchor_offset_y']
    interpolated_scale_aspect_ratio = opts['interpolated_scale_aspect_ratio']
    anchors = []
    while layer_id < num_layers:
        last_same_stride_layer = layer_id
        repeats = 0
        while (last_same_stride_layer < num_layers and
               strides[last_same_stride_layer] == strides[layer_id]):
            last_same_stride_layer += 1
            # aspect_ratios are added twice per iteration
            repeats += 2 if interpolated_scale_aspect_ratio == 1.0 else 1
        stride = strides[layer_id]
        feature_map_height = input_height // stride
        feature_map_width = input_width // stride
        for y in range(feature_map_height):
            y_center = (y + anchor_offset_y) / feature_map_height
            for x in range(feature_map_width):
                x_center = (x + anchor_offset_x) / feature_map_width
                for _ in range(repeats):
                    anchors.append((x_center, y_center))
        layer_id = last_same_stride_layer
    return np.array(anchors, dtype=np.float32)

input_index = interpreter.get_input_details()[0]['index']
input_shape = interpreter.get_input_details()[0]['shape']
bbox_index = interpreter.get_output_details()[0]['index']
score_index = interpreter.get_output_details()[1]['index']
anchors = ssd_generate_anchors(SSD_OPTIONS_SHORT)
RAW_SCORE_LIMIT = 80
height, width = input_shape[1:3]
MIN_SCORE = 0.65
MIN_FACE_SIZE = 200
REQUIRE_SIZE = 224

def face_detector(pixels):
    t1_start = time.process_time()
    # pixels = cv2.resize(pixels,(1000,1000),interpolation=cv2.INTER_CUBIC)
    image = pixels
    base_width, base_height = pixels.shape[1], pixels.shape[0]
    # image = cv2.resize(image,(300,300),interpolation=cv2.INTER_CUBIC)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height))
    image_array = (image_resized/255.).astype('float32')
    input_data = np.expand_dims(image_array, axis=0)
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    raw_boxes = interpreter.get_tensor(bbox_index)
    raw_scores = interpreter.get_tensor(score_index)
    boxes = decode_boxes(raw_boxes)
    scores = get_sigmoid_scores(raw_scores)

    def is_valid(box: np.ndarray) -> bool:
        return np.all(box[1] > box[0])

    score_above_threshold = scores > MIN_SCORE
    filtered_boxes = boxes[np.argwhere(score_above_threshold)[:, 1], :]
    filtered_scores = scores[score_above_threshold]
    if not filtered_scores.any():
        return None, None
    bboxes = []
    for box, score in zip(filtered_boxes, filtered_scores):
        if is_valid(box):
            data = box.reshape(-1, 2)
            xmin, ymin = data[0]
            xmax, ymax = data[1]
            bboxes.append([xmin, ymin, xmax, ymax, score])
    P = torch.tensor(bboxes)
    bb = nms(P, 0.5)
    faces = []
    faces_location = []
    for i,b in enumerate(bb):
        xmin = int(max(1,(bb[i][0] * base_width)))
        ymin = int(max(1,(bb[i][1] * base_height)))
        xmax = int(min(base_width,(bb[i][2] * base_width)))
        ymax = int(min(base_height,(bb[i][3] * base_height)))
        bb_width = xmax-xmin
        bb_height = ymax-ymin
        offset_x = 0
        offset_y = 0
        if bb_width > bb_height:
            offset_y = round((bb_width - bb_height)/2)
            bb_height = bb_width
        elif bb_width < bb_height:
            offset_x = round((bb_height - bb_width)/2)
            bb_width = bb_height
        if (bb_height>=MIN_FACE_SIZE) and (bb_width>=MIN_FACE_SIZE) and ((ymin-offset_y)>=0) and ((ymin-offset_y+bb_height)<=base_height) and ((xmin-offset_x)>=0) and ((xmin-offset_x+bb_width)<=base_width):
            face = pixels[ymin-offset_y:ymin-offset_y+bb_height,xmin-offset_x:xmin-offset_x+bb_width]
            face = cv2.resize(face, (REQUIRE_SIZE, REQUIRE_SIZE))
            faces.append(face)
            faces_location.append((xmin-offset_x,ymin-offset_y,bb_width,bb_height))
    # if faces_location == []:
    #     print("No face detected")
    t1_stop = time.process_time()
    # print("Detect face(s) time: " + str(t1_stop-t1_start))
    return faces, faces_location

def decode_boxes(raw_boxes: np.ndarray) -> np.ndarray:
    """Simplified version of
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
    """
    # width == height so scale is the same across the board
    scale = height
    num_points = raw_boxes.shape[-1] // 2
    # scale all values (applies to positions, width, and height alike)
    boxes = raw_boxes.reshape(-1, num_points, 2) / scale
    # adjust center coordinates and key points to anchor positions
    boxes[:, 0] += anchors
    for i in range(2, num_points):
        boxes[:, i] += anchors
    # convert x_center, y_center, w, h to xmin, ymin, xmax, ymax
    center = np.array(boxes[:, 0])
    half_size = boxes[:, 1] / 2
    boxes[:, 0] = center - half_size
    boxes[:, 1] = center + half_size
    return boxes

def get_sigmoid_scores(raw_scores: np.ndarray) -> np.ndarray:
    """Extracted loop from ProcessCPU (line 327) in
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
    """
    # just a single class ("face"), which simplifies this a lot
    # 1) thresholding; adjusted from 100 to 80, since sigmoid of [-]100
    #    causes overflow with IEEE single precision floats (max ~10e38)
    raw_scores[raw_scores < -RAW_SCORE_LIMIT] = -RAW_SCORE_LIMIT
    raw_scores[raw_scores > RAW_SCORE_LIMIT] = RAW_SCORE_LIMIT
    # 2) apply sigmoid function on clipped confidence scores
    return sigmoid(raw_scores)

def sigmoid(data: np.ndarray) -> np.ndarray:
    """Return sigmoid activation of the given data
    Args:
        data (ndarray): Numpy array containing data
    Returns:
        (ndarray) Sigmoid activation of the data with element range (0,1]
    """
    return 1 / (1 + np.exp(-data))


# import cv2

# x = cv2.imread(r'C:/Trong/python/st/data/A.Long/long1.jpg')
# a,b = face_detector(x)
# cv2.imshow('a',a[0])
# cv2.waitKey()