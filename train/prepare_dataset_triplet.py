import os
import sys
sys.path.append(r'D:\sw\face_rec\face_reg_new_ui')
from faceDetection import face_detector
import cv2
from winApp import get_face
import shutil


# for i in range(1, 201):
#     os.makedirs('D:/sw/face_rec/data/fei/fei_sort/{:04d}'.format(i))

root_dir = r'D:\sw\face_rec\data\fei\fei'
for f_name in os.listdir(root_dir):
    f_name_split = f_name.split('-')
    f_class = f_name_split[0]
    f_path = os.path.join(root_dir, f_name)
    save_path = 'D:/sw/face_rec/data/fei/fei_sort/{:04d}/{}'.format(int(f_class), f_name)
    shutil.copyfile(f_path, save_path)


data_dir = r'D:\sw\face_rec\data\fei\fei_sort'
out_dir1 = r'D:\sw\face_rec\data\fei\fei_aligned'
out_dir2 = r'D:\sw\face_rec\data\fei\fei_masked'
out_dir3 = r'D:\sw\face_rec\data\fei\fei_both'
try:
    os.makedirs(out_dir1)
    os.makedirs(out_dir2)
    os.makedirs(out_dir3)
except:
    pass
for i, fld_name in enumerate(os.listdir(data_dir)):
    print(i)
    fld_path = os.path.join(data_dir, fld_name)
    out_fld1 = os.path.join(out_dir1, fld_name)
    out_fld2 = os.path.join(out_dir2, fld_name)
    out_fld3 = os.path.join(out_dir3, fld_name)
    try:
        os.makedirs(out_fld1)
        os.makedirs(out_fld2)
        os.makedirs(out_fld3)
    except:
        pass
    for file_name in os.listdir(fld_path):
        file_path = os.path.join(fld_path, file_name)
        if file_path.lower().endswith(('jpg', 'jpeg', 'png')):
            img_name = file_name.split('.')[0]
            out_path1 = os.path.join(out_fld1, 'aligned_{}.jpg'.format(img_name))
            out_path2 = os.path.join(out_fld2, 'masked_{}.jpg'.format(img_name))
            out_path3_1 = os.path.join(out_fld3, 'aligned_{}.jpg'.format(img_name))
            out_path3_2 = os.path.join(out_fld3, 'masked_{}.jpg'.format(img_name))
            img = cv2.imread(file_path)
            face_locs, face_locs_margin = face_detector(img.copy())
            if len(face_locs) != 1:
                continue
            face_parts, face_angle, return_layer = get_face(img.copy(), face_locs[0], face_locs_margin[0])
            cv2.imwrite(out_path1, face_parts[0])
            cv2.imwrite(out_path2, face_parts[1])
            cv2.imwrite(out_path3_1, face_parts[0])
            cv2.imwrite(out_path3_2, face_parts[1])


# import shutil
# out_dir = data_dir = r'D:\sw\face_rec\data\CASIA-WebFace_mini'
# i = 0
# for fld_name in os.listdir(out_dir1):
#     print(i)
#     if i < 2000:
#         fld_path = os.path.join(out_dir1, fld_name)
#         if len(os.listdir(fld_path)) < 5:
#             continue
#         i += 1
#         out_fld_path = os.path.join(out_dir, fld_name)
#         os.makedirs(out_fld_path)
#         for j, f_name in enumerate(os.listdir(fld_path)):
#             if j < 5:
#                 f_path = os.path.join(fld_path, f_name)
#                 out_path = os.path.join(out_fld_path, f_name)
#                 shutil.copyfile(f_path, out_path)
