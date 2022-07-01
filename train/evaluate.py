import sys
sys.path.append(r'D:\sw\face_rec\face_reg_new_ui')
# import tensorflow as tf
# from inceptionresnetv2 import get_model
import numpy as np
# import keras
# import os
# import cv2


# model = get_model('storage\model/feature_extraction_model/weights.h5')


# data_dir = r"D:\sw\face_rec\data\CASIA-WebFace_aligned"

# embs = []
# labels = []
# for fld_name in os.listdir(data_dir):
#     fld_path = os.path.join(data_dir, fld_name)
#     for file_name in os.listdir(fld_path):
#         file_path = os.path.join(fld_path, file_name)
#         image = cv2.imread(file_path)
#         image = cv2.resize(image, (299,299))
#         image = image/255.0
#         image = np.expand_dims(image, axis=0)
#         emb=model.predict(image)
#         labels.append(fld_name)
#         embs.append(emb)
# np.savez('ds_casia.npz', labels=labels, embs=embs)

from sklearn import svm
from sklearn import model_selection
import pickle
import json


threshold = [0.0, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
results = {}

print('load embs')
ds = np.load('ds.npz', allow_pickle=True)
embs = ds['embs'][:,0]
labels = ds['labels']
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(embs, labels, test_size=0.2, random_state=23)
print('train svm')
clf = svm.SVC()
clf.fit(X_train, Y_train)
pickle.dump(clf, open('svm_lfw.sav', 'wb'))
print('predict test')
pred_class = []
pred_prob = []
for m in range(len(X_test)):
    p = np.array(clf.decision_function([X_test[m]]))
    prob = np.exp(p)/np.sum(np.exp(p),axis=1, keepdims=True)
    classes = clf.predict([X_test[m]])
    pred_prob.append(prob[0])
    pred_class.append(classes[0])
print('evaluate')
for i in threshold:
    results[i]['pos'] = 0
    results[i]['neg'] = 0
    results[i]['miss'] = 0
    for j in range(len(X_test)):
        if max(pred_prob[j]) >= i:
            if pred_class[i] == Y_test[i]:
                results[i]['pos'] += 1
            else:
                results[i]['neg'] += 1
        else:
            results[i]['miss'] += 1
    results[i]['acc'] = results[i]['pos']/(results[i]['pos']+results[i]['neg'])
    results[i]['pred_rate'] = results[i]['miss']/(results[i]['pos']+results[i]['neg']+results[i]['miss'])
print(results)
with open('result_lfw.json', 'w') as f:
    json.dump(results, f)

