from evaluate_on_lfw import lfw
from utils.featureExtraction import feature_extraction
import cv2
import numpy as np


pair_filename_path = 'dataset/lfw/pairs.txt'
lfw_dir = 'dataset\\lfw\\lfw'
thresholds = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

pairs = lfw.read_pairs(pair_filename_path)
paths, actual_issame = lfw.get_paths(lfw_dir, pairs)

print('Embedding on LFW images')
embeddings = []
for i in range(len(paths)):
    if i % 2 == 0: 
        img0 = cv2.imread(paths[i])
        embedding0 = feature_extraction(img0)
        img1 = cv2.imread(paths[i+1])
        embedding1 = feature_extraction(img1)
        embeddings += (embedding0, embedding1)

np.save('dataset/lfw/embedding.npy', embeddings)
# embeddings = np.load('dataset/lfw/embedding.npy')

for threshold in thresholds:
    print('---------------------------------------')

    accuracy, precision, recall = lfw.evaluate(embeddings, actual_issame, threshold)

    f1_score = 0 if (precision+recall==0) else 2*precision*recall/(precision+recall)
        
    print('Threshold: %3.5f' % threshold)

    print('Accuracy: %3.5f' % accuracy)
    print('Precision: %3.5f' % precision)
    print('Recall: %3.5f' % recall)
    print('F1 score: %3.5f' % f1_score)